# Sage local-cache producer/consumer pattern map — for the standalone speech-redaction plugin

Reference material for refactoring the birdnet fork's microphone-path speech
redaction into a standalone Sage plugin that **consumes audio from the
media-sampler cache** and **produces a redacted audio product back into the
cache** for a downstream plugin (e.g. BirdNET) to classify.

Distilled from:
- `docs/local-cache-design.md` (the two-layer /local-cache proposal)
- `skills/sage-waggle/references/local-cache-ring-buffer.md` (image-sampler2 ring)
- `skills/sage-waggle/references/continuous-producer-patterns.md` (dual-grid loop + from-cache uploader)
- `skills/sage-waggle/references/crop-producer-detect-classify-cascade.md` (sage-yolo2 crop producer)
- `skills/sage-waggle/references/detect-then-crop-then-classify-cascade.md` (sage-yolo2 → sage-bioclip2)
- `skills/sage-waggle/references/frame-anchored-batch-consumers-and-watchers.md` (the capture-ts trap)
- the birdnet fork at `~/AI-Projects/birdnet` (app.py, sage.yaml, jobs/, ecr-meta/, redaction/)

The proven cache pattern today lives in the **vision** family
(image-sampler2 → sage-yolo2 → sage-bioclip2). The birdnet fork is the case
study for the **redaction** half and the Sage deployment plumbing, but its
cache-consume/produce half for audio is the planned refactor, not a built one
(see §6 for the gap).

---

## 1. What the cache looks like on disk

### Root and mount

Shared **hostPath** tree mounted into multiple pods, so a producer's files are
visible to a consumer pod running as a **different uid**.

- Node host path: `/media/plugin-data/local-cache`
- Pod mount:     `/local-cache`
- Provisioned by the same WES ansible/node-setup that already creates
  `/media/plugin-data/uploads`.

This mirrors the already-proven `/uploads` pattern (the `wes-upload-agent`
DaemonSet does the same shape on `/media/plugin-data/uploads`). The ONLY
structural difference: the upload-agent's lifecycle rule is "drain upstream,
then delete" — a cache is **never uploaded**; its lifecycle rule is "evict by
size/age." Same shape, different reaping predicate.

### Why NOT /tmp

`/tmp` inside a pod is the container's own writable layer (or an emptyDir):
**pod-ephemeral** (vanishes on pod exit) AND **pod-private** (no other pod can
see it). A consumer pod literally cannot read a producer's frames written
there. The design doc is blunt: writing to `/tmp` "lands in a black hole" —
you produce but nothing can consume. The architecture is inert in production.

Pete's final call on image-sampler2: **delete the `/tmp` fallback entirely.**
`resolve_cache_root()` is a pure one-liner: `--cache-root` >
`$IS2_CACHE_ROOT` > `/local-cache`. No filesystem probe, no `/tmp` branch. A
fallback that silently "works" but produces data no consumer can read is worse
for a teaching/production audience than a clean error. When a requirement
becomes universal, the opt-in flag to request it is REDUNDANT — delete it.

### Layout (filenames)

Per the design, and as image-sampler2 / sage-yolo2 / sage-bioclip2 actually
implement it:

```
/local-cache/<namespace>/<plugin>/<camera>/<capture_ts_ns>-v2-<vsn>-<camera>.jpg
```

For crop/cascade streams the middle segment becomes an instance label /
cache-name, and per-detection sub-streams are leaf dirs:

```
/local-cache/<crop-cache-name>/<camera>-crops/<camera>-crop-<idx>/
```

Two properties of the filename are load-bearing:

- **`capture_ts_ns` prefix.** Selection by time works, and "oldest" eviction
  is by the filename prefix, NOT mtime (mtime lies if the clock stepped).
  Fallback to mtime only for non-matching files; leave unknown files
  untouched + uncounted.
- **`v2` suffix plus a sha1/SHA256 `unique_id` baked into embedded EXIF
  UserComment JSON**, so names never collide across plugins and a consumer
  can dedup. v2 filename shape: `<capture_ts_ns>-v2-<vsn>-<camera>.jpg`.

### Two eviction layers

The whole point of the design — two genuinely different concerns owned by
different parties:

| | Layer 1 — POLICY | Layer 2 — QUOTA |
|---|---|---|
| Question | What to keep / drop, in what order | Don't let anyone eat the disk |
| Nature | Semantic, data-aware | Blunt, semantics-free |
| Examples | keep N images; keep M MB; LRU; keep-last-per-camera | hard byte ceiling per plugin subdir + per node |
| Fires | continuously, as the plugin writes | only when a plugin EXCEEDS its cap (misbehavior) |
| Owner | the PLUGIN (via pywaggle2 cache primitive) | a WES node service (managing pod) |
| Repo | `pywaggle` (+ each plugin's config) | `waggle-edge-stack` |

**Layer 1 — plugin-owned graceful ring.** The plugin knows its data's meaning,
so it evicts by count/MB on every write (image-sampler2's `cache.py`:
`keep_max_count` + `keep_max_mb`, evict-on-either). Runs
`scan → plan → atomic tmp → fsync → os.replace` so a torn file never appears
under the final name and the ring never transiently exceeds caps. Candidate
pywaggle2 primitive:

```python
# producer
p.cache_file(path_or_bytes, name=..., timestamp=capture_ts,
             keep_max_count=500, keep_max_mb=2048)   # graceful ring, plugin-owned
# consumer
p.read_cache(name=..., select="newest")              # or closest-before/after ts
```

**Layer 2 — WES hard-quota backstop.** A `wes-local-cache-manager` DaemonSet
modeled almost line-for-line on `wes-upload-agent`:
- Mounts shared cache ROOT `hostPath: /media/plugin-data/local-cache` →
  `/local-cache` (parent of all plugin subdirs), exactly as upload-agent
  mounts the uploads parent.
- `priorityClassName: system-node-critical` + `node.kubernetes.io/disk-pressure`
  toleration, so it keeps reclaiming under pressure.
- On a ~60s sweep, walks each `/local-cache/<namespace>/<plugin>/` subdir,
  enforces a **per-subdir hard byte cap** (isolation: one plugin can't evict
  another's data) and a **per-node total cap** (outer bound). When exceeded,
  deletes **oldest-first** (by capture-ts filename prefix). Deliberately
  semantics-free: may throw away data the plugin considered important —
  acceptable precisely because it only fires on a plugin that has ALREADY
  blown past its allocation. A well-behaved plugin's Layer-1 ring keeps it far
  below the cap, so Layer 2 never touches it.

Why a purpose-built sweeper and not a k8s-native storage guard: hostPath
writes **escape all k8s storage accounting**. kubelet's ephemeral-storage
accounting does NOT count hostPath volumes; `emptyDir sizeLimit` is likewise
irrelevant. So the shared, persistent properties we NEED are exactly the
properties that make the cache escape every k8s-native storage guard. A
quota (Layer 2) is the sole mechanism that can bound `/local-cache`.

### The uploads-mount warning — NEVER put a local-only ring under /uploads

Resolved empirically by reading the upload-agent source
(`waggle-sensor/wes-upload-agent`, `main.sh` — it's a bash rsync loop). The
agent loops forever over `/uploads` (host `/media/plugin-data/uploads`),
running `find . -mindepth 3 -maxdepth 4 -type d` filtered to paths shaped like
`[<job>/]<plugin>/<version>/<ts>-<sha1hex>/` where `<version>` matches
`x.y.z | vx.y.z | latest | test` and the leaf matches `<digits>-<hexdigits>`
(the pywaggle staging dir holding `{data,meta}`). For every match it
`rsync ... --remove-source-files` to beehive — it **UPLOADS AND THEN DELETES
the source**, and `rmdir`s emptied dirs.

CONSEQUENCE for a LOCAL-ONLY ring: writing under the uploads mount is
**doubly wrong** — the agent would (i) upload files you must never upload,
and (ii) DELETE your ring files out from under your own eviction logic. Even
if your flat filename does not match the agent's regex TODAY, relying on
out-guessing the agent's scan is fragile.

**Cache home decision: always a dedicated subtree OUTSIDE the uploads mount.
Never point a local-only cache at `/run/waggle/uploads`.**

### Key operational rules

- **Ring size ≥ consumer lookback.** A consumer wanting frames from T-10s can
  lose them if the ring already evicted under load. Size the cache to exceed
  the LONGEST consumer lookback window. Document it.
- **Per-stream rings, not one shared dir.** Each stream is its own process; a
  shared cache dir → cross-process eviction RACE. Each stream gets its own
  subdir, its own independent ring with per-stream caps. No locks, race-free.
- **Stateless management.** Scan the subdir each capture; compute; decide.
  No authoritative in-memory ring state → crash/restart just re-scans.
- **Startup adoption.** Adopt existing name-matching files into the ring
  (count, size, evict). Never wipe the dir at startup; ignore non-matching
  files.
- **Evict BEFORE the file joins the ring.** Acquire into a `.tmp` (not a ring
  member); scan → current count/bytes; E3 guard (drop a single oversized new
  file rather than let it blow the cap); evict-LOOP oldest while
  `count+1 > max_count` OR `bytes+new_bytes > max_bytes`; atomic write
  (fsync `.tmp` then `os.replace(.tmp → final name)`).
- **Fail-SOFT at runtime, fail-FAST at config.** Eviction-delete failure or
  disk-full → WARN + skip/continue; never crash a long-running process.
  Missing/unwritable cache-dir, no cap set, flags misused → fail-fast at
  startup.

---

## 2. What calls consume and produce

### Producer side (image-sampler2 is the reference implementation)

These are the calls being hoisted into pywaggle2 as `cache_file()` /
`read_cache()`:

- `resolve_cache_root()` — pure: `--cache-root` > `$IS2_CACHE_ROOT` >
  `/local-cache`. No filesystem probe, no `/tmp` branch.
- `assert_cache_root_available(root)` — fail-fast at config if the root isn't
  a writable dir. The message names the `wes-local-cache-manager` component
  and the `-v <host>:/local-cache` mount fix.
- `scan_ring(dir)` → current_count, current_bytes, oldest-first (by
  capture_ts prefix). The SINGLE definition of "what is a valid managed v2
  file" — ignores `.tmp` and non-v2.
- `plan_evictions(...)` → which oldest files to drop so
  `count+1 <= max_count` and `bytes+new_bytes <= max_bytes`, with an E3 guard
  that drops an oversized new frame rather than letting one file blow the cap.
- `commit_capture(.tmp → final name)` — fsync then `os.replace` (atomic).
- **Liveness heartbeat.** A continuous producer NEVER uploads, so there is no
  upload record to imply "alive." It publishes
  `env.<plugin>.cache.{count,bytes,written,evicted,last_status}` on its own
  ~60s grid, with `meta={cache_name,camera,vsn}` all strings. Fires even when
  every capture FAILS — that "running but silent" case is the whole reason it
  exists. Dual-grid loop (capture grid + heartbeat grid on ONE thread): sleep
  to the NEAREST of (next capture edge, next heartbeat edge), fire whichever
  is due. A `--continuous` local-only producer MUST publish this heartbeat or
  the fleet cannot tell "running fine, not uploading by design" from
  "crashed / camera dead."

### Consumer side (image-sampler2's `--from-cache` uploader, and the crop-cascade consumers)

- `scan_ring(dir)` — reused as the SINGLE definition of a valid managed v2
  file (same as the producer). Defined in exactly ONE place.
- `parse_v2_name(filename)` → recover `capture_ts_ns` authoritatively from
  the filename prefix.
- `read_frame_metadata(path)` → read vsn/camera/unique_id/provenance from
  embedded EXIF (do NOT re-embed — that changes bytes/unique_id).
- Selection modes:
  - `"newest"` (max capture_ts_ns) for the one-shot uploader.
  - `"all-unseen"` + a bounded seen-set (`f"{timestamp}|{name}"`) for a
    deduping watcher/consumer.
- **UPLOAD a COPY in a temp dir** — `upload_file` may move/consume the source,
  and the cached original must survive (no evict/mutate by the consumer).
  Verify cache count + bytes identical after upload.

### The capture_ts preservation rule (the critical correctness rule)

The cached file ALREADY carries its `<capture_ts_ns>` prefix in the filename
and embedded EXIF. The consumer must publish/upload with
`timestamp = that ORIGINAL capture_ts`:

```python
plugin.upload_file(path, meta=meta, timestamp=capture_ts_ns)  # NOT time.time_ns()
```

NOT re-stamped to now. `upload_timestamp` (the real send time) goes in `meta`
only.

Why: a detection is about **when the scene existed**, not when inference ran.
This is what makes a species/audio result trace to when the photo/audio was
captured. Frame-anchored all the way down: a crop inherits the parent frame's
`capture_ts`; a redacted audio clip should inherit the source clip's
`capture_ts` for the same reason.

The trap that bit a real Slack watcher (from
`frame-anchored-batch-consumers-and-watchers.md`): a pywaggle2 cache-consumer
publishes each record with `timestamp = the frame's CAPTURE time`. A consumer
that wakes every ~10 min publishes the whole backlog at once — dozens of
records in one burst, each stamped MINUTES in the past, non-monotonic, out of
order. A poller filtering on a short recent window (`start=-120s`) NEVER sees
them. Permanently invisible. The fix is on the POLLER (wide lookback ≥ batch
period + object-store lag + margin; seen-ID dedup not a high-water mark), but
the publisher's frame-anchoring is a FEATURE — do not compensate by publishing
publish-time timestamps.

### Provenance blob (for cascades)

Each downstream product carries a nested `source{}` object in UserComment JSON:
`source_class, source_confidence, source_bbox, source_unique_id, detection_index`
(YOLO context + full traceability for the classifier). Harmless no-op on plain
frames (no `source` key → full-frame mode).

### No cross-plugin triggering code

The producer→consumer handoff has NO cross-plugin triggering code. The
coupling is **file-mediated through `/local-cache` only**: the detector crops
each matching bbox and writes it as a v2 frame into a new crop stream; the
classifier reads crops on its own schedule. "The stages never call each other —
the shared local-cache is the only coupling" (`detect-then-crop-then-classify-cascade.md`).

Rate rule: the classifier must DRAIN the crop cache faster than the detector
FILLS it, or crops evict (oldest-first) before they're classified. Same rule
as the raw cache.

---

## 3. What the birdnet fork actually does re: the cache (the gap)

Honest finding: the birdnet fork does NOT currently consume from the cache.

Verified by grepping the whole repo (excluding `.git`) for
`from-cache` / `from_cache` / `read_cache` / `cache_file` / `/local-cache` /
`local_cache` / `scan_ring` — zero matches.

What it does today, in `app.py` `_get_audio` (app.py:621-628), is a three-way
priority switch, none of which touches the shared cache:

```python
def _get_audio(args) -> tuple[str, bool]:
    if args.input:        return args.input, False                     # local audio file
    elif args.camera:     return record_from_camera(...), True         # ffmpeg RTSP/MxPEG/FLV
    else:                 return record_from_microphone(...), True     # pywaggle Microphone
```

- **Microphone path** (`record_from_microphone`, app.py:164-222): pywaggle
  `Microphone.record()` returns an in-memory `AudioSample.data` (1-D float32
  mono), `redact_speech` zeroes speech windows in place on that array, THEN
  `sample.save(flac_path)` writes the already-redacted array to a temp FLAC.
  `run_cycle`'s `finally` does `shutil.rmtree(tmpdir)`. Raw array never
  touches disk before redaction.
- **Camera path** (`record_from_camera`, app.py:225-281): ffmpeg subprocess
  writing a temp FLAC. PCM never exists as a Python array (the
  `CAMERA-PATH-DESIGN.md` proposal to pipe ffmpeg to stdout and redact in
  process is PROPOSED, NOT BUILT).
- **File path**: `--input` reads a local audio file.

`project.md:236-237` states the cache work explicitly as PENDING:

> Refactor the microphone-path implementation into a standalone
> producer/consumer Sage plugin that consumes audio from the media-sampler
> cache and publishes a redacted audio product for downstream applications
> such as BirdNET.

So the fork is the case study for the **redaction** half and the Sage
deployment plumbing, but the cache-consume/produce half for audio is the
planned refactor, not a built one. The proven audio-from-cache analog doesn't
exist yet in this repo; the proven cache pattern lives in the
image-sampler2 / sage-yolo2 / sage-bioclip2 family (all image/vision), which
is the right template to copy from for an audio cache consumer.

What the fork DOES have that the cache pattern needs: the redaction is
already array-shaped and persistence-decoupled. `redact_speech` mutates
`sample.data` in place before `sample.save`, with a three-arm fail-closed
contract:
1. Normal return, `_reason is None`: YAMNet + gate succeeded, speech windows
   zeroed in place. Fall through to `sample.save()`.
2. Normal return, `_reason is not None`: `redact_speech` caught
   `(RedactionFailure, RedactionGateFailure)` itself, ran
   `audio_1d.fill(0.0)`, returned the all-zero buffer with
   `windows=[(0.0, duration)]`. Caller logs WARNING, falls through to
   `sample.save()` (writes silence). Raw audio gone.
3. `except (YAMNetRedactionFailure, RedactionGateFailure)` and
   `except Exception`: defensive force-zero, log, save silence.

This means slotting redaction between "read cached audio into array" and
"write redacted array back to cache as a v2 frame" is structurally the same
insertion it already occupies in `record_from_microphone` — just with the
cache replacing the mic and the temp FLAC. The mic path's "raw array never
touches disk until redacted" invariant maps directly to "raw cached audio is
never re-published until redacted."

---

## 4. sage.yaml structure (birdnet's actual file)

`~/AI-Projects/birdnet/sage.yaml` — 67 lines. Shape:

```yaml
name: "birdnet-species"
description: "Records audio, identifies bird/frog/insect species using BirdNET V2.4, publishes detections"
version: "0.3.0"
namespace: "sage"
authors: "Pete Beckman <pete.beckman@northwestern.edu>"
collaborators: "Dario Dematties (original avian-diversity-monitoring plugin)"
license: "MIT (code), CC BY-NC-SA 4.0 (BirdNET models)"
homepage: "https://github.com/flint-pete/birdnet"
keywords: "microphone, birdsong, bird classification, avian diversity, bioacoustics, BirdNET"
funding: "NSF 2436842"
source:
  architectures:
    - linux/arm64
    # - linux/amd64  # disabled: QEMU crashes on NVIDIA base image during cross-build
  url: "https://github.com/flint-pete/birdnet.git"
  branch: "main"

# NOTE: ECR input types support ONLY "string" and "int".
# Float-valued args are declared as "string" and parsed by argparse at runtime.
inputs:
# Audio input
- id: "input"
  type: "string"
- id: "camera"
  type: "string"
- id: "duration"
  type: "string"
- id: "sample-rate"
  type: "int"
# Model parameters
- id: "min-confidence"
  type: "string"
# ... (every model/location/runtime flag mirrors app.py's argparse 1:1)
- id: "save-match"
  type: "string"

metadata:
  # Three routed detection topics (biophony/anthrophony/geophony) + the audio.*
  # summary heartbeat — the wildcard covers all of them.
  ontology: env.detection.*
```

### Three things to note for the cache-consumer refactor

1. **Every CLI flag MUST be surfaced in `sage.yaml` `inputs`** — the design
   rule: adopters shouldn't edit code. So a future `--from-cache` /
   `--cache-root` / `--cache-name` / `--cache-max-count` / `--cache-max-mb` /
   `--heartbeat-secs` would each need an `id` here.
2. **ECR input types are only `string` and `int`** — that's why
   `duration` / `min-confidence` / `sensitivity` are `"string"` not
   `"float"`; argparse parses them at runtime.
3. **`metadata.ontology` is a wildcard** covering the topics the plugin
   publishes. The current `env.detection.*` covers biophony/anthrophony/
   geophony + audio.summary. A redacted-audio product would add whatever new
   topic (the project.md mentions a proposed `audio.redacted` measurement
   schema).

### Job YAMLs (the SES scheduling layer on top)

`~/AI-Projects/birdnet/jobs/` — three files
(`birdnet-w06c-gps.yaml`, `birdnet-reolink.yaml`, `birdnet-m16.yaml`). Shape:

```yaml
name: birdnet-w06c-gps
plugins:
  - name: birdnet-species
    pluginSpec:
      image: registry.sagecontinuum.org/beckman/birdnet-species:0.3.0
      args:
        - "--duration"
        - "20"
        - "--min-confidence"
        - "0.35"
        - "--save-match"
        - "*:0.5"
        - "--lat"
        - "43.9402"
        - "--lon"
        - "-110.6441"
nodes:
  W06C:
scienceRules:
  - "schedule(birdnet-species): cronjob('birdnet-w06c-gps-cron', '*/2 * * * *')"
successcriteria:
  - WallClock('30day')
```

- `plugins[].pluginSpec.image` — the full registry-qualified tag.
- `args[]` — the CLI flags as a flat list (values as strings).
- `nodes:` — node selector (REQUIRED for volume mounts per the design's §6:
  volume mounting errors out without `--selector`/`--node`).
- `scienceRules:` — the cron schedule.
- `successcriteria:` — `WallClock('30day')` etc.

None of the three job YAMLs mount a `volume:` or reference `/local-cache` —
consistent with the fork not yet being cache-aware. A cache-consumer job would
need to add a `volume:` mapping
`/media/plugin-data/local-cache → /local-cache` (the `pluginSpec.Volume
map[string]string` field exists today per the design's §6, but is
undocumented and requires a nodeSelector).

---

## 5. ecr-meta structure (birdnet's actual directory)

`~/AI-Projects/birdnet/ecr-meta/` — four files:

- **`README.md`** — the public app-store-style description. Sections: title,
  usage examples (mic, camera MxPEG, audio file, multi-cycle, continuous),
  an Arguments table mirroring `sage.yaml` `inputs` (Audio Input, Model
  Parameters, Location Filtering, Runtime), Output (the
  `env.detection.{biophony,anthrophony,geophony}.*` +
  `env.detection.audio.summary` ontology and the upload/FLAC save behavior).
- **`ecr-science-description.md`** — the "Science" + "AI@Edge" narrative for
  the ECR catalog page (why monitor birds, BirdNET V2.4 architecture, 6,522
  species, EfficientNetB0-like, dual mel-spectrograms, three soundscape-
  ecology categories). This is what shows up on
  `portal.sagecontinuum.org/apps`.
- **`ecr-science-image.jpg`**, **`ecr-icon.jpg`** — the catalog thumbnail/icon.

So `ecr-meta` is pure catalog metadata (human-facing description + icon),
decoupled from the machine-readable `sage.yaml`.

### The other ECR plumbing: `scripts/register-ecr-version.py`

148-line script that registers a new version in the ECR catalog via the API
WITHOUT the portal web UI. SES validates a job's image against the ECR
**catalog** (not the raw registry), so a side-loaded image still needs a
catalog record or `sesctl submit` fails with
`[registry.sagecontinuum.org/<ns>/<name>:<ver> does not exist in ECR]`.

It works by:
1. Cloning an existing version's record (`GET /apps/<ns>/<name>/<from_version>`).
2. Building a new payload: copy metadata, bump `version` + `source` (git url,
   branch, architectures, directory, dockerfile, build_args).
3. `POST /submit` with `Authorization: Sage ***` header scheme.
4. Confirming visibility in the public catalog (`GET /apps/<ns>/<name>`).

With the ECR build pipeline now working for birdnet (CPU-only
`python:3.12-slim` base, no CUDA/QEMU path), the normal build creates this
record; `register-ecr-version.py` is only needed for deliberate side-loads.

### Making an app version PUBLIC

A freshly-built ECR image is PRIVATE by default. Skipping this fails per-node
in a confusing way: a node with cached registry creds pulls it fine, but a
node pulling anonymously gets `ErrImagePull` / `insufficient_scope` →
`ImagePullBackOff`. Grant public read via API:

```bash
curl -s -X PUT -H "Authorization: Sage ***" -H 'Content-Type: application/json' \
  https://ecr.sagecontinuum.org/api/permissions/beckman/birdnet-species \
  -d '{"operation":"add","granteeType":"GROUP","grantee":"AllUsers","permission":"READ"}'
```

---

## 6. The consume→produce pattern for audio redaction (the target)

Pulling it together for the refactor — an audio redaction plugin that consumes
from the media-sampler cache and produces a redacted product back into it for
a downstream plugin.

### Pipeline shape

```
media-sampler (producer)        redaction plugin (consumer + producer)      birdnet (downstream consumer)
mic/camera → cache         →    read newest audio from cache           →    read redacted audio from cache
/local-cache/.../mic/           run redact_speech on in-memory array        /local-cache/<redacted>/...
<ts>-v2-<vsn>-<cam>.wav         write redacted array as v2 frame back       classify_file on the cached redacted clip
                                into <redacted>/<cam>/ stream
```

### Consuming (read side)

Vendor the read-side machinery BYTE-IDENTICAL from the image-sampler2 /
sage-yolo2 / sage-bioclip2 family. Concretely:
- `consumer.py` (`scan_ring` / `parse_v2_name` / `read_frame_metadata`),
  `selection.py` (stride/all-unseen),
  `seenstore.py` (dedup), `node_info.py`, `save_match.py`.
- sha256-verify after copy. These ARE the v2 read contract; document in a
  `VENDORED.md` with a sync obligation and let the carried-over tests be the
  drift guard. (After 2+ consumers exist, a shared package is the cleaner
  long-term move than repeated vendoring — track as a follow-up, don't block
  on it.)

For audio (adapting the vision v2 frame to audio):
- `scan_ring` on the producer's audio stream dir.
- `parse_v2_name` to recover `capture_ts_ns` authoritatively.
- Select "newest" (max capture_ts_ns) for a one-shot, or "all-unseen" + a
  bounded seen-set for a deduping continuous consumer.
- **PRESERVE the original capture_ts end-to-end** — the redacted product must
  carry the same `capture_ts_ns` prefix and the same upload/publish timestamp
  as the source, so a downstream BirdNET detection traces to when the audio
  was recorded, not when redaction ran.

### Producing (write side)

The image-sampler2 `cache.py` ring, hoisted to the pywaggle2 `cache_file()`
primitive, OR vendored into a `crop_writer.py`-style module. Merge only the
WRITE pieces into one module:
- `build_v2_name`, `embed_all` / `build_exif_bytes` / `inject_exif` (piexif),
  `scan_ring` / `plan_evictions` / `commit_capture`, plus a `write_frame()`
  one-call helper (scan → plan → atomic tmp → fsync → `os.replace`).
- DROP the read-side + config-probe helpers — the consuming plugin already has
  its own reader.
- `unique_id` = SHA256 of the bytes BEFORE EXIF injection (stable,
  recomputable, no self-reference paradox; also the classifier's dedup key).

On every write, run the per-write graceful Layer-1 ring
(`keep_max_count` / `keep_max_mb`, evict-on-either). Give the redacted stream
its **own bounded sub-stream** — its own `cache-name` / camera leaf dir, its
own caps, separate from the raw audio cache. Write files world-readable +
dirs traversable (`chmod` on write if needed) so the downstream plugin pod
(different uid) can read them.

### The contract test that matters most

Write a test that produces a redacted audio frame with the vendored writer and
reads it back with the SAME `consumer.read_frame_metadata` the downstream
plugin uses — proves byte-compatibility. There is no automated diff (a merge,
not a mirror), so THE TESTS ARE THE SYNC CONTRACT. Register the vendored
module in `VENDORED.md` with its divergence list + this obligation.

### sage.yaml / inputs for the new plugin

Surface every new flag as a string/int `id` (same pattern as birdnet's
existing flags):- `--from-cache` (string — the stream dir to consume from)
- `--cache-root` (string — default `/local-cache`)
- `--cache-name` (string — the redacted product's instance label)
- `--cache-max-count` (int), `--cache-max-mb` (int)
- `--heartbeat-secs` (int)

`metadata.ontology` gets the redacted-audio topic added (the proposed
`audio.redacted` measurement schema).

### Dockerfile — the #1 build trap

`COPY` each source module individually (birdnet's Dockerfile is
`COPY save_match.py .` / `COPY app.py .`, not `COPY . .`). When you add a NEW
module (e.g. `cache_writer.py`, `consumer.py`), it is silently omitted from
the image → the container crashes at startup with `ImportError`, invisible to
`make test` (which runs from the repo, not the image). ALWAYS add a matching
`COPY <newmodule>.py .` line when introducing a new top-level module; grep the
Dockerfile for the new filename as a build gate. This was hit twice on
image-sampler2 (the `heartbeat.py` case) and again on sage-yolo2
(`crop_writer.py`).

### Rate rule

The downstream consumer (BirdNET) must DRAIN the redacted cache faster than
redaction FILLS it, or redacted clips evict (oldest-first) before BirdNET
reads them. Same rule as the raw cache. Tune `--every` on the consumer and
the redacted ring caps together once live; leave generous defaults (500/500)
until the real drain cadence is known.

### Mapping the existing redaction into the cache pattern

The redaction is already array-shaped and persistence-decoupled:
`redact_speech` mutates `sample.data` in place before `sample.save`, with the
three-arm fail-closed contract documented in §3. So the insertion is
structurally the same as `record_from_microphone`, with the cache replacing
the mic and the temp FLAC:

| record_from_microphone (today) | cache consumer/producer (target) |
|---|---|
| `mic = Microphone(); sample = mic.record(dur)` | read newest cached audio file, decode to 1-D float32 array + samplerate |
| `redact_speech(sample.data, sample.samplerate)` — in-place zero | `redact_speech(audio_array, samplerate)` — in-place zero (SAME call) |
| `sample.save(flac_path)` — temp FLAC, rmtree'd in finally | `write_frame(redacted_array, timestamp=capture_ts)` — v2 frame into `<redacted>/<cam>/` ring, persisted |
| publish `env.detection.*` with `timestamp=cycle_start_ns` | publish redacted-audio topic with `timestamp=capture_ts_ns` (the SOURCE clip's ts, not now) |

The mic path's "raw array never touches disk until redacted" invariant maps
directly to "raw cached audio is never re-published until redacted." The
downstream BirdNET plugin then classifies the cached redacted clip the same
way it today classifies a temp FLAC — just with `--input /local-cache/<redacted>/<cam>/`
instead of `--camera` or the mic.

---

## See also

- `docs/local-cache-design.md` — the full two-layer /local-cache proposal.
- `skills/sage-waggle/references/local-cache-ring-buffer.md` — image-sampler2
  ring implementation (the Layer-1 reference).
- `skills/sage-waggle/references/continuous-producer-patterns.md` — dual-grid
  loop + from-cache uploader + clean self-exit bounds.
- `skills/sage-waggle/references/crop-producer-detect-classify-cascade.md` —
  sage-yolo2 crop-producer (vendoring the v2 WRITE side).
- `skills/sage-waggle/references/detect-then-crop-then-classify-cascade.md` —
  sage-yolo2 → sage-bioclip2 (building the second plugin = assembly).
- `skills/sage-waggle/references/frame-anchored-batch-consumers-and-watchers.md`
  — the capture-ts preservation rule and the batch-consumer poller trap.
- `~/AI-Projects/birdnet/app.py` — the existing mic/camera/file paths + redaction.
- `~/AI-Projects/birdnet/redaction/` — the redaction package (gate, YAMNet, apply).
- `~/AI-Projects/birdnet/REDACTION-INTEGRATION-NOTES.md` — the integration
  record for the mic-path redaction (Step 4 = the actually-landed caller pattern).
- `~/AI-Projects/birdnet/redaction/CAMERA-PATH-DESIGN.md` — the proposed
  (not built) ffmpeg stdout-pipe → array path, the closest structural analog
  to a cache-consumer (both produce a bare numpy array that feeds `redact_speech`).
- `~/AI-Projects/birdnet/sage.yaml` — the actual sage.yaml.
- `~/AI-Projects/birdnet/ecr-meta/` — the actual ECR catalog metadata.
- `~/AI-Projects/birdnet/scripts/register-ecr-version.py` — the ECR catalog
  API registration script.

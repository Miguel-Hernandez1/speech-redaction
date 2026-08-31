"""Write side of the v2 cache producer.

Writes a redacted product (FLAC + JSON sidecar) into a SEPARATE output cache,
following the media-sampler3 producer contract:
  - organize output as a per-source subtree, <output-cache>/redacted_audio/
    <source>/, mirroring how media-sampler3 organizes its own streams.
  - keep the original capture_ts_ns in the filename (frame-anchoring); change
    only the source label (hummingcam_mic -> hummingcam_mic_redacted).
  - write the sidecar into place FIRST, then the clip, so a consumer that sees a
    clip is guaranteed its sidecar exists.
  - publish atomically (temp name + fsync + rename).
  - bound the output ring by count/MB (evict oldest-first by capture_ts prefix).

Redaction itself is delegated to write_redacted_flac (which calls redact_speech);
this module never re-implements redaction, and never calls redact_speech a second
time (that would redact already-zeroed audio and wipe the real windows).
"""

import hashlib
import json
import logging
import os
import tempfile

from cache_consumer import parse_frame_name
from flac_redaction import write_redacted_flac

logger = logging.getLogger("speech-redaction.producer")

SCHEMA_VERSION = "sage-media-1"

# Output layout: products go under a per-source subtree of the output cache root,
# <output-cache>/redacted_audio/<out_source>/, mirroring how media-sampler3
# organizes its own streams (<cache-root>/<camera-audio>/<mic>/). The ring is
# therefore per-source: each source subdir is its own independent ring.
REDACTED_AUDIO_DIRNAME = "redacted_audio"


def stream_dir(output_cache, out_source):
    """The output stream directory for out_source under the cache root:
    <output_cache>/redacted_audio/<out_source>/."""
    return os.path.join(output_cache, REDACTED_AUDIO_DIRNAME, out_source)


def output_source_label(input_source, override=None):
    """The output stream's source label. Derived from the input source so the
    originating stream stays legible when a node has more than one mic; a fixed
    label would throw that away. Overridable."""
    if override:
        return override
    return f"{input_source}_redacted" if input_source else "redacted"


def build_output_name(capture_ts_ns, vsn, source):
    return f"{capture_ts_ns}-v2-{vsn}-{source}.flac"


def _scan_products(output_dir):
    """Return ([(capture_ts_ns, clip_path, sidecar_path, pair_bytes)] oldest-first,
    total_bytes) for every managed v2 product pair in output_dir."""
    try:
        names = os.listdir(output_dir)
    except FileNotFoundError:
        return [], 0
    items = []
    total = 0
    for name in names:
        if name.startswith("."):
            continue
        parsed = parse_frame_name(name)   # matches *.flac only
        if parsed is None:
            continue
        ts = parsed[0]
        clip = os.path.join(output_dir, name)
        sidecar = clip + ".json"
        pair_bytes = 0
        for p in (clip, sidecar):
            try:
                pair_bytes += os.path.getsize(p)
            except OSError:
                pass
        items.append((ts, clip, sidecar, pair_bytes))
        total += pair_bytes
    items.sort(key=lambda x: x[0])
    return items, total


def _evict_to_fit(output_dir, new_bytes, max_count, max_bytes):
    """Delete oldest product pairs (clip + sidecar) so that count+1 <= max_count
    and existing_bytes + new_bytes <= max_bytes. Fail-soft on delete errors."""
    if max_count is None and max_bytes is None:
        return
    items, total = _scan_products(output_dir)
    count = len(items)
    i = 0
    while i < len(items):
        over_count = max_count is not None and (count + 1) > max_count
        over_bytes = max_bytes is not None and (total + new_bytes) > max_bytes
        if not (over_count or over_bytes):
            break
        _, clip, sidecar, pair_bytes = items[i]
        for p in (clip, sidecar):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Ring eviction could not delete %s: %s", p, e)
        count -= 1
        total -= pair_bytes
        i += 1


def write_product(output_cache, capture_ts_ns, vsn, out_source, audio, samplerate,
                  subtype, source_sidecar, source_unique_id, plugin_name,
                  gate=None, noise_fill=False, max_count=None, max_bytes=None):
    """Redact `audio` and write the product under output_cache.

    The product goes into a per-source subtree,
    <output_cache>/redacted_audio/<out_source>/, so the ring is per-source.

    Returns (clip_path, sidecar_path, unique_id, windows, reason).

    Order: redact + write the FLAC to a temp file (write_redacted_flac calls
    redact_speech once), sha256 the redacted bytes for a fresh unique_id, build
    the sidecar, evict to fit, then rename the SIDECAR into place FIRST and the
    CLIP second. Both renames are atomic.
    """
    stream = stream_dir(output_cache, out_source)
    os.makedirs(stream, exist_ok=True)
    # world-traversable so a downstream consumer pod (different uid) can descend
    # the tree and read the products.
    for d in (os.path.join(output_cache, REDACTED_AUDIO_DIRNAME), stream):
        try:
            os.chmod(d, 0o755)
        except OSError:
            pass

    final_clip_name = build_output_name(capture_ts_ns, vsn, out_source)
    final_clip = os.path.join(stream, final_clip_name)
    final_sidecar = final_clip + ".json"

    # 1) redact + write the FLAC to a temp file. write_redacted_flac calls
    #    redact_speech internally; we do NOT call it separately or we redact
    #    twice and lose the real windows.
    fd, tmp_clip = tempfile.mkstemp(prefix=".tmp-clip-", suffix=".flac", dir=stream)
    os.close(fd)
    try:
        _, windows, reason = write_redacted_flac(
            audio, samplerate, tmp_clip, gate=gate, noise_fill=noise_fill,
            subtype=subtype)
        with open(tmp_clip, "rb") as fh:
            redacted_bytes = fh.read()
    except Exception:
        try:
            os.unlink(tmp_clip)
        except OSError:
            pass
        raise

    unique_id = hashlib.sha256(redacted_bytes).hexdigest()

    # 2) build the output sidecar: same field set + schema, capture_ts unchanged,
    #    object_name/source/vsn/plugin updated, fresh unique_id, provenance +
    #    redaction windows + fail-closed reason.
    sidecar = dict(source_sidecar) if isinstance(source_sidecar, dict) else {}
    sidecar["schema_version"] = SCHEMA_VERSION
    sidecar["capture_timestamp_ns"] = capture_ts_ns
    sidecar["object_name"] = final_clip_name
    sidecar["source"] = out_source
    sidecar["vsn"] = vsn
    sidecar["plugin"] = plugin_name
    sidecar["unique_id"] = unique_id
    sidecar["source_unique_id"] = source_unique_id
    sidecar["redaction_windows"] = [[float(s), float(e - s)] for s, e in windows]
    sidecar["redaction_fail_closed"] = reason is not None
    sidecar["redaction_fail_closed_reason"] = reason

    sidecar_bytes = json.dumps(sidecar).encode("utf-8")
    new_bytes = len(redacted_bytes) + len(sidecar_bytes)

    # 3) evict oldest to fit BEFORE the new pair joins the ring (per-source).
    _evict_to_fit(stream, new_bytes, max_count, max_bytes)

    # 4) write the sidecar to a temp file and fsync it.
    fd, tmp_sidecar = tempfile.mkstemp(prefix=".tmp-side-", suffix=".json", dir=stream)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(sidecar_bytes)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        for p in (tmp_sidecar, tmp_clip):
            try:
                os.unlink(p)
            except OSError:
                pass
        raise

    # 5) rename the sidecar into place FIRST, then the clip. A consumer that sees
    #    the clip is now guaranteed the sidecar already exists.
    os.replace(tmp_sidecar, final_sidecar)
    os.replace(tmp_clip, final_clip)
    for p in (final_sidecar, final_clip):
        try:
            os.chmod(p, 0o644)
        except OSError:
            pass

    return final_clip, final_sidecar, unique_id, windows, reason

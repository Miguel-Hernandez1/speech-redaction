#!/usr/bin/env python3
"""speech-redaction — a Sage local-cache consumer/producer plugin.

Reads audio from the local cache (or a single file for dev), removes human
speech in memory with the redaction package, and writes the redacted audio
back into the cache as a NEW v2 product for a downstream plugin (e.g. BirdNET)
to consume.

Privacy contract (from CACHE-PATTERN-MAP.md §6 + the redaction package):
  - The raw unredacted array is NEVER re-published to the cache. redact_speech
    mutates the array IN PLACE and is fail-closed: on any model/gate error it
    zeroes the ENTIRE buffer rather than return raw audio. Only after
    redact_speech returns do we write a product.
  - If something unexpected escapes redact_speech (defensive except), we skip
    the write entirely — we never emit a product we cannot prove is redacted.
  - capture_ts is preserved end-to-end: the output v2 filename carries the SAME
    capture_ts_ns prefix as the source file, and publish() is stamped with that
    capture time (NOT time.time_ns()), so a downstream detection traces to when
    the audio was recorded, not when redaction ran.

Cache write side (Layer-1 ring, per the image-sampler2 reference):
  - per-stream subdir under --cache-root/--cache-name
  - atomic write (tmp + fsync + os.replace)
  - evict oldest (by capture_ts filename prefix) on-either count/MB cap on every write
  - world-readable files, traversable dirs (downstream plugin runs as another uid)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from redaction.apply import redact_speech

logger = logging.getLogger("speech-redaction")

# ── v2 filename convention ──────────────────────────────────────────────────
# <capture_ts_ns>-v2-<unique_id>.<ext>   (audio uses .wav; the unique_id is a
# sha1 of the redacted bytes, so names never collide and a consumer can dedup).
# capture_ts_ns is a positive integer nanosecond timestamp prefix — the SINGLE
# authoritative sort/eviction key (mtime lies if the clock stepped).
_V2_NAME_RE = re.compile(r"^(\d{19})-v2-[0-9a-f]{40}\.\w+$")

# Heartbeat topic + cadence (a continuous consumer must publish a heartbeat so
# a silent run is distinguishable from a dead plugin — see CACHE-PATTERN-MAP.md
# §1 "Liveness heartbeat"). 60s default.
HEARTBEAT_TOPIC = "env.speechredaction.heartbeat"
REDACTION_EVENT_TOPIC = "env.speechredaction.event"
_DEFAULT_HEARTBEAT_SECS = 60


# ── audio I/O ───────────────────────────────────────────────────────────────
def load_audio_mono(path):
    """Read an audio file into a 1-D float32 array in [-1, 1] + its samplerate.

    Uses soundfile (already a dependency); mirrors run_redaction_on_capture's
    loader. Multichannel is downmixed to mono and integer PCM is normalized.
    """
    import soundfile as sf
    data, sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32), int(sr)


def write_audio_mono(path, audio_1d, samplerate):
    """Write a 1-D float32 array to a WAV (16-bit PCM).

    Returns the raw bytes on disk (for the sha1 unique_id). Called on a .tmp;
    the caller fsyncs + os.replace to the final v2 name.
    """
    import soundfile as sf
    # 16-bit PCM keeps the redacted product small and is universally readable by
    # downstream consumers (librosa/soundfile/BirdNET). subtype PCM_16.
    sf.write(path, audio_1d, samplerate, subtype="PCM_16")
    with open(path, "rb") as fh:
        return fh.read()


# ── v2 name helpers ─────────────────────────────────────────────────────────
def parse_capture_ts_ns(filename):
    """Recover the capture_ts_ns prefix from a v2 filename, or None if not v2.

    The SINGLE definition of "a valid managed v2 file" — ignores .tmp and
    non-v2 files (they are not ring members, not counted, not evicted).
    """
    m = _V2_NAME_RE.match(os.path.basename(filename))
    return int(m.group(1)) if m else None


def build_v2_name(capture_ts_ns, unique_id, ext="wav"):
    return f"{capture_ts_ns}-v2-{unique_id}.{ext}"


# ── Layer-1 ring (vendor of the image-sampler2 write side) ──────────────────
def scan_ring(stream_dir):
    """Return (list of (capture_ts_ns, path, size) sorted oldest-first, total_bytes).

    Only files matching the v2 name pattern count as ring members. .tmp and
    non-v2 files are ignored (never counted, never evicted). Stateles: re-scans
    disk every call, so crash/restart just re-scans — no in-memory ring state.
    """
    if not os.path.isdir(stream_dir):
        return [], 0
    members = []
    total = 0
    for name in os.listdir(stream_dir):
        ts = parse_capture_ts_ns(name)
        if ts is None:
            continue
        p = os.path.join(stream_dir, name)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        members.append((ts, p, sz))
        total += sz
    members.sort(key=lambda x: x[0])  # oldest-first by capture_ts prefix
    return members, total


def plan_evictions(members, total_bytes, new_bytes, max_count, max_bytes):
    """Return a list of paths to delete so count+1 <= max_count AND
    bytes+new_bytes <= max_bytes. members is oldest-first.

    E3 guard: if max_bytes is set and new_bytes alone exceeds max_bytes even
    with an empty ring, returns None to signal "drop the new file instead of
    letting one file blow the cap." The caller deletes the .tmp and skips the
    write.
    """
    if max_bytes is not None and new_bytes > max_bytes and not members:
        return None
    to_delete = []
    count = len(members)
    bytes_acc = total_bytes + new_bytes
    i = 0
    while i < len(members):
        over_count = max_count is not None and (count + 1) > max_count
        over_bytes = max_bytes is not None and bytes_acc > max_bytes
        if not (over_count or over_bytes):
            break
        ts, path, sz = members[i]
        to_delete.append(path)
        count -= 1
        bytes_acc -= sz
        i += 1
    return to_delete


def commit_capture(tmp_path, final_path):
    """fsync the .tmp then atomically os.replace to the final v2 name.

    A torn file never appears under the final name and the ring never
    transiently exceeds caps (eviction ran BEFORE this rename).
    """
    with open(tmp_path, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)
    # Make the new file world-readable so a downstream plugin pod (different uid)
    # can read it. chmod the file; the dir is made traversable at creation.
    try:
        os.chmod(final_path, 0o644)
    except OSError:
        pass


def evict_and_commit(stream_dir, audio_1d, samplerate, capture_ts_ns, ext,
                     max_count, max_bytes):
    """One-call write helper: write .tmp → scan → plan evictions → evict → commit.

    Returns the final path on success, or None if the E3 guard dropped the new
    file (oversized under a byte cap). Never raises on eviction-delete failure
    (fail-SOFT at runtime) — logs and continues.
    """
    os.makedirs(stream_dir, exist_ok=True)
    # Make the stream dir traversable by other uids (downstream consumer pod).
    try:
        os.chmod(stream_dir, 0o755)
    except OSError:
        pass

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=f".{ext}", dir=stream_dir)
    os.close(fd)
    try:
        raw_bytes = write_audio_mono(tmp_path, audio_1d, samplerate)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    new_bytes = len(raw_bytes)
    unique_id = hashlib.sha1(raw_bytes).hexdigest()
    final_name = build_v2_name(capture_ts_ns, unique_id, ext=ext)
    final_path = os.path.join(stream_dir, final_name)

    members, total_bytes = scan_ring(stream_dir)
    plan = plan_evictions(members, total_bytes, new_bytes, max_count, max_bytes)
    if plan is None:
        # E3 guard: new file alone exceeds the byte cap. Drop it, keep the cache
        # valid, skip the write entirely.
        logger.warning(
            "E3 guard: redacted %d-byte clip exceeds byte cap %d; dropping new "
            "file (no write).", new_bytes, max_bytes)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None

    for p in plan:
        try:
            os.unlink(p)
        except OSError as e:
            # fail-SOFT: a locked/permission delete error must not crash the run.
            logger.warning("Ring eviction could not delete %s: %s", p, e)

    commit_capture(tmp_path, final_path)
    return final_path


# ── core: process one audio source ──────────────────────────────────────────
def process_audio_file(src_path, plugin, args, capture_ts_ns=None):
    """Load audio from src_path, redact it, write the redacted product to the
    cache, and publish a redaction event. Returns True on success.

    If capture_ts_ns is None (the --input dev path), the current time is used
    as the capture timestamp. The --from-cache path MUST pass the source v2
    filename's capture_ts_ns so the product inherits the original capture time.
    """
    if capture_ts_ns is None:
        capture_ts_ns = time.time_ns()

    # 1) Load the source audio into a numpy array. The raw unredacted array
    #    lives only in memory; it is never written to disk.
    try:
        audio, sr = load_audio_mono(src_path)
    except Exception as e:
        logger.error("Could not load audio from %s: %s", src_path, e)
        return False
    if audio.size == 0:
        logger.warning("Empty audio array from %s; skipping.", src_path)
        return False

    # 2) Redact IN PLACE. redact_speech is fail-closed: on any model/gate
    #    error it zeroes the ENTIRE buffer (privacy-safe silence) and returns a
    #    non-None reason. We still write that silence — it is a valid redacted
    #    product. Only if something unexpected ESCAPES redact_speech do we skip
    #    the write (defensive except below) so we never emit an unverifiable
    #    product.
    duration_s = audio.size / float(sr) if sr else 0.0
    try:
        redacted, windows, reason = redact_speech(audio, sr)
    except Exception:
        logger.exception(
            "Unexpected redaction failure on %s; redact_speech should have "
            "caught this. Skipping write — no product emitted.", src_path)
        return False

    if reason is not None:
        logger.warning(
            "Speech redaction failed closed (%s) on %.2fs clip from %s; entire "
            "buffer zeroed before write.", reason, duration_s, src_path)
    else:
        redacted_dur = sum(e - s for s, e in windows) if windows else 0.0
        logger.info(
            "Redacted %s: %d window(s), %.3fs of %.3fs zeroed.",
            os.path.basename(src_path), len(windows), redacted_dur, duration_s)

    # 2b) Trust-but-verify: confirm the array we're about to persist is actually
    #     silent across every declared redaction window. redact_speech's contract
    #     is "windows are zeroed in place"; if it ever returns reason=None with
    #     non-zero samples inside a window (a package bug, a future edit, or an
    #     integer-precision edge), we must NOT write that product — fail closed.
    if reason is None and sr and redacted is not None:
        for start_s, end_s in (windows or []):
            i0 = max(0, int(start_s * sr))
            i1 = min(redacted.size, int(end_s * sr))
            if i0 < i1 and not np.allclose(redacted[i0:i1], 0.0):
                logger.error(
                    "Redaction verification FAILED for %s: declared window "
                    "[%.3fs, %.3fs] (samples %d:%d) is non-silent AFTER "
                    "redact_speech. Refusing to write — no product emitted.",
                    os.path.basename(src_path), start_s, end_s, i0, i1)
                return False

    # 3) Write the redacted product to the cache atomically + run Layer-1 ring.
    stream_dir = os.path.join(args.cache_root, args.cache_name)
    final_path = None
    if not args.dry_run:
        try:
            final_path = evict_and_commit(
                stream_dir, redacted, sr, capture_ts_ns, ext="wav",
                max_count=args.cache_max_count, max_bytes=args.cache_max_mb,
            )
        except Exception as e:
            logger.error("Cache write failed for %s: %s — no product emitted.",
                         os.path.basename(src_path), e)
            return False
        if final_path is None:
            # E3 guard dropped it (oversized under a byte cap).
            return False
        logger.info("Wrote redacted product: %s", final_path)

    # 4) Publish a redaction event stamped with the SOURCE capture time (not
    #    now), so a downstream detection traces to when the audio was recorded.
    #    pywaggle publish accepts arbitrary dotted names + ns timestamps + string
    #    meta. Heartbeat is published separately on its own grid.
    if not args.dry_run and plugin is not None:
        try:
            plugin.publish(
                REDACTION_EVENT_TOPIC,
                json.dumps({
                    "capture_ts_ns": str(capture_ts_ns),
                    "duration_s": f"{duration_s:.3f}",
                    "n_windows": str(len(windows)),
                    "redacted_s": f"{sum(e - s for s, e in windows) if windows else 0.0:.3f}",
                    "fail_closed": str(bool(reason)),
                    "source": os.path.basename(src_path),
                    "product": os.path.basename(final_path) if final_path else "",
                }),
                timestamp=capture_ts_ns,
                meta={"cache_name": str(args.cache_name)},
            )
        except Exception as e:
            # fail-SOFT: a broken broker must not kill the run.
            logger.warning("publish(%s) failed: %s", REDACTION_EVENT_TOPIC, e)

    return True


# ── cache consumer (the --from-cache pattern) ──────────────────────────────
def consume_from_cache(src_dir, plugin, args):
    """Walk the source cache stream, process each v2 file we have not seen,
    preserving each file's capture_ts_ns on its redacted product.

    Uses a bounded seen-set keyed by f"{capture_ts}|{name}" to dedup (a
    crash/restart re-scans; the seen-set is in-memory only and simply discards
    on restart, re-processing is safe — we just redo work, not double-emit,
    because the output v2 name is content-addressed by sha1 so a re-run
    overwrites the identical product name).
    """
    seen = set()
    processed = 0
    failures = 0
    while True:
        # Re-scan each pass (stateless). newest-last to process oldest first
        # (oldest data is most at-risk of eviction before we get to it).
        members, _ = scan_ring(src_dir)
        todo = [(ts, p) for ts, p, _ in members
                if f"{ts}|{os.path.basename(p)}" not in seen]
        if not todo:
            logger.info("No new cached audio in %s; waiting for producer.",
                        src_dir)
            _heartbeat(plugin, args, processed, failures,
                       last_status="idle")
            time.sleep(args.duration if args.duration > 0 else 5)
            continue

        for ts, path in todo:
            ok = process_audio_file(path, plugin, args, capture_ts_ns=ts)
            seen.add(f"{ts}|{os.path.basename(path)}")
            # Bounded seen-set (don't grow forever).
            if len(seen) > 4096:
                seen = set(sorted(seen)[-2048:])
            if ok:
                processed += 1
            else:
                failures += 1
            _heartbeat(plugin, args, processed, failures,
                       last_status="ok" if ok else "fail")


# ── heartbeat ───────────────────────────────────────────────────────────────
def _heartbeat(plugin, args, processed, failures, last_status="none"):
    """Publish a cache heartbeat so a silent run ≠ dead plugin.

    Payload covers processed/failures_since_last and last_status. timestamp is
    now() (the heartbeat is about liveness, NOT capture-anchored). Wrap in
    try/except: a broken broker must never kill the loop.
    """
    if plugin is None:
        return
    try:
        plugin.publish(
            HEARTBEAT_TOPIC,
            json.dumps({
                "processed": str(processed),
                "failures": str(failures),
                "last_status": str(last_status),
            }),
            timestamp=time.time_ns(),
            meta={"cache_name": str(args.cache_name)},
        )
    except Exception as e:
        logger.warning("heartbeat publish failed: %s", e)


# ── argparse ────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Speech-redaction Sage cache consumer/producer. "
                    "Reads audio, removes human speech, writes redacted audio "
                    "back to the local cache.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--input", metavar="FILE",
                   help="read a single audio file (dev/test). Mutually exclusive "
                        "with --from-cache.")
    g.add_argument("--from-cache", metavar="DIR",
                   help="consume audio from a cache stream directory (the real "
                        "pattern). Mutually exclusive with --input.")
    p.add_argument("--cache-root", default="/local-cache",
                   help="cache root dir (default /local-cache).")
    p.add_argument("--cache-name", default="speech-redaction",
                   help="output stream name / instance label under cache-root "
                        "(default speech-redaction).")
    p.add_argument("--cache-max-count", type=int, default=None,
                   help="max files to keep in the output ring (Layer-1).")
    p.add_argument("--cache-max-mb", type=int, default=None,
                   help="max total MB (10^6 bytes) to keep in the output ring.")
    p.add_argument("--duration", type=float, default=5.0,
                   help="in --from-cache mode, seconds to sleep between scans "
                        "when no new audio is found (default 5). In --input mode "
                        "this flag is ignored.")
    p.add_argument("--dry-run", action="store_true",
                   help="run redaction but do NOT write to cache or publish.")
    return p.parse_args(argv)


def _validate(args):
    if not args.input and not args.from_cache:
        # Fail-fast at config: must pick an input mode.
        raise SystemExit("config error: provide --input FILE or --from-cache DIR.")
    if args.from_cache:
        # Fail-fast at config: the source cache dir must exist.
        if not os.path.isdir(args.from_cache):
            raise SystemExit(
                f"config error: --from-cache dir not found: {args.from_cache}")
        # At least one cap should be set for a continuous consumer (unbounded
        # growth otherwise). Warn, don't fail — a one-shot dev loop is fine.
        if args.cache_max_count is None and args.cache_max_mb is None:
            logger.warning(
                "No --cache-max-count or --cache-max-mb set; the output ring is "
                "UNBOUNDED. Set at least one for production.")


# ── main ────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _validate(args)

    # Open the pywaggle Plugin fail-SOFT. A bare off-node test (no RabbitMQ)
    # logs a warning and runs without heartbeats/publish — the cache still works.
    plugin = None
    if not args.dry_run:
        try:
            from waggle.plugin import Plugin
            plugin = Plugin()
        except Exception as e:
            logger.warning("Could not open pywaggle Plugin (%s); running without "
                           "publish/heartbeat.", e)

    if args.input:
        # Single-file dev/test path. capture_ts defaults to now; redaction runs
        # and (unless --dry-run) the product is written under cache-root/cache-name.
        ok = process_audio_file(args.input, plugin, args)
        if not ok:
            sys.exit(1)
        return

    # Continuous cache-consumer mode.
    logger.info("Consuming audio from %s, writing redacted products to %s/%s",
                args.from_cache, args.cache_root, args.cache_name)
    if plugin is not None:
        _heartbeat(plugin, args, processed=0, failures=0, last_status="start")
    try:
        consume_from_cache(args.from_cache, plugin, args)
    except KeyboardInterrupt:
        logger.info("Interrupted; exiting.")


if __name__ == "__main__":
    main()

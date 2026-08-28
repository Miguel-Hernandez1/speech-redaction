#!/usr/bin/env python3
"""speech-redaction -- a v2-style batch cache consumer/producer Sage plugin.

Reads audio frames from a media-sampler3 local cache, removes human speech in
memory (redaction.apply.redact_speech, via flac_redaction.write_redacted_flac),
and writes a redacted product (FLAC + sidecar) into a SEPARATE output cache for a
downstream consumer (e.g. BirdNET) to read. Runs one batch and exits; WES
schedules the cadence.

Contract:
  - redaction happens in memory; only the redacted product is written.
  - the output sidecar is renamed into place BEFORE the clip (producer contract).
  - the original capture_ts_ns is preserved on the product (frame-anchoring);
    only the source label changes (hummingcam_mic -> hummingcam_mic_redacted).
  - a durable, bounded seen-store (keyed on sidecar unique_id) means a restart
    does not reprocess.
  - the output ring is bounded by --cache-max-count / --cache-max-mb.

This plugin knows nothing about BirdNET or any specific downstream consumer.
"""

import argparse
import json
import logging
import os

import cache_consumer
import cache_producer
from seen_store import SeenStore

logger = logging.getLogger("speech-redaction")

PLUGIN_NAME = "speech-redaction"
REDACTION_EVENT_TOPIC = "env.speechredaction.event"


def parse_duration(s):
    """'60s' / '5m' / '1h' / '250ms' / '90' -> seconds (float). Bare number = seconds."""
    s = str(s).strip().lower()
    for suffix, mult in (("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mult
    return float(s)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Speech-redaction v2 batch cache consumer/producer. Reads "
                    "audio from a local cache, removes human speech, and writes "
                    "redacted products to a separate output cache. Runs one batch "
                    "and exits.")
    p.add_argument("--source", default="cache", choices=["cache"],
                   help="input source type (only 'cache' is supported).")
    p.add_argument("--input", metavar="DIR",
                   help="input cache stream dir to consume.")
    p.add_argument("--output-cache", metavar="DIR",
                   help="output cache stream dir for redacted products.")
    p.add_argument("--cache-root", default="/local-cache",
                   help="cache root, used for the default seen-store .state dir "
                        "(default /local-cache).")
    p.add_argument("--every", default=None,
                   help="advisory cadence for v2-family parity. No-op here: this "
                        "plugin runs one batch and exits; WES schedules cadence.")
    p.add_argument("--select-every", default=None,
                   help="select at most one frame per this much CAPTURE time "
                        "(temporal downsample), e.g. 5m.")
    p.add_argument("--all-unseen", action="store_true",
                   help="process every unseen frame (default when --select-every "
                        "is not given).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap the number of frames processed this batch.")
    p.add_argument("--consumer-id", default=PLUGIN_NAME,
                   help="consumer identity; namespaces the seen-store.")
    p.add_argument("--seen-store", default=None,
                   help="override the seen-store path (default "
                        "<cache-root>/.state/<consumer-id>.json).")
    p.add_argument("--seen-max", type=int, default=100000,
                   help="max unique_ids to retain in the seen-store.")
    p.add_argument("--output-source", default=None,
                   help="override the output source label (default "
                        "<input source>_redacted).")
    p.add_argument("--cache-max-count", type=int, default=None,
                   help="max product files to keep in the output ring.")
    p.add_argument("--cache-max-mb", type=int, default=None,
                   help="max total MB (10^6 bytes) to keep in the output ring.")
    p.add_argument("--noise-fill", action="store_true",
                   help="fill redacted spans with level-matched noise instead of "
                        "zeros.")
    p.add_argument("--dry-run", action="store_true",
                   help="select and read frames but do NOT write products or "
                        "update the seen-store.")
    return p.parse_args(argv)


def _validate(args):
    if args.source == "cache":
        if not args.input:
            raise SystemExit("config error: --input DIR is required for --source cache.")
        if not args.output_cache:
            raise SystemExit("config error: --output-cache DIR is required.")
        if not os.path.isdir(args.input):
            raise SystemExit(f"config error: --input dir not found: {args.input}")
        if args.cache_max_count is None and args.cache_max_mb is None:
            logger.warning(
                "No --cache-max-count or --cache-max-mb set; the output ring is "
                "UNBOUNDED. Set at least one for production.")


def _default_seen_path(args):
    return os.path.join(args.cache_root, ".state", f"{args.consumer_id}.json")


def select_frames(frames, seen, args):
    """Filter to unseen frames (by sidecar unique_id), apply --select-every
    temporal downsampling, cap by --max-frames. Returns [(frame, sidecar, uid)]."""
    stride_ns = None
    if args.select_every:
        stride_ns = int(parse_duration(args.select_every) * 1e9)
    selected = []
    last_kept_ts = None
    for frame in frames:  # oldest-first
        sidecar = cache_consumer.load_sidecar(frame)
        uid = cache_consumer.frame_unique_id(frame, sidecar)
        if seen.contains(uid):
            continue
        if stride_ns is not None and last_kept_ts is not None:
            if frame.capture_ts_ns - last_kept_ts < stride_ns:
                continue
        selected.append((frame, sidecar, uid))
        last_kept_ts = frame.capture_ts_ns
        if args.max_frames is not None and len(selected) >= args.max_frames:
            break
    return selected


def _publish_event(plugin, args, frame, unique_id, windows, reason):
    if plugin is None:
        return
    try:
        plugin.publish(
            REDACTION_EVENT_TOPIC,
            json.dumps({
                "capture_ts_ns": str(frame.capture_ts_ns),
                "vsn": frame.vsn,
                "source": frame.source,
                "unique_id": str(unique_id),
                "n_windows": str(len(windows)),
                "fail_closed": str(bool(reason)),
            }),
            timestamp=frame.capture_ts_ns,
            meta={"consumer_id": str(args.consumer_id)})
    except Exception as e:
        # fail-soft: a broken broker must not kill the batch.
        logger.warning("publish(%s) failed: %s", REDACTION_EVENT_TOPIC, e)


def process_batch(args, plugin=None):
    """Run one batch: list, select unseen, redact each, write products, persist
    the seen-store. Returns (processed, skipped)."""
    frames = cache_consumer.list_frames(args.input)
    logger.info("Found %d frame(s) in %s", len(frames), args.input)

    seen_path = args.seen_store or _default_seen_path(args)
    seen = SeenStore(seen_path, max_size=args.seen_max, consumer_id=args.consumer_id)

    selected = select_frames(frames, seen, args)
    logger.info("Selected %d unseen frame(s) to process.", len(selected))

    processed = 0
    skipped = 0
    for frame, sidecar, uid in selected:
        try:
            audio, sr, subtype = cache_consumer.read_clip(frame)
        except Exception as e:
            # clip evicted since listing, or torn/unreadable: skip, do not crash,
            # do not mark seen (a re-run can pick it up if it reappears).
            logger.warning("Skipping %s: could not read clip (%s).", frame.name, e)
            skipped += 1
            continue

        if args.dry_run:
            logger.info("[dry-run] would redact %s", frame.name)
            processed += 1
            continue

        out_source = cache_producer.output_source_label(frame.source, args.output_source)
        try:
            clip_path, _, out_uid, windows, reason = cache_producer.write_product(
                args.output_cache, frame.capture_ts_ns, frame.vsn, out_source,
                audio, sr, subtype, sidecar, uid, PLUGIN_NAME,
                noise_fill=args.noise_fill,
                max_count=args.cache_max_count, max_bytes=args.cache_max_mb)
        except Exception as e:
            logger.exception("Failed to write product for %s (%s); skipping.",
                             frame.name, e)
            skipped += 1
            continue

        seen.add(uid, frame.capture_ts_ns)
        seen.save()
        logger.info("Redacted %s -> %s (%d window(s)%s).",
                    frame.name, os.path.basename(clip_path), len(windows),
                    ", FAIL-CLOSED" if reason else "")
        _publish_event(plugin, args, frame, out_uid, windows, reason)
        processed += 1

    logger.info("Batch done: %d processed, %d skipped, %d seen total.",
                processed, skipped, len(seen))
    return processed, skipped


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _validate(args)

    if args.every is not None:
        logger.info("--every=%s is advisory only: this plugin runs one batch and "
                    "exits. WES schedules the cadence.", args.every)

    plugin = None
    if not args.dry_run:
        try:
            from waggle.plugin import Plugin
            plugin = Plugin()
        except Exception as e:
            logger.warning("Could not open pywaggle Plugin (%s); running without "
                           "publish.", e)

    process_batch(args, plugin)


if __name__ == "__main__":
    main()

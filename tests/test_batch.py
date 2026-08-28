"""End-to-end batch tests.

Dedup-across-runs runs the real fixture cache (5 real 24-bit node frames) through
the batch twice; the clip-vanishes case uses synthetic frames so nothing real is
mutated. speech_scores is monkeypatched so no YAMNet model is loaded. The fixture
is read-only and never copied into the repo.

    python3 -m pytest tests/test_batch.py -v
"""

import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redaction.apply  # noqa: E402
import cache_consumer  # noqa: E402
import main  # noqa: E402

FIXTURE = os.path.expanduser(
    "~/AI-Projects/fixture-cache/hummingcam-audio/hummingcam_mic")


def _no_speech(*_a, **_k):
    return [0.0] * 30


def _write_frame(d, ts, vsn="H00F", source="hummingcam_mic", unique_id=None):
    name = f"{ts}-v2-{vsn}-{source}.flac"
    clip = os.path.join(d, name)
    sf.write(clip, np.zeros(1600, dtype=np.float32), 16000, format="FLAC", subtype="PCM_24")
    with open(clip + ".json", "w") as fh:
        json.dump({"schema_version": "sage-media-1", "capture_timestamp_ns": ts,
                   "object_name": name, "source": source, "vsn": vsn,
                   "unique_id": unique_id or f"uid-{ts}"}, fh)
    return clip


def _args(tmp_path, input_dir, extra=None):
    argv = ["--source", "cache", "--input", input_dir,
            "--output-cache", str(tmp_path / "out"),
            "--cache-root", str(tmp_path / "root"),
            "--all-unseen", "--cache-max-count", "100"]
    argv += extra or []
    return main.parse_args(argv)


@pytest.mark.skipif(not os.path.isdir(FIXTURE), reason="fixture cache not present")
def test_dedup_across_runs_on_real_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    args = _args(tmp_path, FIXTURE)

    # Sanity: the fixture is the expected 5 frames.
    assert len(cache_consumer.list_frames(FIXTURE)) == 5

    processed, skipped = main.process_batch(args)
    assert processed == 5 and skipped == 0

    out = str(tmp_path / "out")
    flacs = sorted(n for n in os.listdir(out) if n.endswith(".flac"))
    jsons = sorted(n for n in os.listdir(out) if n.endswith(".json"))
    assert len(flacs) == 5 and len(jsons) == 5
    # source label changed, capture_ts + vsn preserved, 24-bit preserved.
    assert all("-v2-H00F-hummingcam_mic_redacted.flac" in n for n in flacs)
    assert sf.info(os.path.join(out, flacs[0])).subtype == "PCM_24"
    # provenance recorded.
    sc = json.load(open(os.path.join(out, jsons[0])))
    assert sc["source_unique_id"] and sc["source"] == "hummingcam_mic_redacted"

    # Second run: nothing reprocessed (durable seen-store).
    processed2, skipped2 = main.process_batch(args)
    assert processed2 == 0


@pytest.mark.skipif(not os.path.isdir(FIXTURE), reason="fixture cache not present")
def test_fixture_is_not_mutated(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    before = sorted(os.listdir(FIXTURE))
    main.process_batch(_args(tmp_path, FIXTURE))
    assert sorted(os.listdir(FIXTURE)) == before  # read-only: fixture untouched


def test_clip_vanishes_mid_run(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    src = tmp_path / "in"
    src.mkdir()
    _write_frame(str(src), 100)
    _write_frame(str(src), 200)

    # Simulate the ring evicting frame 100 between listing and open: the first
    # read unlinks its own clip, then the real read raises FileNotFoundError.
    real_read = cache_consumer.read_clip
    def flaky_read(frame):
        if frame.capture_ts_ns == 100:
            os.unlink(frame.clip_path)
        return real_read(frame)
    monkeypatch.setattr(cache_consumer, "read_clip", flaky_read)

    processed, skipped = main.process_batch(_args(tmp_path, str(src)))
    assert processed == 1 and skipped == 1        # 200 processed, 100 skipped, no crash

    out = str(tmp_path / "out")
    flacs = [n for n in os.listdir(out) if n.endswith(".flac")]
    assert flacs == ["200-v2-H00F-hummingcam_mic_redacted.flac"]

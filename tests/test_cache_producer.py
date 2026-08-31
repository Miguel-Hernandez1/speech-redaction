"""Producer write-side tests: the sidecar-before-clip write order, output naming
and sidecar fields (provenance, windows, fail-closed reason), 24-bit subtype
preservation, and output ring eviction. Monkeypatches speech_scores so no YAMNet
model is needed.

    python3 -m pytest tests/test_cache_producer.py -v
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redaction.apply  # noqa: E402
import cache_producer  # noqa: E402


def _no_speech(*_a, **_k):
    return [0.0] * 30  # gate returns no windows


def _src_sidecar(ts=12345, source="hummingcam_mic", vsn="H00F", uid="srcuid"):
    return {"schema_version": "sage-media-1", "capture_timestamp_ns": ts,
            "object_name": f"{ts}-v2-{vsn}-{source}.flac", "source": source,
            "vsn": vsn, "unique_id": uid, "camera": source, "job": "sage"}


def test_output_source_label_derives_and_overrides():
    assert cache_producer.output_source_label("hummingcam_mic") == "hummingcam_mic_redacted"
    assert cache_producer.output_source_label("hummingcam_mic", "custom") == "custom"


def test_sidecar_renamed_into_place_before_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    out = str(tmp_path / "out")

    renames = []
    real_replace = os.replace
    def recording_replace(src, dst):
        renames.append(dst)
        return real_replace(src, dst)
    monkeypatch.setattr(cache_producer.os, "replace", recording_replace)

    audio = np.zeros(16000, dtype=np.float32)
    clip, sidecar, uid, windows, reason = cache_producer.write_product(
        out, 12345, "H00F", "hummingcam_mic_redacted", audio, 16000, "PCM_24",
        _src_sidecar(), "srcuid", "speech-redaction")

    finals = [r for r in renames if r in (sidecar, clip)]
    assert finals == [sidecar, clip]          # sidecar into place FIRST
    assert finals[0].endswith(".json") and finals[1].endswith(".flac")


def test_output_naming_and_sidecar_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    out = str(tmp_path / "out")
    audio = np.zeros(16000, dtype=np.float32)

    clip, sidecar, uid, windows, reason = cache_producer.write_product(
        out, 12345, "H00F", "hummingcam_mic_redacted", audio, 16000, "PCM_24",
        _src_sidecar(), "srcuid", "speech-redaction")

    # capture_ts + vsn preserved in the name; only the source label changed.
    assert os.path.basename(clip) == "12345-v2-H00F-hummingcam_mic_redacted.flac"

    sc = json.load(open(sidecar))
    assert sc["schema_version"] == "sage-media-1"
    assert sc["capture_timestamp_ns"] == 12345            # unchanged
    assert sc["object_name"] == os.path.basename(clip)
    assert sc["source"] == "hummingcam_mic_redacted"      # matches filename
    assert sc["plugin"] == "speech-redaction"
    assert sc["unique_id"] == uid and uid != "srcuid"     # fresh, over redacted bytes
    assert sc["source_unique_id"] == "srcuid"             # provenance
    assert sc["redaction_windows"] == []                  # no speech in this test
    assert sc["redaction_fail_closed"] is False
    assert sc["redaction_fail_closed_reason"] is None
    assert sc["camera"] == "hummingcam_mic"               # carried through from source


def test_subtype_preserved_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    out = str(tmp_path / "out")
    audio = np.zeros(16000, dtype=np.float32)
    clip, _sidecar, _uid, _w, _r = cache_producer.write_product(
        out, 1, "H00F", "mic_redacted", audio, 16000, "PCM_24",
        _src_sidecar(), "srcuid", "speech-redaction")
    assert sf.info(clip).subtype == "PCM_24"


def test_fail_closed_reason_recorded_in_sidecar(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise redaction.apply.RedactionFailure("model missing")
    monkeypatch.setattr(redaction.apply, "speech_scores", _boom)
    out = str(tmp_path / "out")
    audio = np.ones(16000, dtype=np.float32) * 0.5
    clip, sidecar, uid, windows, reason = cache_producer.write_product(
        out, 7, "H00F", "mic_redacted", audio, 16000, "PCM_24",
        _src_sidecar(), "srcuid", "speech-redaction")
    sc = json.load(open(sidecar))
    assert sc["redaction_fail_closed"] is True
    assert "model missing" in sc["redaction_fail_closed_reason"]


def test_product_written_under_redacted_audio_source_tree(tmp_path, monkeypatch):
    """Products land in <output-cache>/redacted_audio/<source>/, not flat in the
    root, mirroring media-sampler3's per-stream layout."""
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    out = str(tmp_path / "out")
    audio = np.zeros(16000, dtype=np.float32)
    clip, sidecar, uid, windows, reason = cache_producer.write_product(
        out, 12345, "H00F", "hummingcam_mic_redacted", audio, 16000, "PCM_24",
        _src_sidecar(), "srcuid", "speech-redaction")

    expected_dir = os.path.join(out, "redacted_audio", "hummingcam_mic_redacted")
    assert cache_producer.stream_dir(out, "hummingcam_mic_redacted") == expected_dir
    assert os.path.dirname(clip) == expected_dir      # clip is under the source tree
    assert os.path.dirname(sidecar) == expected_dir
    assert os.path.isfile(clip) and os.path.isfile(sidecar)
    # nothing was written flat into the output-cache root
    assert not any(n.endswith(".flac") for n in os.listdir(out))


def test_output_ring_evicts_oldest_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction.apply, "speech_scores", _no_speech)
    out = str(tmp_path / "out")
    audio = np.zeros(16000, dtype=np.float32)
    # cap the ring at 2; write three, oldest (ts=1) should be evicted (both files).
    for ts in (1, 2, 3):
        cache_producer.write_product(
            out, ts, "H00F", "mic_redacted", audio, 16000, "PCM_24",
            _src_sidecar(ts=ts), f"src{ts}", "speech-redaction", max_count=2)
    stream = cache_producer.stream_dir(out, "mic_redacted")   # per-source ring
    remaining = sorted(n for n in os.listdir(stream) if n.endswith(".flac"))
    assert remaining == ["2-v2-H00F-mic_redacted.flac", "3-v2-H00F-mic_redacted.flac"]
    assert not os.path.exists(os.path.join(stream, "1-v2-H00F-mic_redacted.flac"))
    assert not os.path.exists(os.path.join(stream, "1-v2-H00F-mic_redacted.flac.json"))

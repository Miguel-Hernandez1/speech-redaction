"""Consumer read-side tests: filename parsing, ordering by capture_ts (not mtime),
and the missing-sidecar fallback to filename identity. Pure filesystem + numpy;
no model needed.

    python3 -m pytest tests/test_cache_consumer.py -v
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_consumer  # noqa: E402


def _write_frame(d, ts, vsn="H00F", source="hummingcam_mic", with_sidecar=True,
                 unique_id=None, subtype="PCM_24"):
    """Write a tiny v2 frame (clip [+ sidecar]) into dir d; return the clip path."""
    name = f"{ts}-v2-{vsn}-{source}.flac"
    clip = os.path.join(d, name)
    sf.write(clip, np.zeros(1600, dtype=np.float32), 16000, format="FLAC", subtype=subtype)
    if with_sidecar:
        sidecar = {
            "schema_version": "sage-media-1", "capture_timestamp_ns": ts,
            "object_name": name, "source": source, "vsn": vsn,
            "unique_id": unique_id if unique_id is not None else f"uid-{ts}",
            "plugin": "media-sampler3:dev",
        }
        with open(clip + ".json", "w") as fh:
            json.dump(sidecar, fh)
    return clip


def test_parse_frame_name_splits_on_first_v2():
    ts, vsn, source = cache_consumer.parse_frame_name(
        "1787676167948756781-v2-H00F-hummingcam_mic.flac")
    assert ts == 1787676167948756781
    assert vsn == "H00F"
    assert source == "hummingcam_mic"


def test_parse_frame_name_rejects_non_v2_and_sidecars():
    assert cache_consumer.parse_frame_name("x.flac") is None
    assert cache_consumer.parse_frame_name("1-v2-H00F-mic.flac.json") is None
    assert cache_consumer.parse_frame_name("notanint-v2-H00F-mic.flac") is None


def test_list_frames_orders_by_capture_ts_not_mtime(tmp_path):
    d = str(tmp_path)
    clips = {ts: _write_frame(d, ts) for ts in (100, 200, 300)}
    # Reverse mtimes vs capture_ts: newest mtime on the OLDEST capture_ts.
    os.utime(clips[300], (1, 1))
    os.utime(clips[200], (2, 2))
    os.utime(clips[100], (3, 3))

    frames = cache_consumer.list_frames(d)
    assert [f.capture_ts_ns for f in frames] == [100, 200, 300]  # by ts, not mtime


def test_list_frames_ignores_sidecars_and_dotfiles(tmp_path):
    d = str(tmp_path)
    _write_frame(d, 100)
    open(os.path.join(d, ".tmp-clip-abc.flac"), "w").close()  # dotfile, ignored
    frames = cache_consumer.list_frames(d)
    assert len(frames) == 1
    assert frames[0].name.endswith(".flac")


def test_missing_sidecar_falls_back_to_filename_identity(tmp_path):
    d = str(tmp_path)
    _write_frame(d, 500, with_sidecar=False)
    frame = cache_consumer.list_frames(d)[0]
    sidecar = cache_consumer.load_sidecar(frame)
    assert sidecar is None                                  # missing -> None
    uid = cache_consumer.frame_unique_id(frame, sidecar)
    assert uid == frame.name                                # filename fallback


def test_corrupt_sidecar_falls_back_to_filename_identity(tmp_path):
    d = str(tmp_path)
    clip = _write_frame(d, 600, with_sidecar=False)
    with open(clip + ".json", "w") as fh:
        fh.write("{ this is not valid json")
    frame = cache_consumer.list_frames(d)[0]
    assert cache_consumer.load_sidecar(frame) is None
    assert cache_consumer.frame_unique_id(frame, None) == frame.name


def test_read_clip_reports_subtype_and_mono(tmp_path):
    d = str(tmp_path)
    _write_frame(d, 700, subtype="PCM_24")
    frame = cache_consumer.list_frames(d)[0]
    audio, sr, subtype = cache_consumer.read_clip(frame)
    assert sr == 16000
    assert subtype == "PCM_24"
    assert audio.ndim == 1

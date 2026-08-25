"""FLAC metadata round-trip tests for flac_redaction.write_redacted_flac.

Needs soundfile + mutagen (unlike test_redaction.py, which is numpy-only).

    python3 -m pytest tests/test_flac_redaction.py -v
"""

import json
import os
import sys

import numpy as np
import pytest

# Repo-root on sys.path so `from redaction.X import ...` and `import flac_redaction`
# resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaction.redaction_gate import RedactionGate  # noqa: E402
from redaction import yamnet_speech  # noqa: E402
import redaction.apply  # noqa: E402
from flac_redaction import (  # noqa: E402
    write_redacted_flac, REDACTION_WINDOWS_FIELD, REDACTION_REASON_FIELD,
)


def _fake_scores(scores):
    def _f(audio_1d, samplerate, include_ambiguous=False):
        return list(scores)
    return _f


def test_write_redacted_flac_roundtrips_windows(tmp_path, monkeypatch):
    """Success path: a FLAC is written and the (start_s, duration_s) windows
    round-trip out of the single Vorbis-comment field; no reason field is set."""
    from mutagen.flac import FLAC
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_scores([0.0, 0.9, 0.0]))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=1.0, frame_duration=1.0)
    audio = np.ones(3000, dtype=np.float32) * 0.5
    out = str(tmp_path / "clip.flac")

    ret_path, windows, reason = write_redacted_flac(audio, 1000, out, gate=gate)

    assert ret_path == out and os.path.exists(out)
    assert reason is None
    assert windows == [(1.0, 2.0)]

    tags = FLAC(out)
    assert REDACTION_REASON_FIELD not in tags          # no reason on success
    stored = json.loads(tags[REDACTION_WINDOWS_FIELD][0])
    assert stored == [[1.0, 1.0]]                       # (start_s, duration_s)


def test_write_redacted_flac_stores_fail_closed_reason(tmp_path, monkeypatch):
    """Fail-closed path: the whole-buffer window round-trips AND the reason is
    stored in metadata."""
    from mutagen.flac import FLAC

    def _raising(audio_1d, samplerate, include_ambiguous=False):
        raise yamnet_speech.RedactionFailure("model missing")

    monkeypatch.setattr(redaction.apply, "speech_scores", _raising)
    audio = np.ones(16000, dtype=np.float32) * 0.5     # 1.0s @ 16k
    out = str(tmp_path / "fc.flac")

    ret_path, windows, reason = write_redacted_flac(audio, 16000, out, gate=RedactionGate())

    assert reason is not None and "model missing" in reason
    assert windows == [(0.0, 1.0)]
    assert os.path.exists(out)

    tags = FLAC(out)
    stored = json.loads(tags[REDACTION_WINDOWS_FIELD][0])
    assert stored == [[0.0, 1.0]]                       # whole buffer, (start, duration)
    assert tags[REDACTION_REASON_FIELD][0] == reason


def test_write_redacted_flac_preserves_24bit_source_subtype(tmp_path, monkeypatch):
    """A 24-bit (PCM_24) source round-trips as 24-bit when the caller passes the
    source's sf.info(path).subtype through. media-sampler3's clips are 24-bit
    FLAC; the PCM_16 default would silently drop 8 bits."""
    import soundfile as sf

    # Build a 24-bit FLAC "source" and recover its subtype the way a real caller
    # would, via sf.info(path).subtype.
    src = str(tmp_path / "source.flac")
    src_audio = ((np.arange(4000, dtype=np.float32) % 100) - 50) / 100.0
    sf.write(src, src_audio, 16000, format="FLAC", subtype="PCM_24")
    src_subtype = sf.info(src).subtype
    assert src_subtype == "PCM_24"

    # No speech -> redaction leaves the buffer as-is; the subtype is the point.
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_scores([0.0, 0.0, 0.0]))
    audio, sr = sf.read(src, dtype="float32")
    out = str(tmp_path / "redacted.flac")

    ret_path, windows, reason = write_redacted_flac(audio, sr, out, subtype=src_subtype)

    assert reason is None
    assert sf.info(out).subtype == "PCM_24"  # preserved, not downgraded to PCM_16


def test_write_redacted_flac_defaults_to_pcm16_when_subtype_unknown(tmp_path, monkeypatch):
    """With no subtype passed (source unknown), PCM_16 is the fallback."""
    import soundfile as sf
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_scores([0.0, 0.0, 0.0]))
    audio = np.ones(4000, dtype=np.float32) * 0.1
    out = str(tmp_path / "default.flac")

    write_redacted_flac(audio, 16000, out)  # no subtype argument

    assert sf.info(out).subtype == "PCM_16"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

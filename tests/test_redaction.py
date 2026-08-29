"""
Unit tests for the redaction package — run locally, no GPU / no pywaggle /
no LiteRT model file required. The LiteRT-backed path (yamnet_speech /
redact_speech calling the real .tflite) is exercised with monkeypatching only;
the model file itself is NOT loaded here.

    python3 -m pytest tests/test_redaction.py -v
    # or, dependency-free:
    python3 tests/test_redaction.py
"""

import os
import sys

import numpy as np
import pytest

# Repo-root on sys.path so `from redaction.X import ...` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redaction.redaction_gate import RedactionGate, RedactionGateFailure  # noqa: E402
from redaction.speech_classes import (  # noqa: E402
    AMBIGUOUS, CORE_SPEECH, NUM_YAMNET_CLASSES, speech_score,
)
from redaction.apply import redact_speech  # noqa: E402
from redaction import yamnet_speech  # noqa: E402


# ── speech_classes (indices verified against canonical CSV this session) ──────

def test_core_speech_index_count():
    assert len(CORE_SPEECH) == 12


def test_ambiguous_index_count():
    assert len(AMBIGUOUS) == 4


def test_no_overlap_between_core_and_ambiguous():
    assert set(CORE_SPEECH).isdisjoint(AMBIGUOUS)


def test_speech_score_uses_core_only_by_default():
    scores = [0.0] * NUM_YAMNET_CLASSES
    scores[0] = 0.7    # Speech
    scores[64] = 0.9   # Crowd (ambiguous) — ignored by default
    assert speech_score(scores) == 0.7


def test_speech_score_raises_on_wrong_length_vector():
    with pytest.raises(ValueError):
        speech_score([0.0] * 10)


# ── RedactionGate (pure-Python hysteresis — tests ported from notes-ref) ──────

def test_isolated_spike():
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=1.0, hangover_seconds=1.0,
                         post_roll_seconds=1.0, frame_hop=1.0, frame_duration=1.0)
    assert gate.get_redaction_windows([0.0, 0.0, 0.9, 0.0, 0.0]) == [(1.0, 4.0)]


def test_sustained_speech():
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=1.0, hangover_seconds=1.0,
                         post_roll_seconds=1.0, frame_hop=1.0, frame_duration=1.0)
    assert gate.get_redaction_windows([0.0, 0.0] + [0.9] * 5 + [0.0] * 3) == [(1.0, 8.0)]


def test_flickering_scores_merge_into_one_window():
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=1.0,
                         post_roll_seconds=1.0, frame_hop=1.0, frame_duration=1.0)
    assert gate.get_redaction_windows([0.9, 0.2, 0.9, 0.2, 0.9]) == [(0.0, 5.0)]


def test_empty_scores_fail_closed_by_default():
    with pytest.raises(RedactionGateFailure):
        RedactionGate().get_redaction_windows([])


def test_empty_scores_can_opt_into_fail_open():
    gate = RedactionGate(fail_closed=False)
    assert gate.get_redaction_windows([]) == []


# ── apply.redact_speech — the privacy-critical contract ──────────────────────
#
# These tests are the heart of the integration verification: redact_speech must
# (a) zero the windows returned by the gate, in place, on the success path;
# (b) NEVER return the raw unredacted buffer on ANY failure path — it must zero
#     the whole buffer and surface the failure.
# We monkeypatch yamnet_speech.speech_scores so no model file is needed.

def _fake_speech_scores_factory(scores):
    """Return a fake speech_scores replacement that ignores its audio args
    and returns the given per-frame scores (mimicking yamnet_speech.speech_scores)."""
    def _fake(audio_1d, samplerate, include_ambiguous=False):
        return list(scores)
    return _fake


def test_redact_speech_zeros_windows_in_place_on_success(monkeypatch):
    # 3s of audio @ 1000 Hz; scores [0, 0.9, 0] with hop=1.0/dur=1.0 means frame 1
    # covers [1.0s, 2.0s] and the gate returns window (1.0, 2.0). That maps to
    # samples [1000, 2000) at 1000 Hz.
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([0.0, 0.9, 0.0]))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=1.0, frame_duration=1.0)
    audio = np.ones(3000, dtype=np.float32) * 0.5  # 3s @ 1000 Hz
    original_first = audio[0]
    redacted, windows, reason = redact_speech(audio, 1000, gate=gate)
    assert reason is None
    assert windows == [(1.0, 2.0)]
    assert redacted is audio  # in place — same array object
    # window (1.0, 2.0) maps to samples [1000, 2000) at 1000 Hz — zeroed.
    assert redacted[1000:2000].sum() == 0.0
    # Untouched regions preserved.
    assert redacted[0] == original_first          # first sample intact
    assert redacted[800] == original_first        # pre-window intact
    assert redacted[2500] == original_first        # post-window intact


def test_redact_speech_fail_closed_on_yamnet_error(monkeypatch):
    """If YAMNet raises, redact_speech must zero the ENTIRE buffer and return
    a fail-closed reason — it must NEVER return the original array unmodified."""
    def _raising_speech_scores(audio_1d, samplerate, include_ambiguous=False):
        raise yamnet_speech.RedactionFailure("simulated YAMNet inference crash")

    # NOTE: apply.py does `from .yamnet_speech import speech_scores`, binding the
    # name into its own namespace. Patching redaction.yamnet_speech.speech_scores
    # would not affect what apply.redact_speech actually calls. Patch the name
    # inside redaction.apply instead.
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores", _raising_speech_scores)
    gate = RedactionGate()
    audio = np.ones(24000, dtype=np.float32) * 0.5  # 1.5s @ 16k simulating mic
    raw_sum = audio.sum()
    redacted, windows, reason = redact_speech(audio, 16000, gate=gate)
    # Critical: fail closed, the raw array did NOT leak.
    assert redacted.sum() == 0.0
    assert redacted.sum() != raw_sum
    assert reason is not None and "simulated YAMNet inference crash" in reason
    assert windows == [(0.0, 1.5)]  # whole buffer


def test_redact_speech_fail_closed_on_empty_scores(monkeypatch):
    """Empty scores -> RedactionGate(fail_closed=True) raises RedactionGateFailure;
    redact_speech must catch it and zero the whole buffer, not let raw audio through."""
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([]))
    gate = RedactionGate()  # fail_closed=True default
    audio = np.ones(24000, dtype=np.float32) * 0.5
    redacted, windows, reason = redact_speech(audio, 16000, gate=gate)
    assert redacted.sum() == 0.0
    assert reason is not None
    # Reason message comes from RedactionGateFailure("no classification scores...").
    assert "no classification scores" in reason.lower()


def test_redact_speech_ceil_captures_tail_sample(monkeypatch):
    """Off-by-one fix: a window whose end lands on a fractional sample must be
    ceil'd so the last partial sample of speech is zeroed, not truncated away."""
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([0.0, 0.9, 0.0]))
    # post_roll 0.0005s pushes the window end to 2.0005s; at 1000 Hz that is
    # sample 2000.5 -> ceil 2001, so sample index 2000 must be zeroed (int() would
    # have truncated to 2000 and left it).
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0005, frame_hop=1.0, frame_duration=1.0)
    audio = np.ones(3000, dtype=np.float32) * 0.5
    redacted, windows, reason = redact_speech(audio, 1000, gate=gate)
    assert reason is None
    assert len(windows) == 1 and windows[0][0] == 1.0
    assert redacted[2000] == 0.0     # ceil captured this tail sample
    assert redacted[999] == 0.5      # floor left the pre-window sample intact
    assert redacted[2001] == 0.5     # audio past ceil(end) untouched


def test_redact_speech_noise_fill_matches_surrounding_level(monkeypatch):
    """noise_fill=True writes non-zero audio into the window at roughly the level
    of the surrounding audio (constant 0.3 -> RMS ~0.3), not silence."""
    import redaction.apply
    # frame 2 active -> window (2.0, 3.0) -> samples [2000, 3000) at 1000 Hz.
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([0.0, 0.0, 0.9, 0.0, 0.0]))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=1.0, frame_duration=1.0)
    audio = np.full(5000, 0.3, dtype=np.float32)  # constant 0.3 => RMS 0.3
    redacted, windows, reason = redact_speech(audio, 1000, gate=gate, noise_fill=True)
    assert reason is None
    assert windows == [(2.0, 3.0)]
    filled = redacted[2000:3000]
    assert np.any(filled != 0.0)               # noise, not zeros
    assert not np.allclose(filled, filled[0])  # actually varying (noise, not a constant)
    fill_rms = float(np.sqrt(np.mean(filled.astype(np.float64) ** 2)))
    assert 0.15 < fill_rms < 0.45              # ~ surrounding level 0.3


def test_redact_speech_noise_fill_is_two_pass_measures_original_audio(monkeypatch):
    """Prove the noise fill measures levels in a FIRST pass, before ANY window is
    written. Two windows sit within one guard region of each other; window one
    originally holds LOUD audio in otherwise-quiet ambient.

    - Two passes (correct): window two's guard overlaps window one's ORIGINAL
      loud samples, so window two's fill comes out clearly elevated.
    - One pass (buggy): window one would already be filled down to the ambient
      level, so window two would measure ~ambient and fill quiet.

    A constant-level buffer canNOT distinguish the two (window one's fill equals
    the ambient it was measured from, so a one-pass read of it is identical), so
    the loud original window is what makes the ordering observable.
    """
    from redaction.apply import _NOISE_MEASURE_SECONDS
    import redaction.apply

    sr = 1000
    guard = int(_NOISE_MEASURE_SECONDS * sr)  # 100 samples

    # frames 20 and 22 active (frame 21 silent) with hop=dur=0.01 and no padding
    # -> windows (0.20, 0.21) and (0.22, 0.23) -> samples [200,210) and [220,230).
    scores = [0.0] * 25
    scores[20] = 0.9
    scores[22] = 0.9
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory(scores))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=0.01, frame_duration=0.01)

    ambient = 0.3
    audio = np.full(600, ambient, dtype=np.float32)
    audio[200:210] = 3.0  # window one originally LOUD (real speech), quiet ambient

    # Precondition: the gap between the windows is inside one guard region, so a
    # one-pass fill of window one WOULD pollute window two's measurement.
    gap_samples = 220 - 210
    assert gap_samples < guard

    redacted, windows, reason = redact_speech(audio, sr, gate=gate, noise_fill=True)
    assert reason is None
    assert len(windows) == 2  # two separate windows (float boundaries ~0.20..0.23)
    assert windows[0][0] == pytest.approx(0.20) and windows[0][1] == pytest.approx(0.21)
    assert windows[1][0] == pytest.approx(0.22) and windows[1][1] == pytest.approx(0.23)

    def _rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    fill1_rms = _rms(redacted[200:210])
    fill2_rms = _rms(redacted[220:230])

    # Window one fills to ~ambient (its own guard is all ambient); its loud
    # original is destroyed.
    assert 0.2 < fill1_rms < 0.4
    # Window two's fill is ELEVATED because the two-pass measurement read window
    # one's ORIGINAL loud samples. A one-pass implementation would give ~ambient
    # here (~0.3) and fail this assertion; the analytic two-pass value is ~0.73.
    assert fill2_rms > 0.5
    assert fill2_rms > fill1_rms * 1.5


def test_redact_speech_fail_closed_noise_fills_whole_buffer(monkeypatch):
    """Fail-closed with noise_fill=True: the WHOLE buffer is overwritten with
    noise at the pre-fill level (no raw sample survives) and a reason is set."""
    def _raising(audio_1d, samplerate, include_ambiguous=False):
        raise yamnet_speech.RedactionFailure("boom")
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores", _raising)
    audio = np.full(16000, 0.5, dtype=np.float32)  # 1.0s @ 16k, constant 0.5
    redacted, windows, reason = redact_speech(audio, 16000, gate=RedactionGate(),
                                              noise_fill=True)
    assert reason is not None and "boom" in reason
    assert windows == [(0.0, 1.0)]              # whole buffer
    assert not np.allclose(redacted, 0.5)       # raw audio destroyed
    assert np.any(redacted != 0.0)              # filled with noise, not zeros
    rms = float(np.sqrt(np.mean(redacted.astype(np.float64) ** 2)))
    assert 0.3 < rms < 0.7                       # ~ pre-fill level 0.5


def test_redact_speech_no_speech_leaves_buffer_untouched(monkeypatch):
    """Success path with NO speech detected: windows is [], reason is None, and
    the buffer is NOT modified — that's correct (ambient audio, no redaction)."""
    import redaction.apply
    monkeypatch.setattr(redaction.apply, "speech_scores",
                        _fake_speech_scores_factory([0.0, 0.0, 0.0, 0.0]))
    gate = RedactionGate(enter_threshold=0.5, exit_threshold=0.3,
                         pre_roll_seconds=0.0, hangover_seconds=0.0,
                         post_roll_seconds=0.0, frame_hop=1.0, frame_duration=1.0)
    audio = np.ones(4000, dtype=np.float32) * 0.5
    pre_sum = audio.sum()
    redacted, windows, reason = redact_speech(audio, 1000, gate=gate)
    assert reason is None
    assert windows == []
    assert redacted.sum() == pre_sum  # untouched — ambient preserved


def test_model_path_is_repo_relative_not_parent_of_repo():
    """Bug 1: the resolver must look for the model inside the repo's own models/
    dir, not one directory up (the parent of the repo). Running from a bare clone
    at .../speech-redaction, the model is at .../speech-redaction/models/, and the
    old hardcoded .../AI-Projects/models/ pointed at the parent."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(yamnet_speech.__file__)))
    expected = os.path.join(repo_root, "models", "yamnet.tflite")
    assert expected in yamnet_speech._YAMNET_TFLITE_PATHS   # inside the repo
    # regression: the parent-of-repo path (the old off-by-one) must NOT be used.
    parent_of_repo = os.path.dirname(repo_root)
    buggy = os.path.join(parent_of_repo, "models", "yamnet.tflite")
    assert buggy not in yamnet_speech._YAMNET_TFLITE_PATHS


def test_missing_ai_edge_litert_fails_closed(tmp_path, monkeypatch):
    """Bug 2: a missing ai_edge_litert must fail closed (zero the buffer with a
    reason naming the package), not crash the batch with ModuleNotFoundError."""
    # Point the resolver at a real (dummy) file so we get PAST the missing-file
    # check and reach the deferred import inside _load_model.
    dummy = tmp_path / "yamnet.tflite"
    dummy.write_bytes(b"not a real model")
    monkeypatch.setattr(yamnet_speech, "YAMNET_TFLITE_PATH", str(dummy))
    monkeypatch.setattr(yamnet_speech, "_interp", None)          # force a fresh load
    monkeypatch.setattr(yamnet_speech, "_interp_model_path", None)
    # Simulate the package being absent even if it happens to be installed here.
    monkeypatch.setitem(sys.modules, "ai_edge_litert", None)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", None)

    audio = np.ones(16000, dtype=np.float32) * 0.5              # 1.0s @ 16k
    redacted, windows, reason = redact_speech(audio, 16000)     # must NOT raise
    assert reason is not None and "ai_edge_litert" in reason
    assert redacted.sum() == 0.0                                 # failed closed
    assert windows == [(0.0, 1.0)]                              # whole buffer


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

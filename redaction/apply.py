"""Apply speech redaction to an in-memory audio array.

The privacy-critical entry point used by ``record_from_microphone``:
takes the raw ``AudioSample.data`` numpy array, runs YAMNet speech scoring +
RedactionGate to compute speech windows, and ZEROS those sample ranges IN
PLACE-then-return so the caller persists only the redacted array.

Invariant (enforced by this module):
  On ANY failure (YAMNet error, model missing, RedactionGate empty-scores),
  this module either (a) returns a fully-zeroed buffer, or (b) raises
  ``RedactionFailure`` -- it NEVER returns the original unredacted array
  and NEVER returns a "no redaction" result on an error.
"""

import inspect
import math

import numpy as np

from .redaction_gate import RedactionGate, RedactionGateFailure
from .yamnet_speech import RedactionFailure, speech_scores

# Default gate parameters -- derived from RedactionGate's own constructor
# defaults so there is exactly one place that defines the shipped config
# (previously this dict duplicated those values by hand and could drift out
# of sync with them). Biased hard toward recall (under-redacting is
# catastrophic). Overridable via a future CLI flag (not added in this change
# -- per REDACTION-INTEGRATION-NOTES.md §4 step 7).
_DEFAULT_GATE = {
    name: param.default
    for name, param in inspect.signature(RedactionGate.__init__).parameters.items()
    if name != "self"
}

# White-noise fill (opt-in via redact_speech(noise_fill=True)). Length of the
# guard region measured on each side of a window to match the fill level to the
# surrounding audio. Array math only -- no file I/O -- so it stays in redaction/.
_NOISE_MEASURE_SECONDS = 0.1


def _rms(samples) -> float:
    """Root-mean-square level of a 1-D array (float64 accumulation). 0.0 if empty."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _guard_rms(audio_1d, i0: int, i1: int, samplerate: int) -> float:
    """RMS of a short region just before and just after [i0, i1), read from the
    CURRENT contents of audio_1d.

    Must be called in a measurement pass BEFORE any window is overwritten, or an
    adjacent window's already-written fill would be measured instead of the
    original audio. Falls back to the window's own span only when there is no
    surrounding audio (window covers the whole buffer).
    """
    n = audio_1d.size
    guard = max(1, int(_NOISE_MEASURE_SECONDS * samplerate))
    before = audio_1d[max(0, i0 - guard):i0]
    after = audio_1d[i1:min(n, i1 + guard)]
    if before.size or after.size:
        return _rms(np.concatenate([before, after]))
    return _rms(audio_1d[i0:i1])


def _fill_noise(audio_1d, i0: int, i1: int, target_rms: float, rng) -> None:
    """Overwrite [i0, i1) IN PLACE with white noise scaled to target_rms.

    A zero/near-zero target writes silence (there is no level to match).
    """
    length = i1 - i0
    if length <= 0:
        return
    if target_rms <= 0.0:
        audio_1d[i0:i1] = 0.0
        return
    noise = rng.standard_normal(length)
    cur = _rms(noise)
    if cur > 0.0:
        noise = noise * (target_rms / cur)
    audio_1d[i0:i1] = noise


def redact_speech(audio_1d, samplerate: int, gate: RedactionGate | None = None,
                  noise_fill: bool = False):
    """Zero out speech windows in an audio array. IN-PLACE.

    Args:
        audio_1d: 1-D numpy float array (the AudioSample.data buffer). Modified
            in place AND returned. Caller must NOT have written this array to
            disk yet.
        samplerate: integer sampling rate of audio_1d.
        gate: optional RedactionGate instance (else the notes' defaults are used).
        noise_fill: when False (default) redacted regions are zeroed, exactly as
            before. When True, each redacted region is overwritten with white
            noise whose level matches the audio just around it (measured in a
            first pass, before any region is written), and the fail-closed path
            fills the whole buffer with noise at the buffer's pre-fill level
            instead of zeros. Either way the raw audio is destroyed in place.

    Returns:
        (redacted_audio, windows, fail_closed_reason) where:
          - ``redacted_audio`` is the SAME array object, zeroed across the
            redaction windows. It is NEVER the raw unredacted buffer on an
            error path.
          - ``windows`` is a list of (start_s, end_s) float pairs, always
            (float, float). On a fail-closed path this is [(0.0, duration)].
          - ``fail_closed_reason`` is None on the normal path, or a string
            describing the failure when the whole buffer was zeroed because
            YAMNet / RedactionGate could not run. The caller should log it.

    Failure handling (fail-closed):
      If speech_scores or RedactionGate raises (model missing, inference
      error, empty scores with fail_closed=True), the ENTIRE buffer is zeroed
      and returned with windows=[(0.0, duration)] and a non-None reason. The
      raw array NEVER leaves this function on an error path. Privacy posture:
      if we cannot classify speech, assume all of it is speech.
    """
    gate = gate or RedactionGate(**_DEFAULT_GATE)
    n = audio_1d.size
    duration_s = n / float(samplerate) if samplerate else 0.0

    try:
        scores = speech_scores(audio_1d, samplerate)
        windows = gate.get_redaction_windows(scores)
    except (RedactionFailure, RedactionGateFailure) as e:
        # Fail closed: destroy the ENTIRE buffer. No raw audio leaves this
        # function on an error path. Caller logs the reason and may publish a
        # redaction event measurement covering the whole capture.
        if noise_fill:
            # Measure the level BEFORE overwriting (same two-pass reason: you
            # cannot measure a region you have already filled), then replace the
            # whole buffer with noise at that level.
            level = _rms(audio_1d)
            _fill_noise(audio_1d, 0, n, level, np.random.default_rng())
        else:
            audio_1d.fill(0.0)
        return audio_1d, [(0.0, duration_s)], str(e)

    # Zero each window in place. Windows are in seconds (from YAMNet's 0.48s
    # hop / 0.96s frames); convert back to sample indices at the ORIGINAL
    # samplerate (48k or whatever the mic was set to) -- speech_scores already
    # handled the 48k->16k resample internally, windows are in wall-clock time.
    # Map each window (wall-clock seconds) to sample indices. floor the start and
    # ceil the end so a fractional-sample boundary never leaves up to one sample
    # of speech at a window's tail; clamp to [0, n].
    idx_windows = [
        (max(0, math.floor(start_s * samplerate)), min(n, math.ceil(end_s * samplerate)))
        for start_s, end_s in windows
    ]

    if noise_fill:
        # TWO PASSES. Pass 1: measure every target level from the still-untouched
        # audio. Pass 2: write the noise. Measuring and writing in one loop would
        # let an adjacent window measure a region a previous iteration overwrote.
        targets = [_guard_rms(audio_1d, i0, i1, samplerate) for i0, i1 in idx_windows]
        rng = np.random.default_rng()
        for (i0, i1), target in zip(idx_windows, targets):
            if i0 < i1:
                _fill_noise(audio_1d, i0, i1, target, rng)
    else:
        for i0, i1 in idx_windows:
            if i0 < i1:
                audio_1d[i0:i1] = 0.0

    return audio_1d, windows, None

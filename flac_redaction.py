"""FLAC writing for redacted audio.

Kept OUT of the redaction/ package so that package stays free of audio file I/O
(it may still read the YAMNet model file). write_redacted_flac() runs the
in-memory redaction (redaction.apply.redact_speech) and then persists the result as a
FLAC, attaching the redaction windows (and any fail-closed reason) as
Vorbis-comment metadata so a downstream consumer can see what was redacted
without re-running detection.
"""

import json

import numpy as np

from redaction.apply import redact_speech

# Single Vorbis-comment fields (one per concern), read back with json.loads.
REDACTION_WINDOWS_FIELD = "REDACTION_WINDOWS"
REDACTION_REASON_FIELD = "REDACTION_FAIL_CLOSED_REASON"


def write_redacted_flac(audio_1d, samplerate, out_path, gate=None, noise_fill=False,
                        subtype=None):
    """Redact speech in audio_1d and write the result to out_path as FLAC.

    Runs redaction.apply.redact_speech (fail-closed; mutates the array in place),
    writes the redacted audio to out_path as FLAC via soundfile, and attaches
    metadata via mutagen:
      - the redaction windows, converted from (start_s, end_s) to
        (start_s, duration_s) pairs, stored as a JSON list in ONE Vorbis comment
        field (REDACTION_WINDOWS_FIELD);
      - the fail-closed reason string in REDACTION_REASON_FIELD, only when
        redact_speech reported one (reason is not None).

    The FLAC subtype (bit depth) is controlled by ``subtype``. It defaults to
    None, meaning the source subtype is unknown and PCM_16 is used only as a
    fallback. A caller that read the audio from a file should pass
    ``sf.info(path).subtype`` so the source bit depth is preserved rather than
    silently downgraded: the media-sampler3 clips, for example, are 24-bit FLAC
    (sf.info reports "PCM_24"), and hardcoding PCM_16 would drop 8 bits.

    Returns (out_path, windows, reason): the (start_s, end_s) window list and the
    fail-closed reason (or None) exactly as redact_speech returned them.

    Integrity note: the metadata is a SECOND write. soundfile writes and closes
    the FLAC, then mutagen reopens the file and saves the tags. A crash between
    the two writes leaves a redacted FLAC with NO redaction record. This is not a
    privacy failure -- the audio on disk is already redacted -- but it is an
    integrity gap: a reader could get redacted audio with no windows/reason
    metadata attached. Left as-is by design; documented here rather than
    restructured into a single atomic write.
    """
    import soundfile as sf
    from mutagen.flac import FLAC

    redacted, windows, reason = redact_speech(
        audio_1d, samplerate, gate=gate, noise_fill=noise_fill)

    # Preserve the source bit depth. FLAC is integer-backed at any depth, so a
    # 24-bit (PCM_24) source must be written as PCM_24 or it is silently
    # downgraded. PCM_16 is only the fallback for when the source subtype is
    # unknown (subtype is None).
    write_subtype = subtype if subtype is not None else "PCM_16"

    # Only the already-redacted array is persisted. Clamp to [-1.0, 1.0] first:
    # FLAC is integer-backed, so out-of-range samples -- e.g. loud noise-fill
    # peaks -- would clip silently. Clip a copy so the caller's (already-redacted)
    # buffer is not mutated again.
    to_write = np.clip(redacted, -1.0, 1.0)
    sf.write(out_path, to_write, samplerate, format="FLAC", subtype=write_subtype)

    # (start_s, end_s) -> (start_s, duration_s), stored as JSON in a single field.
    duration_windows = [[start_s, end_s - start_s] for start_s, end_s in windows]
    tags = FLAC(out_path)
    tags[REDACTION_WINDOWS_FIELD] = json.dumps(duration_windows)
    if reason is not None:
        tags[REDACTION_REASON_FIELD] = reason
    tags.save()

    return out_path, windows, reason

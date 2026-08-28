"""Read side of the v2 cache consumer.

Lists frames from a media-sampler3-style local cache, orders them by the
capture_ts_ns encoded in the FILENAME (never mtime), and reads authoritative
per-frame metadata from the .json sidecar with a fail-soft fallback to
filename-derived identity.

Frame naming (media-sampler3, sage-media-1):
    <capture_ts_ns>-v2-<vsn>-<source>.flac
    <capture_ts_ns>-v2-<vsn>-<source>.flac.json   (sidecar, written FIRST)

The producer writes the sidecar before the clip, so a clip that exists is
guaranteed to have a sidecar. The ring evicts oldest-first, so a clip listed a
moment ago can be gone by the time we open it: read_clip surfaces that as a
plain error the caller skips.
"""

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("speech-redaction.consumer")

_V2_SEP = "-v2-"
_FLAC_EXT = ".flac"


@dataclass
class Frame:
    capture_ts_ns: int
    vsn: str
    source: str
    clip_path: str
    sidecar_path: str

    @property
    def name(self):
        return os.path.basename(self.clip_path)


def parse_frame_name(name):
    """Parse '<ts>-v2-<vsn>-<source>.flac' into (capture_ts_ns, vsn, source).

    Splits on the FIRST '-v2-'. The right side is '<vsn>-<source>'; vsn carries
    no '-', source may carry '_'. Returns None when the name is not a v2 .flac
    clip or the timestamp prefix is not an integer.
    """
    if not name.endswith(_FLAC_EXT):
        return None
    stem = name[: -len(_FLAC_EXT)]
    if _V2_SEP not in stem:
        return None
    ts_str, rest = stem.split(_V2_SEP, 1)
    if not ts_str.isdigit():
        return None
    if "-" in rest:
        vsn, source = rest.split("-", 1)
    else:
        vsn, source = rest, ""
    return int(ts_str), vsn, source


def list_frames(input_dir):
    """Return a Frame for every v2 .flac clip in input_dir, sorted oldest-first by
    the capture_ts_ns in the FILENAME (not mtime, which lies if the clock stepped).

    Ignores sidecars, temp files, dotfiles, and non-matching names. A missing dir
    returns []. The directory can change under us; that only means a frame is
    skipped later, at read time.
    """
    try:
        names = os.listdir(input_dir)
    except FileNotFoundError:
        return []
    frames = []
    for name in names:
        if name.startswith("."):
            continue
        parsed = parse_frame_name(name)
        if parsed is None:
            continue
        ts, vsn, source = parsed
        clip = os.path.join(input_dir, name)
        frames.append(Frame(ts, vsn, source, clip, clip + ".json"))
    frames.sort(key=lambda f: f.capture_ts_ns)
    return frames


def load_sidecar(frame):
    """Return the sidecar dict for frame, or None if missing/unreadable/corrupt.

    Fail soft: the caller falls back to filename-derived identity on None.
    """
    try:
        with open(frame.sidecar_path, "r") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("Sidecar %s unreadable (%s); falling back to filename "
                       "identity.", frame.sidecar_path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Sidecar %s is not a JSON object; ignoring.",
                       frame.sidecar_path)
        return None
    return data


def frame_unique_id(frame, sidecar):
    """Dedup key for a frame: the sidecar unique_id when available, else a
    filename-derived synthetic key (the clip basename) as the fail-soft fallback."""
    if sidecar and sidecar.get("unique_id"):
        return str(sidecar["unique_id"])
    return frame.name


def read_clip(frame):
    """Read frame's FLAC into (audio float32 mono, samplerate, subtype).

    Raises FileNotFoundError if the clip has been evicted since it was listed,
    or a soundfile error if it is torn; the caller skips the frame on either.
    Multichannel is downmixed to mono.
    """
    import numpy as np
    import soundfile as sf
    if not os.path.exists(frame.clip_path):
        raise FileNotFoundError(frame.clip_path)
    subtype = sf.info(frame.clip_path).subtype
    audio, sr = sf.read(frame.clip_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype("float32")
    return audio, int(sr), subtype

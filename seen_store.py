"""Durable, bounded seen-store keyed on unique_id.

Lets the batch consumer skip frames it already processed, across process
restarts. Stored as a JSON file (default under <cache-root>/.state/ to match the
rest of the v2 family). Bounded: keeps the newest max_size entries by
capture_ts_ns and trims the oldest. Saved atomically (temp + fsync + rename).
"""

import json
import logging
import os
import tempfile

logger = logging.getLogger("speech-redaction.seen")

DEFAULT_MAX = 100_000


class SeenStore:
    def __init__(self, path, max_size=DEFAULT_MAX, consumer_id="speech-redaction"):
        self.path = path
        self.max_size = max_size
        self.consumer_id = consumer_id
        self._entries = {}          # unique_id -> capture_ts_ns
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.warning("Seen-store %s unreadable (%s); starting empty. Frames "
                           "may be reprocessed once.", self.path, e)
            return
        entries = data.get("entries", {}) if isinstance(data, dict) else {}
        if isinstance(entries, dict):
            for uid, ts in entries.items():
                try:
                    self._entries[str(uid)] = int(ts)
                except (TypeError, ValueError):
                    continue

    def contains(self, unique_id):
        return str(unique_id) in self._entries

    def add(self, unique_id, capture_ts_ns):
        self._entries[str(unique_id)] = int(capture_ts_ns)

    def _trim(self):
        if len(self._entries) <= self.max_size:
            return
        # keep the newest max_size by capture_ts_ns
        newest = sorted(self._entries.items(), key=lambda kv: kv[1])[-self.max_size:]
        self._entries = dict(newest)

    def save(self):
        self._trim()
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {"consumer_id": self.consumer_id, "entries": self._entries}
        fd, tmp = tempfile.mkstemp(prefix=".seen-", dir=directory)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def __len__(self):
        return len(self._entries)

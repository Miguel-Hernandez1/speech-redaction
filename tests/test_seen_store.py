"""Seen-store tests: durability across runs (dedup) and bounded trimming.

    python3 -m pytest tests/test_seen_store.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seen_store import SeenStore  # noqa: E402


def test_seen_store_persists_across_runs(tmp_path):
    path = str(tmp_path / ".state" / "speech-redaction.json")

    s1 = SeenStore(path)
    assert not s1.contains("uid-a")
    s1.add("uid-a", 100)
    s1.add("uid-b", 200)
    s1.save()

    # A fresh instance (simulating a restart) loads what was saved.
    s2 = SeenStore(path)
    assert s2.contains("uid-a")
    assert s2.contains("uid-b")
    assert not s2.contains("uid-c")
    assert len(s2) == 2


def test_seen_store_is_bounded_and_keeps_newest(tmp_path):
    path = str(tmp_path / "seen.json")
    s = SeenStore(path, max_size=3)
    for uid, ts in [("a", 10), ("b", 20), ("c", 30), ("d", 40), ("e", 50)]:
        s.add(uid, ts)
    s.save()  # _trim runs on save

    reloaded = SeenStore(path, max_size=3)
    assert len(reloaded) == 3
    # newest three by capture_ts kept, oldest two dropped
    assert reloaded.contains("c") and reloaded.contains("d") and reloaded.contains("e")
    assert not reloaded.contains("a") and not reloaded.contains("b")


def test_seen_store_missing_file_starts_empty(tmp_path):
    s = SeenStore(str(tmp_path / "does-not-exist.json"))
    assert len(s) == 0


def test_seen_store_corrupt_file_starts_empty(tmp_path):
    path = str(tmp_path / "seen.json")
    with open(path, "w") as fh:
        fh.write("not json at all")
    s = SeenStore(path)
    assert len(s) == 0  # fail-soft: empty rather than crash

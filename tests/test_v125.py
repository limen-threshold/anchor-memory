"""v1.25 — outside-review fixes. Pure unit tests: no model download, no chromadb."""
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anchor_guards as g  # noqa: E402


class ParseTs(unittest.TestCase):
    def test_naive_and_aware(self):
        self.assertEqual(g.parse_ts_naive("2026-11-14T10:00:00"), datetime(2026, 11, 14, 10))
        self.assertEqual(g.parse_ts_naive("2026-11-14T10:00:00Z"), datetime(2026, 11, 14, 10))
        self.assertEqual(g.parse_ts_naive("2026-11-14T18:00:00+08:00"), datetime(2026, 11, 14, 10))
        self.assertIsNone(g.parse_ts_naive("yesterday"))
        self.assertIsNone(g.parse_ts_naive(None))

    def test_subtraction_is_safe(self):
        dt = g.parse_ts_naive("2026-11-14T18:00:00+08:00")
        (datetime.utcnow() - dt).total_seconds()  # must not raise

    def test_local_roundtrip(self):
        utc = datetime(2026, 11, 14, 10, 0, 0)
        local = g.utc_to_local_naive(utc)
        off = datetime.now().astimezone().utcoffset()
        self.assertEqual(local - utc, off)
        self.assertEqual(g.local_to_utc_naive(local), utc)


class KeywordPool(unittest.TestCase):
    def test_full_match_survives_common_word_flood(self):
        from anchor_db import AnchorDB
        with tempfile.TemporaryDirectory() as d:
            db = AnchorDB(os.path.join(d, "m.db"))
            db.keyword_single_char_allow = set()
            for i in range(350):
                db.insert(f"m{i}", "the garden was wet again", tag="x")
            db.insert("target", "the garden hose by the shed", tag="x")
            pool = db.keyword_search("garden hose", limit=300)
            ids = [r["memory_id"] for r in pool]
            self.assertIn("target", ids)
            self.assertEqual(ids[0], "target")

    def test_order_by_shape(self):
        s = g.kw_order_by(["text LIKE ?", "text LIKE ?"])
        self.assertTrue(s.startswith(" ORDER BY ("))
        self.assertNotIn("timestamp", s)
        self.assertEqual(g.kw_order_by([]), "")


class StoreNormalisesTimestamp(unittest.TestCase):
    def test_offset_becomes_utc_naive(self):
        from anchor_db import AnchorDB
        with tempfile.TemporaryDirectory() as d:
            db = AnchorDB(os.path.join(d, "m.db"))
            db.insert("a", "hello", timestamp="2026-11-14T18:00:00+08:00")
            self.assertEqual(db.get("a")["timestamp"], "2026-11-14T10:00:00")
            db.insert("b", "hello2", timestamp="2026-11-14T10:00:00Z")
            self.assertEqual(db.get("b")["timestamp"], "2026-11-14T10:00:00")

    def test_wal_and_indexes(self):
        from anchor_db import AnchorDB
        with tempfile.TemporaryDirectory() as d:
            db = AnchorDB(os.path.join(d, "m.db"))
            with db._conn() as c:
                self.assertEqual(c.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
                idx = {r[1] for r in c.execute("PRAGMA index_list(memories)")}
            self.assertIn("idx_memories_timestamp", idx)
            self.assertIn("idx_memories_pinned", idx)


class LiteralConflict(unittest.TestCase):
    def test_update_vs_rephrase(self):
        self.assertTrue(g.literal_conflict("the cat is Scout", "the cat is Scott"))
        self.assertTrue(g.literal_conflict("meeting on 11-13", "meeting on 11-12"))
        self.assertFalse(g.literal_conflict("the cat Scout is three", "Scout the cat, three years old"))
        self.assertFalse(g.literal_conflict("今天很累", "今天好累啊"))


class OpenAIKwargs(unittest.TestCase):
    def test_reasoning_models(self):
        from anchor_llm import openai_token_kwargs
        k = openai_token_kwargs("gpt-5-nano", 500, 0.5)
        self.assertNotIn("temperature", k)
        self.assertNotIn("max_tokens", k)
        self.assertGreaterEqual(k["max_completion_tokens"], 500)
        k = openai_token_kwargs("o3-mini", 500, 0.5)
        self.assertIn("max_completion_tokens", k)

    def test_classic_models(self):
        from anchor_llm import openai_token_kwargs
        self.assertEqual(openai_token_kwargs("gpt-4o-mini", 500, 0.5), {"max_tokens": 500, "temperature": 0.5})


class TimelineAppend(unittest.TestCase):
    def test_append_keeps_lines_and_trims(self):
        import anchor_pinned as p
        with tempfile.TemporaryDirectory() as d:
            for i in range(p.TIMELINE_MAX_LINES + 5):
                p.append_timeline_event(d, f"event {i} 🙂")
            with open(os.path.join(d, p.RECENT_TIMELINE), encoding="utf-8") as f:
                lines = [l for l in f.read().split("\n") if l.strip()]
            self.assertEqual(len(lines), p.TIMELINE_MAX_LINES)
            self.assertTrue(lines[-1].endswith(f"event {p.TIMELINE_MAX_LINES + 4} 🙂"))
            self.assertFalse(os.path.exists(os.path.join(d, p.RECENT_TIMELINE + ".tmp")))


if __name__ == "__main__":
    unittest.main()

"""v1.26 — batching, FTS5 pre-filter, proxy robustness. No model download, no chromadb, no network."""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anchor_db import AnchorDB  # noqa: E402


class BatchReads(unittest.TestCase):
    def test_stats_cite_contexts(self):
        with tempfile.TemporaryDirectory() as d:
            db = AnchorDB(os.path.join(d, "m.db"))
            db.insert("a", "alpha", emotion_score=0.9, context="ctx-a")
            db.insert("b", "beta", emotion_score=0.2)
            db.cite("a")
            st = db.stats_for(["a", "b", "missing", "a"])
            self.assertEqual(st["a"], (1, 0.9))
            self.assertEqual(st["b"], (0, 0.2))
            self.assertNotIn("missing", st)
            db.cite_batch(["a", "b", "b"])
            self.assertEqual(db.get_citation_count("a"), 2)
            self.assertEqual(db.get_citation_count("b"), 1)
            self.assertEqual(db.contexts_for(["a", "b"]), {"a": "ctx-a"})
            self.assertEqual(db.stats_for([]), {})

    def test_context_column_checked_once(self):
        with tempfile.TemporaryDirectory() as d:
            db = AnchorDB(os.path.join(d, "m.db"))
            self.assertTrue(db._ctx_ok)


class FtsLane(unittest.TestCase):
    def _db(self, d, fts=True):
        if not fts:
            os.environ["ANCHOR_KEYWORD_FTS"] = "0"
        else:
            os.environ.pop("ANCHOR_KEYWORD_FTS", None)
        try:
            return AnchorDB(os.path.join(d, "m.db"))
        finally:
            os.environ.pop("ANCHOR_KEYWORD_FTS", None)

    def test_fts_maintained_and_backfilled(self):
        with tempfile.TemporaryDirectory() as d:
            db = self._db(d, fts=False)
            for i in range(5):
                db.insert(f"m{i}", f"garden row {i}")
            db2 = self._db(d, fts=True)          # existing store → backfill on open
            with db2._conn() as c:
                self.assertEqual(c.execute("SELECT count(*) FROM memories_fts").fetchone()[0], 5)
            db2.insert("new", "the garden hose")
            db2.insert("m0", "garden row zero rewritten")   # upsert replaces the fts row
            db2.delete("m1")
            with db2._conn() as c:
                self.assertEqual(c.execute("SELECT count(*) FROM memories_fts").fetchone()[0], 5)
                self.assertIn("rewritten", c.execute("SELECT toks FROM memories_fts WHERE memory_id='m0'").fetchone()[0])
                self.assertIsNone(c.execute("SELECT 1 FROM memories_fts WHERE memory_id='m1'").fetchone())

    def test_same_results_as_like_and_full_match_first(self):
        with tempfile.TemporaryDirectory() as d:
            db = self._db(d, fts=True)
            for i in range(350):
                db.insert(f"m{i}", "the garden was wet again", tag="x")
            db.insert("target", "the garden hose by the shed", tag="x")
            db.insert("cjk", "记忆质量问题要先修车道", tag="y")
            ids = [r["memory_id"] for r in db.keyword_search("garden hose", limit=300)]
            self.assertEqual(ids[0], "target")
            self.assertIn("cjk", [r["memory_id"] for r in db.keyword_search("记忆 质量")])
            # tag filter still applies inside the FTS scope
            self.assertEqual([r["memory_id"] for r in db.keyword_search("garden hose", limit=5, tag="y")], [])
            # off switch → same answers through the plain LIKE path
            db.use_fts = False
            ids2 = [r["memory_id"] for r in db.keyword_search("garden hose", limit=300)]
            self.assertEqual(ids2[0], "target")

    def test_fts_prefilter_is_faster_on_a_big_store(self):
        with tempfile.TemporaryDirectory() as d:
            db = self._db(d, fts=True)
            with db._conn() as c:
                c.executemany("INSERT INTO memories(memory_id, text, timestamp) VALUES (?,?,?)",
                              [(f"r{i}", f"filler sentence number {i} about nothing", "2026-01-01") for i in range(20000)])
                c.commit()
            db._ensure_fts()   # backfill for rows written behind the API (as a migration would)
            db.insert("t", "the one row about gardens")
            t0 = time.time(); db.keyword_search("gardens", limit=300); fast = time.time() - t0
            db.use_fts = False
            t0 = time.time(); db.keyword_search("gardens", limit=300); slow = time.time() - t0
            self.assertLess(fast, slow)


class ProxyHelpers(unittest.TestCase):
    def test_inject_keeps_image_parts(self):
        import anchor_proxy as ap
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": [{"type": "text", "text": "看这张"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
        out = ap.inject_into_last_user(msgs, "<context_now>x</context_now>")
        last = out[-1]["content"]
        self.assertIsInstance(last, list)
        self.assertEqual(last[0]["type"], "text")
        self.assertTrue(last[0]["text"].startswith("<context_now>x</context_now>\n\n看这张"))
        self.assertEqual(last[1]["type"], "image_url")
        self.assertEqual(out[0]["content"], "hi")           # earlier messages untouched
        self.assertEqual(ap.inject_into_last_user([{"role": "user", "content": "plain"}], "W")[0]["content"], "W\n\nplain")

    def test_parse_json_lenient_and_dead_letter(self):
        import anchor_proxy as ap
        self.assertEqual(ap.parse_json_lenient('[{"a":1}]'), [{"a": 1}])
        self.assertEqual(ap.parse_json_lenient('Sure! Here it is:\n```json\n[{"a":1}]\n```'), [{"a": 1}])
        self.assertEqual(ap.parse_json_lenient('prose before [{"a":1}] prose after'), [{"a": 1}])
        self.assertEqual(ap.parse_json_lenient('{"skip": false, "intents": ["x"]} trailing'), {"skip": False, "intents": ["x"]})
        self.assertIsNone(ap.parse_json_lenient("nothing here"))
        with tempfile.TemporaryDirectory() as d:
            p = ap.dead_letter(d, "curator", "raw text", "why")
            self.assertTrue(os.path.exists(p))
            self.assertIn("raw text", open(p, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()

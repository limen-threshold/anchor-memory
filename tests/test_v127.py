"""v1.27 — four mechanisms ported from the house gateway. No model, no chromadb, no network."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anchor_db import AnchorDB  # noqa: E402
import anchor_proxy as ap  # noqa: E402


class HostTask(unittest.TestCase):
    def test_owui_title_and_tags_requests_are_tasks(self):
        title = [{"role": "user", "content": "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history.\n### Chat History:\n..."}]
        tags = [{"role": "user", "content": "### Task:\nGenerate 1-3 broad tags categorizing the main themes of the chat history ... JSON format: { \"tags\": [...] }"}]
        sys_task = [{"role": "system", "content": "### Task:\nSuggest 3-5 relevant follow-up questions"}, {"role": "user", "content": "..."}]
        real = [{"role": "user", "content": "我今天带 Alice 去剪头发了"}]
        for m in (title, tags, sys_task):
            self.assertTrue(ap.is_host_task(m))
        self.assertFalse(ap.is_host_task(real))
        self.assertFalse(ap.is_host_task([]))

    def test_curator_prompt_has_unknown_people_rule(self):
        self.assertIn("UNKNOWN PEOPLE", ap.CURATOR_SYSTEM)
        self.assertIn("never invent the relationship", ap.CURATOR_SYSTEM)


class RootVariants(unittest.TestCase):
    def test_roots(self):
        self.assertEqual(AnchorDB._root_variants("姥姥家"), ["姥姥"])
        self.assertEqual(AnchorDB._root_variants("朋友们"), ["朋友"])
        self.assertEqual(AnchorDB._root_variants("姥姥家里"), ["姥姥"])
        self.assertEqual(AnchorDB._root_variants("家里"), [])        # root would be one char
        self.assertEqual(AnchorDB._root_variants("garden"), [])      # CJK only
        self.assertEqual(AnchorDB._root_variants("那边"), [])

    def test_keyword_lane_finds_root_rows_without_dilution(self):
        with tempfile.TemporaryDirectory() as d:
            db = AnchorDB(os.path.join(d, "m.db"))
            db.keyword_single_char_allow = set()
            db.insert("root", "姥姥掌勺做了一桌菜")
            db.insert("surf", "去姥姥家吃饭")
            db.insert("none", "今天天气很好")
            ids = [r["memory_id"] for r in db.keyword_search("姥姥家 吃饭")]
            self.assertIn("root", ids)
            self.assertIn("surf", ids)
            self.assertNotIn("none", ids)
            groups = db.keyword_groups("姥姥家 吃饭")
            self.assertIn(["姥姥家", "姥姥"], groups)
            # scoring in the lane counts groups, not variants: the surface row matches both groups
            text = "去姥姥家吃饭"
            matched = sum(1 for g in groups if any(v in text for v in g))
            self.assertEqual(matched, len(groups))


class SceneMerge(unittest.TestCase):
    def _mem(self, scene=None):
        from anchor_memory import AnchorMemory
        m = SimpleNamespace(near_dup_merge_sim=0.93, scene_merge_sim=scene, scene_merge_window_sec=7200)
        return lambda meta, ent, now=None: AnchorMemory._merge_threshold(m, meta, ent, now)

    def test_opt_in_and_window(self):
        now = datetime(2026, 11, 16, 12, 0, 0)
        recent = (now - timedelta(minutes=20)).isoformat()
        old = (now - timedelta(hours=5)).isoformat()
        off = self._mem(None)
        self.assertEqual(off({"entity": "s1", "timestamp": recent}, "s1", now), 0.93)
        on = self._mem(0.85)
        self.assertEqual(on({"entity": "s1", "timestamp": recent}, "s1", now), 0.85)
        self.assertEqual(on({"entity": "s2", "timestamp": recent}, "s1", now), 0.93)   # other scene
        self.assertEqual(on({"entity": "s1", "timestamp": old}, "s1", now), 0.93)      # outside window
        self.assertEqual(on({"entity": "s1", "timestamp": recent}, None, now), 0.93)   # new text has no entity
        self.assertEqual(on(None, "s1", now), 0.93)


if __name__ == "__main__":
    unittest.main()

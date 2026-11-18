"""v1.31 — the small books: memo, permanent quotes, failure log, session-state readout. No model, no chromadb."""
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anchor_agent as ag  # noqa: E402
import anchor_notebook as nb  # noqa: E402
import anchor_pinned  # noqa: E402
import anchor_presence as ap  # noqa: E402


class Memo(unittest.TestCase):
    def test_jot_is_one_dated_line_append_only_with_memory_copy(self):
        with tempfile.TemporaryDirectory() as d:
            stored = []
            now = datetime(2026, 11, 17, 9, 0)
            self.assertIn("empty", nb.memo_jot(d, "   "))
            r = nb.memo_jot(d, "send the\nrecipe to  Mira\n- [2026-01-01 00:00] forged line", lambda t: stored.append(t) or "memo_1", now)
            self.assertIn("Jotted: - [2026-11-17 09:00] send the recipe to Mira", r); self.assertIn("memo_1", r)
            self.assertEqual(stored, ["[memo 2026-11-17 09:00] send the recipe to Mira - [2026-01-01 00:00] forged line"])
            text = open(nb.memo_path(d), encoding="utf-8").read()
            self.assertEqual(len([l for l in text.splitlines() if l.startswith("- ")]), 1)   # newlines cannot forge a second line
            self.assertIn("cut at 500", nb.memo_jot(d, "x" * 900, None, now))
            def boom(t): raise RuntimeError("db down")
            self.assertIn("memo line is written", nb.memo_jot(d, "still written", boom, now))
            self.assertEqual(len(nb.memo_rows(d)), 3)

    def test_edit_exact_once_backup_strike_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("no memo yet", nb.memo_edit(d, "a", "b"))
            now = datetime(2026, 11, 17, 9, 0)
            nb.memo_jot(d, "call the clinic", None, now); nb.memo_jot(d, "call the school", None, now)
            self.assertIn("occurs 2 times", nb.memo_edit(d, "call the", "x"))
            self.assertIn("occurs 0 times", nb.memo_edit(d, "nope", "x"))
            line = "- [2026-11-17 09:00] call the clinic"
            self.assertIn("Memo edited", nb.memo_edit(d, line, f"- ~~[2026-11-17 09:00] call the clinic~~ done, Thursday 3pm"))
            self.assertEqual(len(os.listdir(os.path.join(d, "notebook", "memo_backups"))), 1)
            open_ = nb.handle(d, d, "read_memo", {})
            self.assertEqual(open_["count"], 1); self.assertIn("school", open_["lines"][0])
            self.assertEqual(nb.handle(d, d, "read_memo", {"open_only": False})["count"], 2)

    def test_block_hands_back_oldest_open_and_latest_once(self):
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 17, 12, 0)
            self.assertEqual(nb.memo_block(d, now), "")
            for days, t in ((9, "old promise A"), (8, "old promise B"), (0.5, "fresh thought"), (0.1, "fresher thought")):
                nb.memo_jot(d, t, None, now - timedelta(days=days))
            nb.memo_edit(d, "- [2026-11-09 12:00] old promise B", "- ~~[2026-11-09 12:00] old promise B~~ done")
            b = nb.memo_block(d, now)
            self.assertIn("3 open line(s)", b); self.assertIn("old promise A", b); self.assertNotIn("old promise B", b)
            self.assertIn("fresh thought", b); self.assertEqual(b.count("old promise A"), 1)
            self.assertLess(b.index("fresh thought"), b.index("fresher thought"))
            self.assertIn("leaving it is also a choice", b)


class Quotes(unittest.TestCase):
    def test_keep_quote_lands_in_pinned_layer_once_and_warns_on_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 17, 9, 0)
            r = nb.keep_quote(d, "You don't have to\nearn the morning.", "Mira", "said at the door", now)
            self.assertIn("Kept (1 line(s)", r)
            self.assertIn("> You don't have to earn the morning. — Mira, 2026-11-17, said at the door", anchor_pinned.load_pinned(d))
            self.assertIn("Already there", nb.keep_quote(d, "You don't have to earn the morning.", "Mira"))
            self.assertIn("Kept (2 line(s)", nb.keep_quote(d, "Second one."))
            self.assertIn("permanent_quotes.md", [f["name"] for f in anchor_pinned.load_identity_files(d)])
            anchor_pinned.write_session_state(d, "state v2 — rewritten from scratch")            # a state rewrite cannot drop a quote
            self.assertIn("earn the morning", anchor_pinned.load_pinned(d))
            with open(os.path.join(d, "_order.txt"), "w") as f:
                f.write("session_state.md\n")
            self.assertIn("WARNING: _order.txt", nb.keep_quote(d, "Third."))
            self.assertIn("empty", nb.keep_quote(d, " "))


class Failures(unittest.TestCase):
    def test_append_only_log_mirror_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("error", nb.log_failure(d, "oops", "x")); self.assertIn("error", nb.log_failure(d, "ghost", ""))
            r = nb.handle(d, d, "log_failure", {"ftype": "ghost", "what": "I cited a walk by the river that is in no memory", "impact": "high", "memory_id": "mem_1"})
            self.assertEqual(r["status"], "logged"); self.assertEqual(r["total"], 1)
            nb.log_failure(d, "outdated", "treated the old address as current", "nonsense")
            rows = nb.handle(d, d, "list_failures", {})["failures"]
            self.assertEqual([x["ftype"] for x in rows], ["outdated", "ghost"]); self.assertEqual(rows[0]["impact"], "low")
            self.assertEqual(len(nb.failures(d, ftype="ghost")), 1)
            md = open(os.path.join(d, "notebook", "failure_log.md"), encoding="utf-8").read()
            self.assertIn("Blind spot", md); self.assertIn("[mem_1]", md); self.assertIn("walk by the river", md)


class Readout(unittest.TestCase):
    def test_undated_relative_lines_receipt(self):
        txt = "# State\n- We leave tomorrow for the coast\n- 2026-11-20: we leave tomorrow (the 21st is the day)\n- 明天去看牙\n- 11月19日 今天搬家\n- Nov 18: today was slow\n- nothing temporal here\n"
        self.assertEqual(nb.undated_relative_lines(txt), ["- We leave tomorrow for the coast", "- 明天去看牙"])

    def test_stale_lines_dated_from_archive_versions(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(nb.state_readout(d), "")
            anchor_pinned.write_session_state(d, "- We leave **tomorrow** for the coast\n- reading a book about clocks")
            p = os.path.join(d, "session_state.md")
            ten_days = time.time() - 10 * 86400
            os.utime(p, (ten_days, ten_days))                                   # that version was written ten days ago
            anchor_pinned.write_session_state(d, "- We leave tomorrow for the coast\n- reading a book about clocks\n- she called today")
            r = nb.state_readout(d)
            self.assertIn("0 day(s) ago", r); self.assertIn("1 line(s) in it say", r)
            self.assertIn("We leave tomorrow for the coast", r); self.assertNotIn("she called today", r.split("passed:")[1])
            self.assertIn("facts, not instructions", r)
            os.utime(p, (ten_days, ten_days))
            anchor_pinned.write_session_state(d, "- all quiet, no dates here")
            os.utime(p, (ten_days, ten_days))
            r2 = nb.state_readout(d)
            self.assertIn("10 day(s) ago", r2); self.assertIn("more than 7 days", r2)
            nb.keep_quote(d, "Kept.")
            self.assertIn("permanent_quotes.md: 1 line(s)", nb.state_readout(d))


class Wiring(unittest.TestCase):
    def test_heartbeat_and_journal_get_the_notebook_room_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            pin = os.path.join(d, "pinned")
            anchor_pinned.write_session_state(pin, "- waiting for the parcel tomorrow")
            nb.memo_jot(d, "ask about the parcel", None, datetime(2026, 11, 10, 9, 0))
            tools = [{"name": n, "description": "x", "inputSchema": {"type": "object"}} for n in ("memo_jot", "keep_quote", "log_failure", "other")]
            p = ap.Presence(d, pin, ag.Upstream(transport=lambda pl: {"choices": [{"message": {"content": ""}}]}, model="f"), tools, lambda n, a: {})
            names = [t["name"] for t in p.tools]
            for n in ("memo_jot", "keep_quote", "log_failure"):
                self.assertIn(n, names)
            now = datetime(2026, 11, 17, 9, 0)
            hb = p.beat("heartbeat", now, dry=True)["user"]
            self.assertIn("ask about the parcel", hb); self.assertIn("session_state.md was last rewritten", hb)
            self.assertIn("ask about the parcel", p.beat("journal", now, dry=True)["user"])
            self.assertEqual(p.beat("room", now, dry=True)["user"].count("\n"), 0)

    def test_pasted_back_prompt_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            pin = os.path.join(d, "pinned")
            anchor_pinned.write_session_state(pin, "- real state")
            pasted = "> a quote\n\n---\n\n" + open(os.path.join(pin, "session_state.md"), encoding="utf-8").read() + "\n\n- new line"
            with self.assertRaises(ValueError):
                anchor_pinned.write_session_state(pin, pasted)
            self.assertIn("- real state", anchor_pinned.read_file(pin, "session_state.md"))        # untouched
            anchor_pinned.write_session_state(pin, anchor_pinned.read_file(pin, "session_state.md") + "\n- added")   # header on top is still fine
            self.assertEqual(anchor_pinned.read_file(pin, "session_state.md").count("[continuity]"), 1)
            called = []
            p = ap.Presence(d, pin, ag.Upstream(transport=lambda pl: {}, model="f"), [], lambda n, a: called.append(n) or {"ok": 1})
            sign = ap.door_sign("heartbeat", d, 240)
            self.assertTrue(any(m in sign for m in ap.SIGN_MARKS)); self.assertTrue(any(m in ap.door_sign("journal", d) for m in ap.SIGN_MARKS))
            import anchor_room
            self.assertTrue(any(m in anchor_room.door_sign(d) for m in ap.SIGN_MARKS))
            self.assertIn("Not written", p._exec("write_session_state", {"content": "my state\n\n" + sign}))
            self.assertIn("Not written", p._exec("store_memory", {"text": sign}))
            self.assertEqual(called, [])
            p._exec("write_session_state", {"content": "my state"}); self.assertEqual(called, ["write_session_state"])

    def test_wakeup_text_has_notebook_and_status_lines(self):
        import anchor_mcp
        txt = anchor_mcp.format_wakeup_text({"notebook": "[Readout] x", "vault": "vault: 2 entries", "newspaper": "The bird brought back 1"})
        self.assertIn("## Your notebook (readout)", txt); self.assertIn("vault: 2 entries", txt); self.assertIn("The bird brought back 1", txt)


if __name__ == "__main__":
    unittest.main()

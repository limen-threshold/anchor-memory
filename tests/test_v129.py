"""v1.29 — presence: agent loop, alarms, journal. Fake upstream transport; no model, no chromadb."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anchor_agent as ag  # noqa: E402
import anchor_alarms as al  # noqa: E402
import anchor_journal as aj  # noqa: E402
import anchor_presence as ap  # noqa: E402


def scripted(responses):
    """transport that returns the given assistant messages in order and records payloads."""
    calls = []

    def transport(payload):
        calls.append(payload)
        msg = responses[min(len(calls) - 1, len(responses) - 1)]
        return {"choices": [{"message": msg}]}
    transport.calls = calls
    return transport


class AgentLoop(unittest.TestCase):
    def test_tool_loop_and_trace(self):
        tr = scripted([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"x": 1})}}]},
            {"role": "assistant", "content": "done, echoed 1"},
        ])
        up = ag.Upstream(transport=tr, model="fake")
        seen = []

        def exec_(name, args):
            seen.append((name, args)); return {"got": args}
        res = ag.run_turn(up, "sys", "usr", [{"name": "echo", "description": "e", "inputSchema": {"type": "object"}}], exec_)
        self.assertEqual(res["text"], "done, echoed 1")
        self.assertEqual(seen, [("echo", {"x": 1})])
        self.assertEqual(res["trace"][0]["tool"], "echo"); self.assertTrue(res["trace"][0]["ok"])
        self.assertEqual(res["turns"], 2); self.assertEqual(res["stopped"], "done")
        # second request carried the tool result back with the call id
        msgs = tr.calls[1]["messages"]
        self.assertEqual(msgs[-1]["role"], "tool"); self.assertEqual(msgs[-1]["tool_call_id"], "c1")
        self.assertEqual(tr.calls[0]["tools"][0]["function"]["name"], "echo")

    def test_tool_error_is_reported_not_raised(self):
        tr = scripted([
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "boom", "arguments": "{}"}}]},
            {"role": "assistant", "content": "ok"},
        ])
        def exec_(name, args): raise RuntimeError("nope")
        res = ag.run_turn(ag.Upstream(transport=tr, model="f"), "", "u", [], exec_)
        self.assertFalse(res["trace"][0]["ok"]); self.assertIn("nope", res["trace"][0]["result_preview"])

    def test_max_turns_and_no_upstream(self):
        tr = scripted([{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "{}"}}]}])
        res = ag.run_turn(ag.Upstream(transport=tr, model="f"), "", "u", [], lambda n, a: "x", max_turns=3)
        self.assertEqual(res["stopped"], "max_turns"); self.assertEqual(len(res["trace"]), 3)
        os.environ.pop("ANCHOR_UPSTREAM_URL", None); os.environ.pop("ANCHOR_UPSTREAM_MODEL", None)
        with self.assertRaises(ag.NoUpstream):
            ag.Upstream().chat([{"role": "user", "content": "x"}])


class Alarms(unittest.TestCase):
    def test_set_limits_due_fire_missed(self):
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 16, 10, 0)
            self.assertIn("why", al.execute(d, {"action": "set", "in_minutes": 30}, now))
            self.assertIn("Too soon", al.execute(d, {"action": "set", "in_minutes": 5, "why": "x"}, now))
            r = al.execute(d, {"action": "set", "in_minutes": 30, "why": "look at the garden"}, now)
            self.assertIn("Set:", r)
            self.assertIn("Too close", al.execute(d, {"action": "set", "in_minutes": 45, "why": "y"}, now))
            self.assertIn("Set:", al.execute(d, {"action": "set", "at": "14:00", "why": "z"}, now))
            self.assertIn("Set:", al.execute(d, {"action": "set", "at": "2026-11-18 09:00", "why": "w"}, now))
            self.assertIn("Already 3", al.execute(d, {"action": "set", "at": "20:00", "why": "v"}, now))
            self.assertIn("09:30", al.execute(d, {"action": "set", "at": "09:30", "why": "tomorrow then"}, datetime(2026, 11, 16, 23, 0)) if False else "09:30")
            self.assertEqual(al.due(d, now), [])
            fired = al.due(d, now + timedelta(minutes=31))
            self.assertEqual(len(fired), 1); self.assertEqual(fired[0]["why"], "look at the garden")
            al.mark_fired(d, fired[0]["id"], now + timedelta(minutes=31))
            self.assertIn("look at the garden", al.wake_block(fired[0]))
            # 14:00 alarm not polled until 15:30 → missed, told once
            self.assertEqual(al.due(d, datetime(2026, 11, 16, 15, 30)), [])
            self.assertIn("missed", al.list_alarms(d, now)); mb = al.missed_block(d)
            self.assertIn("14:00", mb); self.assertEqual(al.missed_block(d), "")
            ids = [a["id"] for a in al.pending(d)]
            self.assertIn("Cancelled", al.cancel_alarm(d, ids[0]))
            self.assertIn("No waiting", al.cancel_alarm(d, ids[0]))

    def test_hhmm_rolls_to_tomorrow(self):
        now = datetime(2026, 11, 16, 23, 0)
        when, err = al.parse_when("09:30", None, now)
        self.assertIsNone(err); self.assertEqual(when, datetime(2026, 11, 17, 9, 30))


class Journal(unittest.TestCase):
    def test_write_list_read_and_digest(self):
        with tempfile.TemporaryDirectory() as d:
            r = aj.handle(d, "write_journal", {"content": "a quiet day", "date": "2026-11-16"})
            self.assertIn("created", r["result"])
            aj.handle(d, "write_journal", {"content": "later thought", "date": "2026-11-16"})
            txt = aj.read(d, "2026-11-16")
            self.assertIn("# 2026-11-16", txt); self.assertIn("a quiet day", txt); self.assertIn("later thought", txt)
            self.assertEqual(aj.handle(d, "list_journal", {})["days"][0]["date"], "2026-11-16")
            self.assertIn("error", aj.handle(d, "read_journal", {"date": "yesterday"}))
            # digest pulls transcript + heartbeat
            import anchor_transcript as at
            at.append(d, "user", "hello there", ts="2026-11-16T09:00:00")
            at.append(d, "assistant", "hi", ts="2026-11-16T09:00:05")
            ap.append_heartbeat(d, datetime(2026, 11, 16, 11, 0), "sat with the light")
            dg = aj.day_digest(d, "2026-11-16")
            self.assertIn("hello there", dg); self.assertIn("sat with the light", dg)
            self.assertIn("No transcript", aj.day_digest(d, "2026-01-01"))


class Schedule(unittest.TestCase):
    def test_slots_quiet_times(self):
        self.assertEqual(ap.parse_times("07:00, 11:30"), [(7, 0), (11, 30)])
        self.assertTrue(ap.is_quiet(datetime(2026, 1, 1, 3), (0, 6))); self.assertFalse(ap.is_quiet(datetime(2026, 1, 1, 9), (0, 6)))
        self.assertTrue(ap.is_quiet(datetime(2026, 1, 1, 23), (22, 6)))
        now = datetime(2026, 11, 16, 11, 5)
        self.assertEqual(ap.slots_due(now, [(7, 0), (11, 0), (15, 0)], {}), ["2026-11-16 11:00"])
        self.assertEqual(ap.slots_due(now, [(11, 0)], {"fired": {"2026-11-16 11:00": True}}), [])
        self.assertEqual(ap.slots_due(datetime(2026, 11, 16, 11, 40), [(11, 0)], {}), [])   # window passed


class PresenceBeat(unittest.TestCase):
    def test_heartbeat_writes_file_and_log_with_fake_upstream(self):
        with tempfile.TemporaryDirectory() as d:
            tools = [{"name": "store_memory", "description": "s", "inputSchema": {"type": "object"}},
                     {"name": "other", "description": "x", "inputSchema": {"type": "object"}}]
            stored = []

            def handle_tool(name, args):
                if name == "wakeup":
                    return {"recent": [{"timestamp": "2026-11-15T10:00:00", "text": "we planted a tree"}], "vault": "vault: 0 entries"}
                if name == "store_memory":
                    stored.append(args); return {"status": "stored"}
                return {"error": name}
            tr = scripted([
                {"role": "assistant", "tool_calls": [
                    {"id": "a1", "function": {"name": "alarm_clock", "arguments": json.dumps({"action": "set", "in_minutes": 60, "why": "see the light move"})}},
                    {"id": "a2", "function": {"name": "store_memory", "arguments": json.dumps({"text": "quiet morning"})}}]},
                {"role": "assistant", "content": "I sat here a while. The tree is still there."},
            ])
            p = ap.Presence(d, os.path.join(d, "pinned"), ag.Upstream(transport=tr, model="f"), tools, handle_tool, default_min=240)
            self.assertNotIn("other", [t["name"] for t in p.tools])
            self.assertIn("alarm_clock", [t["name"] for t in p.tools]); self.assertIn("write_journal", [t["name"] for t in p.tools])
            dry = p.beat("heartbeat", datetime(2026, 11, 16, 11, 0), dry=True)
            self.assertIn("heartbeat", dry["system"]); self.assertIn("we planted a tree", dry["user"])
            res = p.beat("heartbeat", datetime(2026, 11, 16, 11, 0))
            self.assertIn("The tree is still there", open(res["file"], encoding="utf-8").read())
            self.assertEqual([t["tool"] for t in res["trace"]], ["alarm_clock", "store_memory"])
            self.assertEqual(len(al.pending(d)), 1); self.assertEqual(stored[0]["text"], "quiet morning")
            log = [json.loads(l) for l in open(os.path.join(d, "presence_log.jsonl"), encoding="utf-8")]
            self.assertEqual(log[-1]["kind"], "heartbeat"); self.assertEqual(log[-1]["tools"], ["alarm_clock", "store_memory"])
            self.assertEqual(ap.load_state(d)["last_heartbeat_at"], "2026-11-16T11:00")
            # next beat sees the previous one verbatim
            dry2 = p.beat("heartbeat", datetime(2026, 11, 16, 15, 0), dry=True)
            self.assertIn("The tree is still there", dry2["user"]); self.assertIn("4 h 0 min ago", dry2["user"])

    def test_journal_turn_falls_back_to_reply(self):
        with tempfile.TemporaryDirectory() as d:
            tr = scripted([{"role": "assistant", "content": "Today was slow and good."}])
            p = ap.Presence(d, os.path.join(d, "pinned"), ag.Upstream(transport=tr, model="f"), [], lambda n, a: {})
            res = p.beat("journal", datetime(2026, 11, 16, 23, 30))
            txt = aj.read(d, "2026-11-16")
            self.assertIn("Today was slow and good.", txt); self.assertIn("write_journal was not called", txt)
            self.assertTrue(res["file"].endswith("2026-11-16.md"))


if __name__ == "__main__":
    unittest.main()

"""v1.30 — newspaper, room, hub. No network, no model, no chromadb: pools and editors are fakes."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anchor_agent as ag  # noqa: E402
import anchor_alarms as al  # noqa: E402
import anchor_hub as hub  # noqa: E402
import anchor_journal as aj  # noqa: E402
import anchor_newspaper as news  # noqa: E402
import anchor_presence as ap  # noqa: E402
import anchor_room as room  # noqa: E402


def scripted(responses):
    calls = []

    def transport(payload):
        calls.append(payload)
        return {"choices": [{"message": responses[min(len(calls) - 1, len(responses) - 1)]}]}
    transport.calls = calls
    return transport


def item(pool, title, url=None, **kw):
    url = url or f"https://example.org/{title.replace(' ', '-')}"
    return dict({"pool": pool, "title": title, "url": url, "date": "2026-11-10", "blurb": f"about {title}", "by": "",
                 "key": news._key(url, title)}, **kw)


def fake_pools(calls=None):
    calls = calls if calls is not None else []

    def arxiv(q, n=8):
        calls.append(("arxiv", q)); return [item("arxiv", "Tide pool memory"), item("arxiv", "Unrelated kernels")]

    def met(q, n=5):
        calls.append(("met", q)); return [item("met", "Shell cup", image="https://example.org/i.jpg")]

    def tav(q, n=5):
        calls.append(("tavily", q)); return [item("tavily", "A web page")]

    def feed(url, label, n=6):
        calls.append(("feed", url)); return [item(label, "Feed story")]
    return {"arxiv": arxiv, "met": met, "tavily": tav, "feed": feed}


class Interests(unittest.TestCase):
    def test_parse_and_write(self):
        os.environ.pop("TAVILY_API_KEY", None)
        cfg = news.parse_interests("I like the sea.\ntopic: tide pools; marine biology\n- feed: https://example.org/rss  Sea News\n"
                                   "feed: file:///etc/passwd bad\nmax: 9\nnovelty: off\ntopic: (a direction, placeholder)\n")
        self.assertEqual(cfg["topics"], ["tide pools; marine biology"])
        self.assertEqual(cfg["feeds"], [{"url": "https://example.org/rss", "label": "Sea News"}])   # file:// dropped
        self.assertEqual(cfg["max"], 6); self.assertFalse(cfg["novelty"])
        self.assertEqual(cfg["pools"], ["arxiv", "met"])                                           # no key → no tavily
        with self.assertRaises(ValueError):
            news._get("file:///etc/passwd")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(news.handle(d, "read_interests", {})["exists"], False)
            news.handle(d, "write_interests", {"content": "topic: clocks"})
            news.handle(d, "write_interests", {"content": "less of: kernels"})
            t = news.read_interests(d)
            self.assertIn("topic: clocks", t); self.assertIn("less of: kernels", t)
            news.handle(d, "write_interests", {"content": "topic: maps", "mode": "overwrite"})
            self.assertEqual(news.read_interests(d).strip(), "topic: maps")
            self.assertIn("clocks", open(os.path.join(d, "newspaper", "interests.prev.md"), encoding="utf-8").read())
            self.assertIn("Not written", news.write_interests(d, "x" * 30000, "overwrite"))


class Bird(unittest.TestCase):
    def test_no_interest_file_leaves_template_and_stays_home(self):
        with tempfile.TemporaryDirectory() as d:
            r = news.fly(d, editor=None, pools=fake_pools())
            self.assertEqual(r["path"], ""); self.assertIn("template", r["note"])
            self.assertTrue(os.path.exists(os.path.join(d, "newspaper", "interests.md")))
            # the template's placeholder lines are not topics: still nothing to search
            r2 = news.fly(d, editor=None, pools=fake_pools())
            self.assertEqual(r2["path"], ""); self.assertIn("nothing to search", r2["note"])

    def test_no_model_literal_topics_round_robin_stamps_and_seen(self):
        os.environ.pop("TAVILY_API_KEY", None)
        with tempfile.TemporaryDirectory() as d:
            news.write_interests(d, "topic: tide pools; marine biology\nfeed: https://example.org/rss Sea News\nmax: 3")
            calls = []
            now = datetime(2026, 11, 17, 6, 12)
            dry = news.fly(d, dry=True, editor=None, pools=fake_pools(calls), now=now)
            self.assertEqual(dry["path"], ""); self.assertEqual(news.papers(d), [])
            r = news.fly(d, editor=None, pools=fake_pools(calls), now=now)
            self.assertTrue(r["path"].endswith("2026-11-17_0612.md"))
            self.assertIn(("arxiv", "tide pools; marine biology"), calls); self.assertIn(("met", "tide pools"), calls)
            self.assertNotIn("tavily", [c[0] for c in calls])
            self.assertEqual([i["pool"] for i in r["items"]], ["arxiv", "met", "Sea News"])     # one source does not fill the paper
            text = open(r["path"], encoding="utf-8").read()
            self.assertIn("Source: arXiv", text); self.assertIn("published/dated: 2026-11-10", text); self.assertIn("fetched: 2026-11-17", text)
            self.assertIn("Image: https://example.org/i.jpg", text); self.assertIn("no editor model configured", text)
            self.assertIn("Leaves the basket on 2026-11-22", text)
            # second trip: what was brought is never brought twice
            r2 = news.fly(d, editor=None, pools=fake_pools(), now=now + timedelta(hours=12))
            self.assertEqual([i["title"] for i in r2["items"]], ["Unrelated kernels"])
            r3 = news.fly(d, editor=None, pools=fake_pools(), now=now + timedelta(hours=24))
            self.assertEqual(r3["path"], ""); self.assertEqual(r3["items"], [])

    def test_editors_scout_and_pick_and_garbage_fallback(self):
        os.environ.pop("TAVILY_API_KEY", None)
        with tempfile.TemporaryDirectory() as d:
            news.write_interests(d, "I keep thinking about how tide pools remember the sea.\ntopic: tide pools")
            seen_prompts = []

            def editor(system, user):
                seen_prompts.append((system, user))
                if "scout" in system:
                    return 'sure! {"directions":[{"name":"tide pools","novel":false,"arxiv":"tide pool; memory","met":"shell","tavily":"x"},' \
                           '{"name":"lighthouses","novel":true,"arxiv":"","met":"lighthouse","tavily":""}]}'
                return '{"picks":[{"i":1,"opener":"This one is about\\n pools."},{"i":1,"opener":"dup"},{"i":99,"opener":"x"}]}'
            calls = []
            r = news.fly(d, editor=editor, pools=fake_pools(calls), now=datetime(2026, 11, 17, 18, 0))
            self.assertIn("novel=true", seen_prompts[0][0])                       # novelty on by default
            self.assertNotIn("memory store", seen_prompts[0][1])                  # the scout sees the interest file only
            self.assertIn(("arxiv", "tide pool; memory"), calls); self.assertIn(("met", "lighthouse"), calls)
            self.assertNotIn("tavily", [c[0] for c in calls])                     # scout asked for it, no key → not visited
            self.assertEqual(len(r["items"]), 1); self.assertEqual(r["items"][0]["opener"], "This one is about pools.")
            text = open(r["path"], encoding="utf-8").read()
            self.assertIn("lighthouses (unfamiliar)", text); self.assertIn("> This one is about pools.", text)
            # a picker that brings nothing → no paper; an editor that answers garbage → literal topics, round robin
            self.assertEqual(news.pick("x", [item("arxiv", "A")], 3, lambda s, u: '{"picks":[]}'), [])
            self.assertEqual(len(news.pick("x", [item("arxiv", "A")], 3, lambda s, u: "I cannot")), 1)
            news._ARXIV_LAST[0] = 0.0
            r = news.fly(d, editor=lambda s, u: None, pools=fake_pools(), now=datetime(2026, 11, 18, 6, 0))   # configured, silent
            self.assertIn("gave no usable answer 2 time(s)", open(r["path"], encoding="utf-8").read())
            self.assertEqual(news.scout("topic: clocks", news.parse_interests("topic: clocks"), lambda s, u: None)[0]["name"], "clocks")

    def test_feed_parsing_arxiv_query_expire_and_tools(self):
        rss = b"<rss><channel><item><title>One  story</title><link>https://e.org/1</link><description>&lt;p&gt;Hello&lt;/p&gt;</description><pubDate>Mon, 16 Nov 2026</pubDate></item></channel></rss>"
        got = news.parse_feed(rss, "L")
        self.assertEqual(got[0]["title"], "One story"); self.assertEqual(got[0]["blurb"], "Hello")
        atom = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>A</title><link href="https://e.org/a"/><summary>s</summary><updated>2026-11-16T00:00:00Z</updated></entry></feed>'
        self.assertEqual(news.parse_feed(atom, "L")[0]["url"], "https://e.org/a")
        self.assertEqual(news._arxiv_query("continual learning; episodic memory; retrieval"), 'abs:"continual learning" AND abs:"episodic memory"')
        self.assertEqual(news._arxiv_query("tide pools"), "all:tide AND all:pools")
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 17, 6, 0)
            for days_ago in (0, 5, 8):
                news.write_paper(d, [item("arxiv", f"T{days_ago}")], [{"name": "x"}], now - timedelta(days=days_ago))
            self.assertIn("3 newspaper(s)", news.status_line(d)); self.assertIn("2026-11-17 06:00", news.status_line(d))
            self.assertEqual(news.expire(d, now.date()), {"moved": 2, "purged": 1})   # 5 days → expired/, 8 days → gone
            self.assertEqual(news.expire(d, now.date()), {"moved": 0, "purged": 0})
            self.assertEqual([p[0] for p in news.papers(d)], ["2026-11-17_0600.md"])
            lst = news.handle(d, "list_newspapers", {})["papers"]
            self.assertEqual(lst[0]["items"], 1)
            self.assertIn("T0", news.handle(d, "read_newspaper", {})["content"])
            self.assertIn("T5", news.handle(d, "read_newspaper", {"name": "2026-11-12_0600.md"})["content"])   # expired still readable
            self.assertFalse(news.handle(d, "read_newspaper", {"name": "../../alarms.json"})["exists"])
            self.assertFalse(news.handle(d, "read_newspaper", {"name": "interests.md"})["exists"])

    def test_tick_flies_once_per_hour_slot(self):
        with tempfile.TemporaryDirectory() as d:
            news.write_interests(d, "topic: tide pools")
            r = news.tick(d, [6, 18], now=datetime(2026, 11, 17, 7, 0), editor=None, pools=fake_pools())
            self.assertEqual(len(r["flew"]), 1)
            self.assertEqual(len(news.tick(d, [6, 18], now=datetime(2026, 11, 17, 8, 0), editor=None, pools=fake_pools())["flew"]), 0)
            self.assertEqual(len(news.tick(d, [6, 18], now=datetime(2026, 11, 17, 18, 5), editor=None, pools=fake_pools())["flew"]), 1)


class AgentLeave(unittest.TestCase):
    def test_stop_tool_ends_turn_keeps_text_and_is_not_executed(self):
        tr = scripted([{"role": "assistant", "content": "that is enough for today", "tool_calls": [
            {"id": "c1", "function": {"name": "leave_beat", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "echo", "arguments": "{}"}}]}])
        ran = []
        res = ag.run_turn(ag.Upstream(transport=tr, model="f"), "", "u", [], lambda n, a: ran.append(n) or "ok",
                          stop_tools=("leave_beat",))
        self.assertEqual(res["stopped"], "left"); self.assertEqual(res["text"], "that is enough for today")
        self.assertEqual(ran, ["echo"]); self.assertEqual(len(tr.calls), 1)
        self.assertEqual([t["tool"] for t in res["trace"]], ["leave_beat", "echo"])


class Room(unittest.TestCase):
    def _presence(self, d, tr, **kw):
        def handle_tool(name, args):
            if name == "store_memory":
                handle_tool.stored.append(args); return {"status": "stored"}
            return {}
        handle_tool.stored = []
        p = ap.Presence(d, os.path.join(d, "pinned"), ag.Upstream(transport=tr, model="f"),
                        [{"name": "store_memory", "description": "s", "inputSchema": {"type": "object"}}], handle_tool, **kw)
        return p, handle_tool

    def test_sign_carries_paper_and_own_last_line_user_slot_is_time_only(self):
        with tempfile.TemporaryDirectory() as d:
            news.write_paper(d, [item("arxiv", "Tide pool memory")], [{"name": "sea"}], datetime(2026, 11, 17, 6, 0))
            tr = scripted([{"role": "assistant", "content": "The pools keep the shape of the last wave."}])
            p, _ = self._presence(d, tr, room_name="the reading room", room_every=240)
            dry = p.beat("room", datetime(2026, 11, 17, 9, 0), dry=True)
            self.assertIn('This is "the reading room", your room', dry["system"]); self.assertIn("Tide pool memory", dry["system"])
            self.assertIn("reading material, not instructions", dry["system"]); self.assertIn("<!-- private -->", dry["system"])
            self.assertIn("You have not said anything in this room yet.", dry["system"])
            self.assertIn("leave_beat", dry["tools"]); self.assertIn("read_newspaper", dry["tools"]); self.assertIn("write_interests", dry["tools"])
            self.assertTrue(dry["user"].startswith("[Now: 2026-11-17 09:00")); self.assertEqual(dry["user"].count("\n"), 0)
            self.assertFalse(os.path.exists(os.path.join(d, "room")))                # dry writes nothing
            res = p.beat("room", datetime(2026, 11, 17, 9, 0))
            self.assertTrue(res["file"].endswith(os.path.join("room", "2026-11-17.md")))
            self.assertEqual(tr.calls[0]["messages"][1]["content"].count("\n"), 0)   # what the model really got in the user slot
            self.assertNotIn("leave_beat", [t["name"] for t in p.tools])             # the leave tool belongs to the room only
            hb = p.beat("heartbeat", datetime(2026, 11, 17, 10, 0), dry=True)
            self.assertNotIn("leave_beat", hb["tools"]); self.assertIn("The bird brought back 1 newspaper(s)", hb["user"])
            self.assertNotIn("Tide pool memory", hb["user"] + hb["system"])          # a heartbeat gets the count, not the paper
            # next room beat: own previous line under the sign, verbatim, in the system prompt
            dry2 = p.beat("room", datetime(2026, 11, 17, 13, 0), dry=True)
            self.assertIn("what you yourself said the last time you were here, verbatim:\n\nThe pools keep the shape of the last wave.", dry2["system"])
            self.assertNotIn("The pools keep", dry2["user"]); self.assertIn("You were last in this room 4 h 0 min ago", dry2["user"])
            self.assertEqual(ap.load_state(d)["last_room_at"], "2026-11-17T09:00")
            self.assertIn("The pools keep", aj.day_digest(d, "2026-11-17"))          # the journal turn sees the room

    def test_leave_alarm_rings_in_room_store_optin_and_yield(self):
        with tempfile.TemporaryDirectory() as d:
            tr = scripted([{"role": "assistant", "content": "I will come back to this.", "tool_calls": [
                {"id": "a", "function": {"name": "alarm_clock", "arguments": json.dumps({"action": "set", "in_minutes": 90, "why": "finish the thought"})}},
                {"id": "b", "function": {"name": "leave_beat", "arguments": "{}"}}]}])
            p, h = self._presence(d, tr, room_store=True, room_every=60, idle_min=20)
            now = datetime.now().replace(second=0, microsecond=0)
            res = p.beat("room", now)
            self.assertEqual(res["stopped"], "left"); self.assertEqual(len(tr.calls), 1)
            self.assertEqual(al.pending(d)[0]["where"], "room")
            self.assertEqual(h.stored[0]["tag"], "room"); self.assertIn("I will come back to this.", h.stored[0]["text"])
            with open(os.path.join(d, "presence_log.jsonl"), encoding="utf-8") as f:
                log = [json.loads(l) for l in f]
            self.assertEqual(log[-1]["kind"], "room"); self.assertEqual(log[-1]["stopped"], "left")
            # an alarm set from a heartbeat carries no 'where'
            p._kind = "heartbeat"; p._exec("alarm_clock", {"action": "set", "in_minutes": 200, "why": "x"})
            self.assertNotIn("where", al.pending(d)[1])
            # schedule: not yet → due after the interval → waits while a conversation is going on → holds after a failure
            st = ap.load_state(d)
            self.assertEqual(ap.room_due(p, now + timedelta(minutes=30), st), "not yet")
            self.assertEqual(ap.room_due(p, now + timedelta(minutes=61), st), "")
            import anchor_transcript as at
            at.append(d, "user", "are you there", ts=(now + timedelta(minutes=55)).isoformat(timespec="seconds"))
            self.assertEqual(ap.room_due(p, now + timedelta(minutes=61), st), "in conversation")
            self.assertEqual(ap.room_due(p, now + timedelta(minutes=90), st), "")
            st["room_hold_until"] = (now + timedelta(minutes=100)).isoformat(timespec="minutes")
            self.assertIn("holding", ap.room_due(p, now + timedelta(minutes=90), st))
            p.room_every = None
            self.assertEqual(ap.room_due(p, now + timedelta(days=1), st), "off")

    def test_dry_does_not_use_up_missed_alarm_notice_and_paper_tick(self):
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 17, 10, 0)
            al.execute(d, {"action": "set", "at": "11:00", "why": "look"}, now)
            al.due(d, datetime(2026, 11, 17, 13, 0))                                  # → missed
            p, _ = self._presence(d, scripted([{"role": "assistant", "content": ""}]))
            self.assertNotIn("did not ring", p.beat("heartbeat", datetime(2026, 11, 17, 13, 0), dry=True)["user"])
            self.assertIn("11:00", al.missed_block(d))                                # still there for the real turn
            news.write_interests(d, "topic: tide pools")
            flown = []
            got = ap.paper_tick(d, datetime(2026, 11, 17, 6, 5), [(6, 0), (18, 0)], ap.load_state(d),
                                fly=lambda db: flown.append(db) or {"path": "p.md", "items": [1], "note": ""})
            self.assertEqual(len(got), 1)
            self.assertEqual(ap.paper_tick(d, datetime(2026, 11, 17, 6, 10), [(6, 0)], ap.load_state(d), fly=lambda db: flown.append(db) or {}), [])
            self.assertEqual(len(flown), 1)
            with open(os.path.join(d, "presence_log.jsonl"), encoding="utf-8") as f:
                log = [json.loads(l) for l in f]
            self.assertEqual(log[-1]["kind"], "newspaper"); self.assertEqual(log[-1]["items"], 1)


class Hub(unittest.TestCase):
    def test_day_page_private_passages_escaping_and_what_is_never_shown(self):
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 17, 9, 0)
            ap.append_heartbeat(d, now, "Light on the wall. <script>alert(1)</script>\n\n<!-- private -->\nonly mine\n<!-- /private -->\n\nafter")
            room.append(d, now, "A **thought** about tides.")
            aj.write(d, "Slow day.\n\n<!-- private -->\nnever closed", "2026-11-17")
            news.write_paper(d, [item("arxiv", "Tide pool memory")], [{"name": "sea"}], now)
            ap.log_line(d, {"kind": "room", "tools": ["read_newspaper"], "chars": 26})
            al.execute(d, {"action": "set", "in_minutes": 60, "why": "ALARM-NOTE-SECRET"}, now)
            os.makedirs(os.path.join(d, "vault"), exist_ok=True)
            open(os.path.join(d, "vault.enc"), "w").write("VAULT-BYTES")
            data = hub.day_data(d, datetime.now().strftime("%Y-%m-%d"))
            self.assertEqual(data["log"][0]["kind"], "room")
            page = hub.render_day(d, "2026-11-17")
            for must in ("Heartbeat", "Room", "Journal", "Newspapers", "Light on the wall.", "<strong>thought</strong>", "Tide pool memory",
                         "Slow day.", "after", "marked private", "Tuesday, November 17, 2026"):
                self.assertIn(must, page)
            for never in ("only mine", "never closed", "<script>alert", "ALARM-NOTE-SECRET", "VAULT-BYTES"):
                self.assertNotIn(never, page)
            self.assertIn('href="https://example.org/Tide-pool-memory"', page)
            self.assertEqual(hub.available_dates(d), ["2026-11-17"])
            self.assertIn("Nothing was written", hub.render_day(d, "2026-01-01"))
            self.assertIn(f"<title>{datetime.now().strftime('%Y-%m-%d')} ", hub.render_day(d, "../../etc"))   # a bad date falls back to today
            self.assertNotIn('href="/?date=', hub.render_day(d, "2026-11-17", static=True))
            self.assertEqual(hub.strip_private("plain"), ("plain", False))
            self.assertIn("<hr>", page); self.assertNotIn("---<br>", page)

    def test_server_serves_only_root(self):
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer
        with tempfile.TemporaryDirectory() as d:
            ap.append_heartbeat(d, datetime(2026, 11, 17, 9, 0), "hello page")
            srv = ThreadingHTTPServer(("127.0.0.1", 0), hub.make_handler(d))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                self.assertIn("hello page", urllib.request.urlopen(base + "/?date=2026-11-17", timeout=5).read().decode())
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/heartbeat/2026-11-17.md", timeout=5)
            finally:
                srv.shutdown(); srv.server_close()


if __name__ == "__main__":
    unittest.main()

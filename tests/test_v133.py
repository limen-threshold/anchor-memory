"""v1.33 — Telegram: mouth, courier button, chat window. A fake Bot API; no network, no model."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_courier as co  # noqa: E402
import anchor_telegram as tg  # noqa: E402
from test_v132 import FakeIMAP, FakeSMTP, letter  # noqa: E402

ENV = {"ANCHOR_TELEGRAM_TOKEN": "t0k", "ANCHOR_TELEGRAM_CHAT_ID": "777",
       "ANCHOR_MAIL_ADDRESS": "me@example.org", "ANCHOR_MAIL_PASSWORD": "p"}


class FakeAPI:
    def __init__(self, ok=True, boom=False):
        self.calls, self.ok, self.boom = [], ok, boom

    def __call__(self, token, method, _timeout=35, **params):
        if self.boom:
            raise OSError("network down")
        self.calls.append((method, params))
        return {"ok": self.ok, "description": "chat not found", "result": []}

    def texts(self):
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


class Telegram(unittest.TestCase):
    def setUp(self):
        self._old = {k: os.environ.get(k) for k in list(ENV) + ["ANCHOR_TELEGRAM_CHAT_URL"]}
        os.environ.update(ENV); os.environ.pop("ANCHOR_TELEGRAM_CHAT_URL", None)
        tg.DEBOUNCE = 0

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_absent_without_token_and_honest_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ.pop("ANCHOR_TELEGRAM_CHAT_ID")
            self.assertFalse(tg.enabled()); self.assertFalse(tg.send(d, "hi")["delivered"])
            os.environ["ANCHOR_TELEGRAM_CHAT_ID"] = "777"
            api = FakeAPI()
            r = tg.send(d, "thinking of the tide today", call=api)
            self.assertTrue(r["delivered"]); self.assertEqual(api.calls[0][1]["chat_id"], "777")
            self.assertFalse(tg.send(d, "  ", call=api)["delivered"])
            refused = tg.send(d, "x", call=FakeAPI(ok=False)); self.assertFalse(refused["delivered"]); self.assertIn("chat not found", refused["error"])
            down = tg.send(d, "x", call=FakeAPI(boom=True)); self.assertFalse(down["delivered"]); self.assertIn("network down", down["error"])
            self.assertIn("NOT delivered", self._send_via(d, FakeAPI(boom=True)))
            # only what was really delivered is in the window
            win = tg.window_id(d)
            tg.record(d, "user", "hello?", "chat")
            self.assertEqual([m["role"] for m in tg.window_messages(d, win)], ["user"])
            long = "\n".join(["line"] * 2000)
            api2 = FakeAPI(); tg.send(d, long, via="machine", call=api2)
            self.assertGreater(len(api2.calls), 1); self.assertTrue(all(len(p["text"]) <= 3900 for _, p in api2.calls))

    def _send_via(self, d, api):
        r = tg.send(d, "x", call=api)
        return "Delivered" if r["delivered"] else "NOT delivered — " + r["error"]

    def test_fuse_stops_a_loop(self):
        with tempfile.TemporaryDirectory() as d:
            api = FakeAPI(); now = datetime(2026, 11, 17, 9, 0)
            for i in range(tg.FUSE_PER_HOUR):
                self.assertTrue(tg.send(d, f"m{i}", call=api, now=now + timedelta(seconds=i))["delivered"])
            r = tg.send(d, "one more", call=api, now=now + timedelta(minutes=5))
            self.assertFalse(r["delivered"]); self.assertIn("fuse", r["error"])
            self.assertTrue(tg.send(d, "later", call=api, now=now + timedelta(minutes=70))["delivered"])

    def test_draft_buttons_send_only_from_the_configured_chat_and_never_twice(self):
        with tempfile.TemporaryDirectory() as d:
            t0 = datetime(2026, 11, 17, 12, 0)
            co.know(d, "mira@example.org", "Mira")
            co.scan(d, imap_factory=lambda: FakeIMAP([letter("mira@example.org", "bread", "How was it?", t0 + timedelta(hours=1), "<m1@x>")]), now=t0)
            did = co.write_reply(d, 1, "It rose.")["draft_id"]
            api = FakeAPI()
            self.assertEqual(tg.announce_drafts(d, api), 1)
            m, p = api.calls[0]
            self.assertIn("It rose.", p["text"]); self.assertIn("Mira", p["text"]); self.assertTrue(p["text"].startswith("[machine]"))
            self.assertEqual(p["reply_markup"]["inline_keyboard"][0][0]["callback_data"], f"mail:send:{did}")
            self.assertEqual(tg.announce_drafts(d, api), 0)                                   # not again
            co.write_reply(d, 1, "It rose. Barely.")
            import time; time.sleep(1.1); co.write_reply(d, 1, "It rose. Barely. Proud anyway.")
            self.assertEqual(tg.announce_drafts(d, api), 1)                                   # rewritten → shown again
            smtp = FakeSMTP()
            sender = lambda db, i: co.send(db, i, smtp_factory=lambda: smtp)
            br = tg.Bridge(d, api, mail_send=sender)
            stranger = {"callback_query": {"id": "c0", "data": f"mail:send:{did}", "from": {"id": 999}, "message": {"chat": {"id": 999}, "message_id": 5}}}
            self.assertIn("ignored", br.on_update(stranger)); self.assertEqual(smtp.sent, [])
            press = {"callback_query": {"id": "c1", "data": f"mail:send:{did}", "from": {"id": 777}, "message": {"chat": {"id": 777}, "message_id": 5}}}
            self.assertEqual(br.on_update(press), "sent to mira@example.org"); self.assertEqual(len(smtp.sent), 1)
            self.assertIn("Proud anyway", smtp.sent[0].get_content())
            self.assertIn("not sent", br.on_update(press)); self.assertEqual(len(smtp.sent), 1)   # a second press sends nothing
            self.assertIn("answerCallbackQuery", [m for m, _ in api.calls])
            self.assertEqual(tg.window_messages(d, tg.window_id(d)), [])                      # none of this is in the chat window
            self.assertIn("unknown button", tg.on_callback(d, {"id": "c2", "data": "rm -rf", "from": {"id": 777}, "message": {"chat": {"id": 777}}}, api))

    def test_chat_window_goes_through_the_same_endpoint_and_knows_what_the_mouth_sent(self):
        with tempfile.TemporaryDirectory() as d:
            api = FakeAPI()
            br = tg.Bridge(d, api)
            msg = lambda text, chat=777: {"message": {"chat": {"id": chat}, "text": text}}
            self.assertEqual(br.on_update(msg("hello")), "chat off"); self.assertIn("chat window is off", api.texts()[-1])
            os.environ["ANCHOR_TELEGRAM_CHAT_URL"] = "http://127.0.0.1:8210/v1"
            posted = []

            def post(url, body, key):
                posted.append((url, body))
                return {"choices": [{"message": {"content": "<think>hmm</think>The tide is out."}}]}
            br = tg.Bridge(d, api, post=post)
            tg.send(d, "I found a clock in the paper today.", call=api)                      # the AI wrote first, from a heartbeat
            self.assertIn("ignored", br.on_update(msg("let me in", chat=1)))
            self.assertEqual(br.on_update(msg("which clock?")), "queued"); self.assertEqual(br.on_update(msg("and where?")), "queued")
            self.assertEqual(br.flush(), "answered")
            url, body = posted[0]
            self.assertEqual(url, "http://127.0.0.1:8210/v1/chat/completions"); self.assertFalse(body["stream"])
            self.assertEqual([m["role"] for m in body["messages"]], ["user"])                # two lines, one turn; never opens on an assistant turn
            first = body["messages"][0]["content"]
            self.assertTrue(first.endswith("which clock?\nand where?"))
            self.assertIn("[Before this, you wrote to them on your own,", first); self.assertIn("I found a clock in the paper today.", first)
            self.assertEqual(api.texts()[-1], "The tide is out.")                             # thinking stripped
            br.on_update(msg("go on")); br.flush()
            roles = [m["role"] for m in posted[1][1]["messages"]]
            self.assertEqual(roles, ["user", "assistant", "user"])
            # a message the AI sends on its own later is in the window, marked as such
            tg.send(d, "Still thinking about that clock.", call=api)
            br.on_update(msg("me too")); br.flush()
            self.assertIn("[sent by you earlier, on your own,", posted[2][1]["messages"][-2]["content"])
            # failure speaks in the machine's voice, not the AI's
            br2 = tg.Bridge(d, api, post=lambda u, b, k: (_ for _ in ()).throw(OSError("proxy down")))
            br2.on_update(msg("hi")); self.assertEqual(br2.flush(), "error")
            self.assertTrue(api.texts()[-1].startswith("[machine] this turn did not run")); self.assertEqual(br2.flush(), "")
            # /new starts fresh lines; nothing else changes
            self.assertEqual(br.on_update(msg("/new")), "new window")
            br.on_update(msg("fresh")); br.flush()
            self.assertEqual([m["content"] for m in posted[-1][1]["messages"]], ["fresh"])
            self.assertEqual(br.on_update({"message": {"chat": {"id": 777}, "photo": [{}]}}), "non-text")

    def test_window_goes_stale_after_silence(self):
        with tempfile.TemporaryDirectory() as d:
            t = datetime(2026, 11, 17, 9, 0)
            w1 = tg.window_id(d, t); self.assertEqual(tg.window_id(d, t + timedelta(hours=3)), w1)
            self.assertNotEqual(tg.window_id(d, t + timedelta(hours=3 + tg.NEW_WINDOW_HOURS + 1)), w1)


if __name__ == "__main__":
    unittest.main()

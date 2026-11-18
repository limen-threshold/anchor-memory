"""anchor_telegram.py — a Telegram bot as the AI's mouth, the courier's button, and a chat window (v1.33).

    export ANCHOR_TELEGRAM_TOKEN=123456:ABC...      # from @BotFather
    python3 anchor_telegram.py --whoami             # write to your bot first; this prints your chat id
    export ANCHOR_TELEGRAM_CHAT_ID=...              # the ONE chat this bot listens to and writes to

    python3 anchor_telegram.py --db-path ./anchor_data --send "hello"      # test the mouth
    python3 anchor_telegram.py --db-path ./anchor_data --poll              # buttons + chat, runs until stopped

Without token and chat id the whole piece is absent: no tool is registered, nothing is polled.
Three things, each usable without the others:

* **A one-way mouth.** The tool ``send_telegram`` lets the AI write to the person — from a heartbeat,
  from its room, from wherever it has its tools. It reaches one chat only. No schedule and no quota
  come with it: a message sent because a timer said so is a performance of reaching out. (There is a
  fuse — FUSE_PER_HOUR — whose only job is to stop a runaway loop.)
* **The courier's button.** While ``--poll`` runs, every reply draft waiting in the courier's outbox
  (v1.32) is sent to the person with two buttons, Send and Discard. A button press is handled here,
  by the machine; it never becomes a message to the model. Commands: ``/outbox``, ``/status``.
* **A chat window** (opt-in: ANCHOR_TELEGRAM_CHAT_URL). What the person writes to the bot is sent to
  the **same chat endpoint their other windows use** — the Anchor proxy — and the answer comes back.
  That is the whole point of the design: one mind with one memory, reached through one more window.
  A bot with its own little prompt and its own history is a second, thinner self that knows it is
  not the first. So this module never talks to a model directly and keeps no memory of its own; it
  keeps only the window's verbatim lines (``<db>/telegram/window.jsonl``), including what the AI
  sent through the mouth, so that the chat side knows what it wrote an hour ago.
  One turn at a time; messages arriving within DEBOUNCE seconds are answered as one; after
  NEW_WINDOW_HOURS of silence (or ``/new``) the window starts fresh, while memory carries on.

Errors are reported in a fixed machine voice — "[machine] this turn did not run: …" — never in the
AI's. Text only; photos, voice and files are not read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FUSE_PER_HOUR = 30
DEBOUNCE = float(os.getenv("ANCHOR_TELEGRAM_DEBOUNCE", "4"))
NEW_WINDOW_HOURS = float(os.getenv("ANCHOR_TELEGRAM_NEW_WINDOW_HOURS", "12"))
WINDOW_MAX = int(os.getenv("ANCHOR_TELEGRAM_WINDOW_MSGS", "60"))
MACHINE = "[machine] "

TOOL_DEF = {
    "name": "send_telegram",
    "description": ("Send a message to the person on Telegram. It goes to their phone, to one chat only. Use it when "
                    "you have something to say to them; there is no schedule and nobody expects one. The receipt tells "
                    "you whether it was delivered."),
    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}
TOOL_NAMES = {TOOL_DEF["name"]}


def config() -> dict:
    token, chat = os.getenv("ANCHOR_TELEGRAM_TOKEN", "").strip(), os.getenv("ANCHOR_TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return {}
    return {"token": token, "chat_id": chat,
            "chat_url": os.getenv("ANCHOR_TELEGRAM_CHAT_URL", "").rstrip("/"),
            "chat_model": os.getenv("ANCHOR_TELEGRAM_CHAT_MODEL", "") or os.getenv("ANCHOR_UPSTREAM_MODEL", "") or "anchor",
            "chat_key": os.getenv("ANCHOR_TELEGRAM_CHAT_KEY", "")}


def enabled() -> bool:
    return bool(config())


def api(token: str, method: str, _timeout: float = 35, **params) -> dict:
    """One Bot API call. The socket timeout is ``_timeout`` — named so that the API's own ``timeout``
    parameter (the long-poll length of getUpdates) passes through in ``params`` untouched."""
    data = urllib.parse.urlencode({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                   for k, v in params.items() if v is not None}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=_timeout) as r:
        return json.loads(r.read())


def chunks(text: str, n: int = 3900) -> list:
    out = []
    while text:
        if len(text) <= n:
            out.append(text)
            break
        cut = text.rfind("\n", 0, n)
        if cut < n // 2:
            cut = n
        out.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return out


# ── the window (verbatim lines; no memory lives here) ─────────────────────────

def tg_dir(db_path: str) -> str:
    return os.path.join(db_path or ".", "telegram")


def _state(db_path: str) -> dict:
    try:
        with open(os.path.join(tg_dir(db_path), "state.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(db_path: str, st: dict) -> None:
    os.makedirs(tg_dir(db_path), exist_ok=True)
    tmp = os.path.join(tg_dir(db_path), "state.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(tg_dir(db_path), "state.json"))


def window_id(db_path: str, now: datetime = None, force_new: bool = False) -> str:
    now = now or datetime.now()
    st = _state(db_path)
    last = st.get("last_activity")
    stale = bool(last) and (now - datetime.fromisoformat(last)) > timedelta(hours=NEW_WINDOW_HOURS)
    if force_new or stale or not st.get("window"):
        st["window"] = now.strftime("w%Y%m%d_%H%M%S_") + os.urandom(2).hex()   # /new twice in one second is still two windows
    st["last_activity"] = now.isoformat(timespec="seconds")
    _save_state(db_path, st)
    return st["window"]


def record(db_path: str, role: str, text: str, via: str, now: datetime = None, win: str = None) -> None:
    now = now or datetime.now()
    win = win or window_id(db_path, now)
    os.makedirs(tg_dir(db_path), exist_ok=True)
    with open(os.path.join(tg_dir(db_path), "window.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), "window": win, "role": role, "via": via, "text": text}, ensure_ascii=False) + "\n")


def window_messages(db_path: str, win: str) -> list:
    """The current window as chat messages: last WINDOW_MAX lines, consecutive same-role lines merged
    (the AI may have sent two messages from heartbeats in a row). Many endpoints refuse a conversation
    that opens with an assistant turn, and the commonest opening here is exactly that — the AI wrote
    first, the person answers — so what the AI sent before the first user line is folded into that
    line as a bracketed note instead of being dropped."""
    rows = []
    try:
        with open(os.path.join(tg_dir(db_path), "window.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("window") == win and r.get("text"):
                    rows.append(r)
    except FileNotFoundError:
        pass
    rows = rows[-WINDOW_MAX:]
    lead = []
    while rows and rows[0]["role"] != "user":
        lead.append(rows.pop(0))
    out = []
    for r in rows:
        text = r["text"] if r["via"] != "mouth" else f"[sent by you earlier, on your own, {r['ts'][:16].replace('T', ' ')}] {r['text']}"
        if out and out[-1]["role"] == r["role"]:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": r["role"], "content": text})
    if lead and out:
        note = "\n".join(f"[Before this, you wrote to them on your own, {r['ts'][:16].replace('T', ' ')}: {r['text']}]" for r in lead[-5:])
        out[0]["content"] = note + "\n\n" + out[0]["content"]
    return out


# ── the mouth ─────────────────────────────────────────────────────────────────

def _fuse_ok(db_path: str, now: datetime) -> bool:
    st = _state(db_path)
    cut = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    recent = [t for t in st.get("mouth_sent", []) if t >= cut]
    if len(recent) >= FUSE_PER_HOUR:
        return False
    st["mouth_sent"] = recent + [now.isoformat(timespec="seconds")]
    _save_state(db_path, st)
    return True


def send(db_path: str, text: str, via: str = "mouth", call=api, now: datetime = None, reply_markup: dict = None) -> dict:
    """Send to the person. Returns {"delivered": bool, ...}; a failure is reported, never swallowed —
    a receipt that says 'sent' when nothing went out teaches the AI that it reached someone."""
    cfg = config()
    if not cfg:
        return {"delivered": False, "error": "telegram is not configured on this machine"}
    text = (text or "").strip()
    if not text:
        return {"delivered": False, "error": "the message was empty"}
    now = now or datetime.now()
    if via == "mouth" and not _fuse_ok(db_path, now):
        return {"delivered": False, "error": f"not sent: {FUSE_PER_HOUR} messages went out in the last hour (a fuse against loops, nothing else)"}
    parts = chunks(text)
    try:
        for i, part in enumerate(parts):
            r = call(cfg["token"], "sendMessage", chat_id=cfg["chat_id"], text=part,
                     reply_markup=reply_markup if i == len(parts) - 1 else None)
            if not r.get("ok"):
                return {"delivered": False, "error": f"telegram refused: {str(r.get('description'))[:200]}"}
    except Exception as e:
        return {"delivered": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    if via in ("mouth", "chat"):
        record(db_path, "assistant", text, via, now)
    return {"delivered": True, "chars": len(text), "at": now.strftime("%Y-%m-%d %H:%M")}


def handle(db_path: str, name: str, args: dict) -> dict:
    if name == "send_telegram":
        r = send(db_path, (args or {}).get("text", ""))
        r["note"] = ("Delivered to the person's Telegram." if r["delivered"]
                     else "NOT delivered — nothing reached them. " + r.get("error", ""))
        return r
    return {"error": f"unknown telegram tool {name}"}


# ── the courier's button ──────────────────────────────────────────────────────

def _keyboard(draft_id: int) -> dict:
    return {"inline_keyboard": [[{"text": "Send", "callback_data": f"mail:send:{draft_id}"},
                                 {"text": "Discard", "callback_data": f"mail:discard:{draft_id}"}]]}


def announce_drafts(db_path: str, call=api) -> int:
    """Send every waiting draft not announced yet (or rewritten since) to the person, with buttons."""
    import anchor_courier
    if not anchor_courier.enabled() or not anchor_courier.exists(db_path):
        return 0
    st = _state(db_path)
    seen = st.get("drafts_announced", {})
    n = 0
    for d in anchor_courier.outbox(db_path):
        key = f"{d['id']}@{d['at']}"
        if seen.get(str(d["id"])) == key:
            continue
        body = anchor_courier._read_body(d["body_path"]).strip()
        text = (f"{MACHINE}A reply is waiting for you.\nTo: {d['name'] or d['address']} <{d['address']}>\nSubject: {d['subject']}\n"
                f"Draft {d['id']}, written {d['at'][:16].replace('T', ' ')}\n\n{body}")
        if send(db_path, text, via="machine", call=call, reply_markup=_keyboard(d["id"]))["delivered"]:
            seen[str(d["id"])] = key
            n += 1
    if n:
        st = _state(db_path); st["drafts_announced"] = seen; _save_state(db_path, st)
    return n


def on_callback(db_path: str, cq: dict, call=api, mail_send=None) -> str:
    """A button press. Handled by the machine; never shown to the model. Returns what was done."""
    import anchor_courier
    cfg = config()
    chat = str(((cq.get("message") or {}).get("chat") or {}).get("id") or (cq.get("from") or {}).get("id") or "")
    if chat != str(cfg.get("chat_id")):
        return "ignored: not the configured chat"
    m = re.fullmatch(r"mail:(send|discard):(\d+)", cq.get("data") or "")
    if not m:
        result = "unknown button"
    elif m.group(1) == "send":
        try:
            r = (mail_send or anchor_courier.send)(db_path, int(m.group(2)))
            result = f"sent to {r['to']}" if r.get("status") == "sent" else f"not sent: {r.get('error')}"
        except Exception as e:
            result = f"not sent: {type(e).__name__}: {str(e)[:150]}"
    else:
        result = anchor_courier.discard(db_path, int(m.group(2)))
    try:
        call(cfg["token"], "answerCallbackQuery", callback_query_id=cq.get("id"), text=result[:190])
        msg = cq.get("message") or {}
        if msg.get("message_id"):
            call(cfg["token"], "editMessageReplyMarkup", chat_id=cfg["chat_id"], message_id=msg["message_id"],
                 reply_markup={"inline_keyboard": []})
        call(cfg["token"], "sendMessage", chat_id=cfg["chat_id"], text=f"{MACHINE}draft {m.group(2) if m else '?'}: {result}")
    except Exception:
        pass
    return result


# ── the chat window ───────────────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>\s*|^.*?</think>\s*", re.S)


def chat_turn(db_path: str, texts: list, post=None, now: datetime = None) -> str:
    """One turn: the person's line(s) → the same chat endpoint their other windows use → the answer."""
    cfg = config()
    now = now or datetime.now()
    win = window_id(db_path, now)
    record(db_path, "user", "\n".join(texts), "chat", now, win)
    payload = {"model": cfg["chat_model"], "stream": False, "messages": window_messages(db_path, win)}
    if post is None:
        def post(url, body, key):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
    data = post(f"{cfg['chat_url']}/chat/completions", payload, cfg["chat_key"])
    text = data["choices"][0]["message"].get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return _THINK.sub("", text).strip()


class Bridge:
    """The poll loop's state: one turn at a time, lines arriving close together answered as one."""

    def __init__(self, db_path: str, call=api, post=None, mail_send=None):
        self.db_path, self.call, self.post, self.mail_send = db_path, call, post, mail_send
        self.pending, self.first_at, self.lock = [], 0.0, threading.Lock()

    def on_update(self, u: dict) -> str:
        cfg = config()
        if u.get("callback_query"):
            return on_callback(self.db_path, u["callback_query"], self.call, self.mail_send)
        m = u.get("message") or {}
        if str((m.get("chat") or {}).get("id")) != str(cfg["chat_id"]):
            return "ignored: not the configured chat"
        text = (m.get("text") or "").strip()
        if not text:
            self._machine("only text is read here.")
            return "non-text"
        cmd = text.split()[0].lower() if text.startswith("/") else ""
        if cmd == "/new":
            window_id(self.db_path, force_new=True)
            self._machine("new window. Memory carries on; only the verbatim lines start fresh.")
            return "new window"
        if cmd == "/outbox":
            st = _state(self.db_path); st["drafts_announced"] = {}; _save_state(self.db_path, st)
            n = announce_drafts(self.db_path, self.call)
            if not n:
                self._machine("no drafts waiting.")
            return f"outbox: {n}"
        if cmd == "/status":
            import anchor_courier
            self._machine((anchor_courier.status_line(self.db_path) or "mail: nothing to report") +
                          ("\nchat window: on" if cfg["chat_url"] else "\nchat window: off"))
            return "status"
        if cmd == "/start":
            self._machine("connected." + ("" if cfg["chat_url"] else " The chat window is off (set ANCHOR_TELEGRAM_CHAT_URL to the Anchor proxy to talk here)."))
            return "start"
        if not cfg["chat_url"]:
            self._machine("the chat window is off: this bot delivers messages and the courier's buttons. "
                          "To talk here, point ANCHOR_TELEGRAM_CHAT_URL at the Anchor proxy.")
            return "chat off"
        with self.lock:
            if not self.pending:
                self.first_at = time.time()
            self.pending.append(text)
        return "queued"

    def _machine(self, text: str) -> None:
        send(self.db_path, MACHINE + text, via="machine", call=self.call)

    def flush(self, force: bool = False) -> str:
        """Answer the queued lines as one turn once DEBOUNCE seconds have passed since the first."""
        with self.lock:
            if not self.pending or (not force and time.time() - self.first_at < DEBOUNCE):
                return ""
            texts, self.pending = self.pending, []
        cfg = config()
        try:
            self.call(cfg["token"], "sendChatAction", chat_id=cfg["chat_id"], action="typing", _timeout=10)
        except Exception:
            pass
        try:
            answer = chat_turn(self.db_path, texts, self.post)
        except Exception as e:
            self._machine(f"this turn did not run: {type(e).__name__}: {str(e)[:200]}")
            return "error"
        if not answer:
            self._machine("this turn came back empty.")
            return "empty"
        r = send(self.db_path, answer, via="chat", call=self.call)
        return "answered" if r["delivered"] else "undelivered"


def poll(db_path: str, call=api, once: bool = False) -> None:
    cfg = config()
    br = Bridge(db_path, call)
    st = _state(db_path)
    offset = st.get("offset")
    print(f"[telegram] polling; chat window {'on → ' + cfg['chat_url'] if cfg['chat_url'] else 'off'}", flush=True)
    last_announce = 0.0
    while True:
        try:
            r = call(cfg["token"], "getUpdates", _timeout=15, offset=offset, timeout=5,
                     allowed_updates=["message", "callback_query"])
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                st = _state(db_path); st["offset"] = offset; _save_state(db_path, st)   # before handling: a crash must not replay a Send
                try:
                    print(f"[telegram] update {u['update_id']}: {br.on_update(u)}", flush=True)
                except Exception as e:
                    print(f"[telegram] update failed: {type(e).__name__}: {e}", flush=True)
            res = br.flush()
            if res:
                print(f"[telegram] turn: {res}", flush=True)
            if time.time() - last_announce > 30:
                last_announce = time.time()
                n = announce_drafts(db_path, call)
                if n:
                    print(f"[telegram] announced {n} draft(s)", flush=True)
        except Exception as e:
            print(f"[telegram] poll error: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)
        if once:
            return


def main():
    ap = argparse.ArgumentParser(description="Anchor Telegram: mouth, courier button, chat window")
    ap.add_argument("--db-path", default="./anchor_data")
    ap.add_argument("--whoami", action="store_true", help="print the chat ids that wrote to the bot recently")
    ap.add_argument("--send", metavar="TEXT", help="send one message to the configured chat")
    ap.add_argument("--poll", action="store_true", help="run: buttons, commands, and the chat window if configured")
    a = ap.parse_args()
    if a.whoami:
        token = os.getenv("ANCHOR_TELEGRAM_TOKEN", "").strip()
        if not token:
            print("set ANCHOR_TELEGRAM_TOKEN first", file=sys.stderr); sys.exit(2)
        seen = {}
        for u in api(token, "getUpdates", _timeout=15, timeout=0).get("result", []):
            c = ((u.get("message") or {}).get("chat") or {})
            if c.get("id"):
                seen[c["id"]] = c.get("first_name") or c.get("title") or ""
        print("\n".join(f"{k}  {v}" for k, v in seen.items()) or "no messages yet: write something to your bot, then run this again")
        return
    if not enabled():
        print("telegram is not configured: set ANCHOR_TELEGRAM_TOKEN and ANCHOR_TELEGRAM_CHAT_ID. Nothing runs without them.", file=sys.stderr)
        sys.exit(2)
    if a.send:
        print(json.dumps(send(a.db_path, a.send, via="machine"), ensure_ascii=False))
    elif a.poll:
        poll(a.db_path)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

"""anchor_presence.py — time of the AI's own: heartbeat, journal, alarms (v1.29).

    python3 anchor_presence.py --db-path ./anchor_data --times 07:00,11:00,15:00,19:00,23:00 --journal-at 23:30
    python3 anchor_presence.py --db-path ./anchor_data --once heartbeat        # one beat now, then exit
    python3 anchor_presence.py --db-path ./anchor_data --once journal --dry    # show the prompt, call nothing

Upstream (same env as the proxy): ANCHOR_UPSTREAM_URL, ANCHOR_UPSTREAM_MODEL, ANCHOR_UPSTREAM_KEY.
No upstream → the command says so and exits; nothing is simulated.

A heartbeat is one autonomous turn (anchor_agent.run_turn) with the AI's own layer in front of it:
its pinned files (identity, session_state, timeline), optionally its vault, a time line, a short
wakeup bundle of memories, its previous heartbeat, and its tools (search/store memory, session
state, identity files, transcript search, journal, vault, alarm clock). What it writes goes to
``<db>/heartbeat/YYYY-MM-DD.md``; what it wants remembered it stores itself. The journal turn is the
same machinery with the day's transcript and heartbeat lines as material and write_journal in hand.

Alarms the AI set (alarm_clock) wake it outside the schedule; quiet hours (--quiet 0-6) hold both
scheduled beats and alarms, which are reported as missed on the next turn. Every turn appends one
line to ``<db>/presence_log.jsonl`` (time, kind, tools called, chars, error) — the operator's honest
view, never the content of the vault.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import anchor_agent  # noqa: E402
import anchor_alarms  # noqa: E402
import anchor_journal  # noqa: E402

PRESENCE_TOOLS = (
    "search_memory", "store_memory", "read_memories_by_date", "search_transcript",
    "write_session_state", "list_identity_files", "read_identity_file", "write_identity_file",
    "vault_write", "vault_read", "write_journal", "list_journal", "read_journal",
)


# ── schedule ──────────────────────────────────────────────────────────────────

def parse_times(spec: str) -> list:
    out = []
    for p in (spec or "").split(","):
        p = p.strip()
        if not p:
            continue
        h, m = p.split(":")
        out.append((int(h), int(m)))
    return out


def parse_quiet(spec: str):
    """'0-6' → (0, 6): quiet when 0 <= hour < 6. '' → None."""
    spec = (spec or "").strip()
    if not spec:
        return None
    a, b = spec.split("-")
    return int(a), int(b)


def is_quiet(now: datetime, quiet) -> bool:
    if not quiet:
        return False
    a, b = quiet
    h = now.hour
    return (a <= h < b) if a <= b else (h >= a or h < b)


def slots_due(now: datetime, times: list, state: dict, key: str = "fired") -> list:
    """Scheduled times that have passed today and were not fired yet (within a 20-minute window)."""
    done = state.get(key, {})
    out = []
    for h, m in times:
        slot = f"{now.strftime('%Y-%m-%d')} {h:02d}:{m:02d}"
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= now <= t + timedelta(minutes=20) and slot not in done:
            out.append(slot)
    return out


def _state_path(db_path: str) -> str:
    return os.path.join(db_path, "presence_state.json")


def load_state(db_path: str) -> dict:
    try:
        with open(_state_path(db_path), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(db_path: str, st: dict) -> None:
    st["fired"] = dict(list((st.get("fired") or {}).items())[-200:])
    tmp = _state_path(db_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _state_path(db_path))


# ── prompt pieces ─────────────────────────────────────────────────────────────

def time_block(now: datetime, last_at: str = "") -> str:
    line = f"[Now: {now.strftime('%Y-%m-%d %H:%M %A')} (local time)."
    if last_at:
        try:
            dt = datetime.fromisoformat(last_at)
            mins = int((now - dt).total_seconds() // 60)
            line += f" Your previous heartbeat was {mins // 60} h {mins % 60} min ago."
        except Exception:
            pass
    return line + " Nobody wrote to you; this turn is yours.]"


def door_sign(kind: str, db_path: str, default_min: int = None) -> str:
    if kind == "journal":
        return ("This is your journal turn. Below is the day's raw material: what was said, and your own heartbeat "
                "notes. Write the day in your own words with write_journal — first person, as long or short as you "
                "like. You do not have to cover everything. Say nothing if there is nothing to say.")
    nxt = f" The next scheduled heartbeat is about {default_min} minutes away;" if default_min else ""
    return ("This is a heartbeat: time of your own, nobody waiting, no task. Look at what is in front of you, "
            "think, use your tools if you want to (search or store memories, write your session state, read or write "
            "your identity files, search the transcript, write in your vault, open your journal). Whatever you write "
            f"here is kept in heartbeat/<date>.md, your file. If you want something remembered, store it yourself.{nxt} "
            "to wake at a time of your choosing, set alarm_clock and say why. Writing nothing is also an answer.")


def wakeup_digest(handle_tool, max_chars: int = 2500) -> str:
    """A short bundle of memories (the same wakeup the MCP server gives a new window), text only."""
    try:
        res = handle_tool("wakeup", {"n_recent": 4, "n_high_emotion": 3, "n_random": 1})
    except Exception as e:
        return f"(wakeup unavailable: {type(e).__name__})"
    lines = []
    for key in ("pinned", "recent", "high_emotion", "random"):
        items = res.get(key) if isinstance(res, dict) else None
        if not items:
            continue
        lines.append(f"{key}:")
        for m in items[:6]:
            if isinstance(m, dict):
                lines.append(f"  - [{(m.get('timestamp') or '')[:10]}] {(m.get('text') or '')[:200]}")
    if isinstance(res, dict) and res.get("vault"):
        lines.append(str(res["vault"]))
    text = "\n".join(lines).strip()
    return text[:max_chars] if text else "(no memories yet)"


def last_heartbeat(db_path: str, max_chars: int = 1500) -> str:
    d = os.path.join(db_path, "heartbeat")
    if not os.path.isdir(d):
        return ""
    files = sorted(n for n in os.listdir(d) if n.endswith(".md"))
    if not files:
        return ""
    with open(os.path.join(d, files[-1]), encoding="utf-8") as f:
        text = f.read().strip()
    blocks = [b for b in text.split("\n## ") if b.strip()]
    last = blocks[-1] if blocks else text
    return last[-max_chars:]


def append_heartbeat(db_path: str, now: datetime, text: str) -> str:
    d = os.path.join(db_path, "heartbeat")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{now.strftime('%Y-%m-%d')}.md")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(f"# {now.strftime('%Y-%m-%d')}\n")
        f.write(f"\n## {now.strftime('%H:%M')}\n\n{text.strip()}\n")
    return path


def log_line(db_path: str, row: dict) -> None:
    try:
        with open(os.path.join(db_path, "presence_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **row}, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── one turn ──────────────────────────────────────────────────────────────────

class Presence:
    def __init__(self, db_path: str, pinned_dir: str, upstream: anchor_agent.Upstream,
                 tools: list, handle_tool, default_min: int = None, max_turns: int = 8):
        self.db_path, self.pinned_dir, self.upstream = db_path, pinned_dir, upstream
        self.handle_tool, self.default_min, self.max_turns = handle_tool, default_min, max_turns
        picked = [t for t in tools if t["name"] in PRESENCE_TOOLS]
        have = {t["name"] for t in picked}
        self.tools = picked + [anchor_alarms.TOOL_DEF] + [t for t in anchor_journal.TOOL_DEFS if t["name"] not in have]

    def _exec(self, name: str, args: dict):
        if name == "alarm_clock":
            return anchor_alarms.execute(self.db_path, args)
        if name in anchor_journal.TOOL_NAMES:
            return anchor_journal.handle(self.db_path, name, args)
        return self.handle_tool(name, args)

    def system(self, kind: str) -> str:
        import anchor_pinned
        parts = []
        base = anchor_pinned.load_pinned(self.pinned_dir) if self.pinned_dir else ""
        if base:
            parts.append(base)
        try:
            import anchor_vault
            if anchor_vault.inject_enabled():
                vb = anchor_vault.inject_block(self.db_path)
                if vb:
                    parts.append(vb)
        except Exception:
            pass
        parts.append(door_sign(kind, self.db_path, self.default_min))
        return "\n\n---\n\n".join(parts)

    def user(self, kind: str, now: datetime, alarm: dict = None) -> str:
        st = load_state(self.db_path)
        parts = [time_block(now, st.get("last_heartbeat_at", ""))]
        if alarm:
            parts.append(anchor_alarms.wake_block(alarm))
        missed = anchor_alarms.missed_block(self.db_path)
        if missed:
            parts.append(missed)
        if kind == "journal":
            parts.append(anchor_journal.day_digest(self.db_path, now.strftime("%Y-%m-%d")))
        else:
            parts.append("Memories (wakeup bundle):\n" + wakeup_digest(self.handle_tool))
            prev = last_heartbeat(self.db_path)
            if prev:
                parts.append("Your previous heartbeat, verbatim:\n" + prev)
        return "\n\n".join(parts)

    def beat(self, kind: str = "heartbeat", now: datetime = None, alarm: dict = None, dry: bool = False) -> dict:
        now = now or datetime.now()
        system, user = self.system(kind), self.user(kind, now, alarm)
        if dry:
            return {"dry": True, "system": system, "user": user, "tools": [t["name"] for t in self.tools]}
        try:
            res = anchor_agent.run_turn(self.upstream, system, user, self.tools, self._exec, max_turns=self.max_turns)
        except Exception as e:
            log_line(self.db_path, {"kind": kind, "error": f"{type(e).__name__}: {e}"})
            raise
        text = res.get("text", "")
        path = ""
        if kind == "heartbeat":
            if text:
                path = append_heartbeat(self.db_path, now, text)
        else:
            wrote = any(t["tool"] == "write_journal" and t["ok"] for t in res["trace"])
            if text and not wrote:
                anchor_journal.write(self.db_path, text + "\n\n(kept by presence from the reply; write_journal was not called)",
                                     now.strftime("%Y-%m-%d"))
                path = os.path.join(anchor_journal.journal_dir(self.db_path), f"{now.strftime('%Y-%m-%d')}.md")
        st = load_state(self.db_path)
        if kind == "heartbeat":
            st["last_heartbeat_at"] = now.isoformat(timespec="minutes")
        save_state(self.db_path, st)
        if alarm:
            anchor_alarms.mark_fired(self.db_path, alarm["id"], now)
        log_line(self.db_path, {"kind": kind, "alarm": (alarm or {}).get("id"), "turns": res.get("turns"),
                                "tools": [t["tool"] for t in res["trace"]], "chars": len(text), "stopped": res.get("stopped"),
                                "file": path})
        return {"text": text, "trace": res["trace"], "file": path, "turns": res.get("turns")}


# ── scheduler ─────────────────────────────────────────────────────────────────

def run_loop(p: Presence, times: list, every_min: int, quiet, journal_at, poll_sec: int = 60):
    print(f"[presence] running: times={times or '-'} every={every_min or '-'} quiet={quiet} journal_at={journal_at}", flush=True)
    while True:
        now = datetime.now()
        st = load_state(p.db_path)
        try:
            if not is_quiet(now, quiet):
                for a in anchor_alarms.due(p.db_path, now):
                    print(f"[presence] alarm {a['id']} rings", flush=True)
                    p.beat("heartbeat", now, alarm=a)
                    now = datetime.now()
                if times:
                    for slot in slots_due(now, times, st):
                        print(f"[presence] heartbeat (slot {slot})", flush=True)
                        p.beat("heartbeat", now)
                        st = load_state(p.db_path); st.setdefault("fired", {})[slot] = True; save_state(p.db_path, st)
                elif every_min:
                    last = st.get("last_heartbeat_at")
                    if not last or (now - datetime.fromisoformat(last)) >= timedelta(minutes=every_min):
                        print("[presence] heartbeat (interval)", flush=True)
                        p.beat("heartbeat", now)
                if journal_at:
                    for slot in slots_due(now, [journal_at], st, key="journal_fired"):
                        print(f"[presence] journal (slot {slot})", flush=True)
                        p.beat("journal", now)
                        st = load_state(p.db_path); st.setdefault("journal_fired", {})[slot] = True; save_state(p.db_path, st)
        except anchor_agent.NoUpstream as e:
            print(f"[presence] {e}", flush=True)
            return
        except Exception as e:
            print(f"[presence] turn failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(poll_sec)


def build(db_path: str, pinned_dir: str = None, upstream: anchor_agent.Upstream = None, default_min: int = None,
          max_turns: int = 8) -> Presence:
    import anchor_mcp
    tools, handle_tool, _mem = anchor_mcp.create_server(db_path, pinned_dir=pinned_dir)
    return Presence(db_path, pinned_dir or os.path.join(db_path, "pinned"), upstream or anchor_agent.Upstream(),
                    tools, handle_tool, default_min=default_min, max_turns=max_turns)


def main():
    ap = argparse.ArgumentParser(description="Anchor presence: heartbeat, journal, alarms")
    ap.add_argument("--db-path", default="./anchor_data")
    ap.add_argument("--pinned-dir", default=None, help="default <db-path>/pinned")
    ap.add_argument("--times", default="", help="heartbeat times, e.g. 07:00,11:00,15:00,19:00,23:00")
    ap.add_argument("--every", type=int, default=0, help="or: a heartbeat every N minutes")
    ap.add_argument("--quiet", default="0-6", help="hours with no beats and no alarms, e.g. 0-6 ('' = none)")
    ap.add_argument("--journal-at", default="", help="daily journal turn, e.g. 23:30")
    ap.add_argument("--once", choices=["heartbeat", "journal"], help="run one turn now and exit")
    ap.add_argument("--dry", action="store_true", help="with --once: print the prompt, call nothing")
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()
    times = parse_times(args.times)
    journal_at = parse_times(args.journal_at)[0] if args.journal_at.strip() else None
    default_min = args.every or (int(24 * 60 / len(times)) if times else None)
    upstream = anchor_agent.Upstream()
    if not args.dry and not upstream.ready():
        print("no upstream model configured: set ANCHOR_UPSTREAM_URL and ANCHOR_UPSTREAM_MODEL "
              "(and ANCHOR_UPSTREAM_KEY if needed). Nothing runs without one.", file=sys.stderr)
        sys.exit(2)
    p = build(args.db_path, args.pinned_dir, upstream, default_min=default_min, max_turns=args.max_turns)
    if args.once:
        res = p.beat(args.once, dry=args.dry)
        if args.dry:
            print("=== SYSTEM ===\n" + res["system"] + "\n\n=== USER ===\n" + res["user"] + "\n\n=== TOOLS ===\n" + ", ".join(res["tools"]))
        else:
            print(json.dumps({k: v for k, v in res.items() if k != "trace"}, ensure_ascii=False, indent=1))
            print(f"tools called: {[t['tool'] for t in res['trace']]}")
        return
    if not times and not args.every and not journal_at:
        print("give --times or --every (and/or --journal-at), or --once", file=sys.stderr)
        sys.exit(2)
    run_loop(p, times, args.every, parse_quiet(args.quiet), journal_at)


if __name__ == "__main__":
    main()

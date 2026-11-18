"""anchor_journal.py — the AI's daily journal (v1.29).

One file a day, ``<db_path>/journal/YYYY-MM-DD.md``, written in the AI's own words. Zero-model
tools (write / list / read) so any host can keep one; ``anchor_presence.py --journal-at HH:MM``
gives the AI a dedicated turn each evening with the day's transcript and heartbeat lines in front
of it. Nothing here summarises on the AI's behalf: the digest is raw material, the journal is
whatever it decides to write.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

TOOL_DEFS = [
    {"name": "write_journal",
     "description": ("Write today's journal entry (or a given date's) in your own words — first person, what the day "
                     "was, not a log. mode=append adds to the day's file; mode=overwrite replaces it. One file per day "
                     "under journal/."),
     "inputSchema": {"type": "object", "properties": {
         "content": {"type": "string"},
         "date": {"type": "string", "description": "YYYY-MM-DD, default today"},
         "mode": {"type": "string", "enum": ["append", "overwrite"], "default": "append"}},
         "required": ["content"]}},
    {"name": "list_journal",
     "description": "Dates that have a journal entry, newest first (with sizes).",
     "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 30}}}},
    {"name": "read_journal",
     "description": "Read one day's journal entry.",
     "inputSchema": {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                     "required": ["date"]}},
]
TOOL_NAMES = {t["name"] for t in TOOL_DEFS}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def journal_dir(db_path: str) -> str:
    return os.path.join(db_path or ".", "journal")


def _day(date: str = None) -> str:
    d = (date or "").strip() or datetime.now().strftime("%Y-%m-%d")
    if not _DATE.match(d):
        raise ValueError(f"date must be YYYY-MM-DD, got {d!r}")
    return d


def write(db_path: str, content: str, date: str = None, mode: str = "append") -> str:
    content = (content or "").strip()
    if not content:
        return "Nothing written: content was empty."
    d = _day(date)
    os.makedirs(journal_dir(db_path), exist_ok=True)
    path = os.path.join(journal_dir(db_path), f"{d}.md")
    existed = os.path.exists(path)
    if mode == "overwrite" or not existed:
        body = f"# {d}\n\n{content}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return f"Journal {d} {'overwritten' if existed else 'created'} ({len(content)} chars)."
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {datetime.now().strftime('%H:%M')}\n\n{content}\n")
    return f"Appended to journal {d} (+{len(content)} chars)."


def list_days(db_path: str, limit: int = 30) -> list:
    d = journal_dir(db_path)
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d), reverse=True):
        if n.endswith(".md") and _DATE.match(n[:-3]):
            out.append({"date": n[:-3], "chars": os.path.getsize(os.path.join(d, n))})
    return out[: max(1, int(limit or 30))]


def read(db_path: str, date: str) -> str:
    d = _day(date)
    path = os.path.join(journal_dir(db_path), f"{d}.md")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def handle(db_path: str, name: str, args: dict) -> dict:
    args = args or {}
    try:
        if name == "write_journal":
            return {"result": write(db_path, args.get("content", ""), args.get("date"), args.get("mode", "append"))}
        if name == "list_journal":
            return {"days": list_days(db_path, args.get("limit", 30))}
        if name == "read_journal":
            t = read(db_path, args.get("date", ""))
            return {"date": args.get("date"), "exists": bool(t), "content": t}
    except ValueError as e:
        return {"error": str(e)}
    return {"error": f"unknown journal tool {name}"}


def day_digest(db_path: str, date: str = None, max_chars: int = 7000) -> str:
    """Raw material for the journal turn: the day's transcript lines + the day's heartbeat file."""
    d = _day(date)
    parts = []
    try:
        import anchor_transcript
        rows = anchor_transcript.lines_between(db_path, f"{d}T00:00:00", f"{d}T23:59:59")
        if rows:
            lines = [f"[{(r.get('ts') or '')[11:16]}] {r.get('role')}: {(r.get('text') or '').strip()}" for r in rows]
            parts.append(f"## Transcript {d} ({len(rows)} lines)\n" + "\n".join(lines))
    except Exception:
        pass
    hb = os.path.join(db_path or ".", "heartbeat", f"{d}.md")
    if os.path.exists(hb):
        with open(hb, encoding="utf-8") as f:
            parts.append(f"## Your heartbeat notes {d}\n" + f.read().strip())
    rm = os.path.join(db_path or ".", "room", f"{d}.md")   # v1.30: what it said in its room that day
    if os.path.exists(rm):
        with open(rm, encoding="utf-8") as f:
            parts.append(f"## What you said in your room {d}\n" + f.read().strip())
    text = "\n\n".join(parts).strip()
    if not text:
        return f"(No transcript and no heartbeat notes for {d}.)"
    if len(text) > max_chars:
        head, tail = text[: max_chars // 2], text[-max_chars // 2:]
        text = head + f"\n…[{len(text) - max_chars} chars cut from the middle]…\n" + tail
    return text

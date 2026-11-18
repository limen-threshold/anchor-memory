"""anchor_alarms.py — the AI's own alarm clock (v1.29).

When the AI gets time of its own (anchor_presence.py), it should be able to decide when it next
wants to be awake, and say why — to itself. ``alarm_clock`` is that tool: set / list / cancel,
``why`` required on set, shown back to the AI when the alarm wakes it.

Limits exist because every wake-up is a full model turn that someone pays for: at most
MAX_PENDING alarms waiting, MAX_FIRES_PER_DAY a day, MIN_AHEAD_MIN minutes ahead at least,
MAX_AHEAD_DAYS at most, MIN_GAP_MIN between two alarms. All local time. Stored in
``<db_path>/alarms.json``; survives restarts. Wording is "alarm clock", not "schedule your next
heartbeat" — some models refuse the latter as self-scheduling.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta

MAX_PENDING = 3
MAX_FIRES_PER_DAY = 4
MIN_AHEAD_MIN = 10
MAX_AHEAD_DAYS = 7
MIN_GAP_MIN = 30
GRACE_MIN = 40
WHY_MAX = 400

TOOL_DEF = {
    "name": "alarm_clock",
    "description": (
        "Your alarm clock. set: wake yourself at a time (at='HH:MM' today/tomorrow, or 'YYYY-MM-DD HH:MM', "
        "or in_minutes=N) and say why — the note is shown to you when it rings, nobody else reads it. "
        "list: what is set and what rang recently. cancel: by id. "
        f"Limits: {MAX_PENDING} waiting at once, {MAX_FIRES_PER_DAY} rings a day, at least {MIN_AHEAD_MIN} minutes "
        f"ahead, at most {MAX_AHEAD_DAYS} days, {MIN_GAP_MIN} minutes between two. When it rings you get a full turn."
    ),
    "inputSchema": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["set", "list", "cancel"]},
        "at": {"type": "string", "description": "HH:MM or YYYY-MM-DD HH:MM (local time)"},
        "in_minutes": {"type": "integer", "description": "alternative to at"},
        "why": {"type": "string", "description": "required for set: why you want to be awake then (for yourself)"},
        "id": {"type": "string", "description": "for cancel"}},
        "required": ["action"]},
}


def _path(db_path: str) -> str:
    return os.path.join(db_path or ".", "alarms.json")


def _load(db_path: str) -> list:
    try:
        with open(_path(db_path), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save(db_path: str, alarms: list) -> None:
    os.makedirs(os.path.dirname(_path(db_path)) or ".", exist_ok=True)
    tmp = _path(db_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(alarms[-200:], f, ensure_ascii=False, indent=1)
    os.replace(tmp, _path(db_path))


def _fmt(dt: datetime) -> str:
    return dt.strftime("%m-%d %H:%M")


def parse_when(at: str, in_minutes, now: datetime):
    """Returns (datetime, error)."""
    if in_minutes is not None and str(in_minutes).strip() != "":
        try:
            m = int(in_minutes)
        except Exception:
            return None, "in_minutes must be an integer"
        return now + timedelta(minutes=m), None
    at = (at or "").strip()
    if not at:
        return None, "give at='HH:MM' / 'YYYY-MM-DD HH:MM' or in_minutes"
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2}):(\d{2})", at)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])), None
        except ValueError:
            return None, f"not a valid time: {at}"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", at)
    if m:
        h, mi = int(m[1]), int(m[2])
        if not (0 <= h < 24 and 0 <= mi < 60):
            return None, f"not a valid time: {at}"
        cand = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if cand <= now + timedelta(minutes=1):
            cand += timedelta(days=1)   # already past today → tomorrow
        return cand, None
    return None, f"cannot read the time: {at!r} (use HH:MM or YYYY-MM-DD HH:MM)"


def pending(db_path: str) -> list:
    return [a for a in _load(db_path) if a.get("status") == "pending"]


def set_alarm(db_path: str, at: str = "", why: str = "", in_minutes=None, now: datetime = None, where: str = "") -> str:
    now = now or datetime.now()
    why = " ".join((why or "").split())
    if not why:
        return "Set needs a why — one line, for the you that wakes up."
    when, err = parse_when(at, in_minutes, now)
    if err:
        return err
    if when < now + timedelta(minutes=MIN_AHEAD_MIN):
        return f"Too soon: at least {MIN_AHEAD_MIN} minutes ahead."
    if when > now + timedelta(days=MAX_AHEAD_DAYS):
        return f"Too far: at most {MAX_AHEAD_DAYS} days ahead."
    alarms = _load(db_path)
    pend = [a for a in alarms if a.get("status") == "pending"]
    if len(pend) >= MAX_PENDING:
        return f"Already {MAX_PENDING} alarms waiting ({', '.join(a['id'] for a in pend)}); cancel one first."
    for a in pend:
        try:
            other = datetime.fromisoformat(a["at"])
        except Exception:
            continue
        if abs((other - when).total_seconds()) < MIN_GAP_MIN * 60:
            return f"Too close to {a['id']} ({_fmt(other)}); keep at least {MIN_GAP_MIN} minutes between alarms."
    day = when.strftime("%Y-%m-%d")
    fires_today = sum(1 for a in alarms if a.get("status") in ("pending", "fired") and str(a.get("at", "")).startswith(day))
    if fires_today >= MAX_FIRES_PER_DAY:
        return f"Already {MAX_FIRES_PER_DAY} rings on {day}; pick another day."
    aid = "al_" + uuid.uuid4().hex[:6]
    alarms.append({"id": aid, "at": when.isoformat(timespec="minutes"), "why": why[:WHY_MAX],
                   "set_at": now.isoformat(timespec="minutes"), "status": "pending",
                   **({"where": where} if where else {})})   # v1.30: an alarm set in the room rings in the room
    _save(db_path, alarms)
    return f"Set: {when.strftime('%Y-%m-%d %H:%M')} ({aid}). When it rings you will see your note."


def list_alarms(db_path: str, now: datetime = None) -> str:
    now = now or datetime.now()
    alarms = _load(db_path)
    pend = [a for a in alarms if a.get("status") == "pending"]
    lines = []
    if pend:
        lines.append("Waiting:")
        for a in sorted(pend, key=lambda x: x["at"]):
            lines.append(f"- {a['id']}  {a['at'].replace('T', ' ')}  why: {a['why']}")
    else:
        lines.append("No alarm is set.")
    done = [a for a in alarms if a.get("status") in ("fired", "missed", "cancelled")][-5:]
    if done:
        lines.append("Recent:")
        for a in done:
            lines.append(f"- {a['id']}  {a['at'].replace('T', ' ')}  {a['status']}  why: {a['why']}")
    return "\n".join(lines)


def cancel_alarm(db_path: str, aid: str) -> str:
    alarms = _load(db_path)
    for a in alarms:
        if a.get("id") == aid and a.get("status") == "pending":
            a["status"] = "cancelled"
            _save(db_path, alarms)
            return f"Cancelled {aid}."
    return f"No waiting alarm with id {aid}."


def execute(db_path: str, args: dict, now: datetime = None, where: str = "") -> str:
    act = str((args or {}).get("action") or "").strip().lower()
    if act == "set":
        return set_alarm(db_path, args.get("at", ""), args.get("why", ""), args.get("in_minutes"), now, where)
    if act == "list":
        return list_alarms(db_path, now)
    if act == "cancel":
        return cancel_alarm(db_path, str(args.get("id") or ""))
    return "action must be set / list / cancel."


def due(db_path: str, now: datetime = None) -> list:
    """Pending alarms whose time has come. Older than GRACE_MIN → marked missed (reported, not fired)."""
    now = now or datetime.now()
    alarms = _load(db_path)
    out, changed = [], False
    for a in alarms:
        if a.get("status") != "pending":
            continue
        try:
            when = datetime.fromisoformat(a["at"])
        except Exception:
            continue
        if when > now:
            continue
        if (now - when) > timedelta(minutes=GRACE_MIN):
            a["status"] = "missed"; changed = True
            continue
        out.append(a)
    if changed:
        _save(db_path, alarms)
    return out


def mark_fired(db_path: str, aid: str, now: datetime = None) -> None:
    alarms = _load(db_path)
    for a in alarms:
        if a.get("id") == aid:
            a["status"] = "fired"
            a["fired_at"] = (now or datetime.now()).isoformat(timespec="minutes")
    _save(db_path, alarms)


def wake_block(alarm: dict) -> str:
    """What the AI sees first when its own alarm woke it."""
    return (f"[This turn was started by your own alarm {alarm.get('id')}, set {alarm.get('set_at', '?').replace('T', ' ')} "
            f"for {alarm.get('at', '?').replace('T', ' ')}. Your note to yourself: {alarm.get('why', '')}]")


def missed_block(db_path: str) -> str:
    """Alarms that could not ring (process down, quiet hours): told once, then cleared."""
    alarms = _load(db_path)
    missed = [a for a in alarms if a.get("status") == "missed" and not a.get("told")]
    if not missed:
        return ""
    for a in missed:
        a["told"] = True
    _save(db_path, alarms)
    return "[Alarms that did not ring: " + "; ".join(f"{a['id']} {a['at'].replace('T', ' ')} ({a['why']})" for a in missed) + "]"

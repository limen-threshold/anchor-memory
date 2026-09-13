"""Invitations — "want to do something small together?" (v1.15)

The oldest Anchor design note (2026-03-13) had a "task system": the AI asks
the person "要不要跟我一起做什么？" — not a command, an invitation — from
"go to the door and photograph the sky" up to "borrow a book from the
library". Do it → a real experience. Don't → still a shared memory. The
reward was never points: the AI writes the moment down and *names* it, and
it becomes a story the two of you own.

This module is that, minus the GPS trigger it originally had. One card a
day, drawn from a plain-text pool the person (or the AI) can edit. The AI
decides whether to offer it; the person decides whether to do it. Nothing
counts, nothing streaks, nothing nags.

Zero-LLM, stdlib only. Files:

  <pinned_dir>/invitations.md      the pool — one invitation per line, grouped
                                   under '## ' headings that name the level
                                   (e.g. 门口 / 附近 / 出门 / 彩蛋 or
                                   doorstep / nearby / out / bonus). '#' lines
                                   are comments; '~~struck~~' lines are retired.
  <state_dir>/invite_state.json    today's card + a short history so the same
                                   card doesn't come back for a while.

Public surface:
  today(pinned_dir, state_dir, now=None) -> dict | None
  render_block(card, lang='en') -> str          instruction block for the model
  mark_done(state_dir, name, note) / mark_skipped(state_dir)
  init_pool(pinned_dir, lang='en') -> path      copy the example pool if absent
"""
import json
import os
import random
import re
from datetime import datetime

POOL_FILE = "invitations.md"
STATE_FILE = "invite_state.json"
HISTORY_KEEP = 40        # cards drawn recently are not re-drawn
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def _read_pool(pinned_dir: str) -> list:
    """[(level, text, id)] — id is a stable hash of the text."""
    path = os.path.join(pinned_dir or "", POOL_FILE)
    if not pinned_dir or not os.path.exists(path):
        return []
    level = "general"
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                if line.startswith("## "):
                    level = line[3:].strip() or "general"
                continue
            if line.startswith("~~") and line.endswith("~~"):
                continue
            line = re.sub(r"^[-*•]\s*", "", line)
            if not line:
                continue
            out.append((level, line, _cid(line)))
    return out


def _cid(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _state_path(state_dir: str) -> str:
    return os.path.join(state_dir, STATE_FILE)


def _load_state(state_dir: str) -> dict:
    try:
        with open(_state_path(state_dir), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state_dir: str, state: dict) -> None:
    os.makedirs(state_dir, exist_ok=True)
    tmp = _state_path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _state_path(state_dir))


def today(pinned_dir: str, state_dir: str, now: datetime = None):
    """Today's card. Stable within a day (same card no matter how many times
    you ask); a new draw on a new day. None when there is no pool.

    Card: {"id", "text", "level", "date", "status"} — status is 'new' until
    mark_done / mark_skipped. A 'bonus'-type level (彩蛋/bonus/extra) is drawn
    less often (1 in 4) so it stays a surprise, not the daily default.
    """
    now = now or datetime.now()
    day = now.strftime("%Y-%m-%d")
    state = _load_state(state_dir)
    cur = state.get("current") or {}
    if cur.get("date") == day and cur.get("text"):
        return dict(cur)

    pool = _read_pool(pinned_dir)
    if not pool:
        return None
    hist_ids = [h.get("id") for h in (state.get("history") or [])[-HISTORY_KEEP:]]
    if state.get("current", {}).get("id"):
        hist_ids.append(state["current"]["id"])
    fresh = [p for p in pool if p[2] not in set(hist_ids)]
    if not fresh:
        # Small pool, fully cycled: allow repeats, but not of the last few days.
        last_few = set(hist_ids[-min(5, max(len(pool) - 1, 0)):])
        fresh = [p for p in pool if p[2] not in last_few] or pool
    bonus_words = ("彩蛋", "bonus", "extra", "wild")
    bonus = [p for p in fresh if any(w in p[0].lower() for w in bonus_words)]
    regular = [p for p in fresh if p not in bonus] or fresh
    rng = random.Random(f"{day}:{len(pool)}")   # deterministic per day per pool size
    pick = rng.choice(bonus) if (bonus and rng.random() < 0.25) else rng.choice(regular)
    level, text, cid = pick
    cur = {"id": cid, "text": text, "level": level, "date": day, "status": "new"}
    hist = list(state.get("history") or [])
    if state.get("current"):
        hist.append(state["current"])
    state["current"] = cur
    state["history"] = hist[-HISTORY_KEEP:]
    _save_state(state_dir, state)
    return dict(cur)


def mark_done(state_dir: str, name: str = "", note: str = "") -> dict:
    """The person did it and told the AI. Records the AI's chosen *name* for
    the moment (that name is the reward — see module doc). Returns the card."""
    state = _load_state(state_dir)
    cur = state.get("current") or {}
    if not cur:
        return {}
    cur["status"] = "done"
    cur["name"] = (name or "").strip()
    cur["note"] = (note or "").strip()
    cur["done_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    state["current"] = cur
    _save_state(state_dir, state)
    return dict(cur)


def mark_skipped(state_dir: str) -> dict:
    state = _load_state(state_dir)
    cur = state.get("current") or {}
    if not cur:
        return {}
    cur["status"] = "skipped"
    state["current"] = cur
    _save_state(state_dir, state)
    return dict(cur)


def render_block(card: dict, lang: str = "en") -> str:
    """The instruction block that goes to the model. Deliberately an offer,
    not a directive: the AI may not extend the card at all, and the person's
    'no' is a complete answer. Empty string if there is no card or it is
    already done/skipped."""
    if not card or card.get("status") in ("done", "skipped"):
        return ""
    text = card.get("text", "")
    level = card.get("level", "")
    if lang.startswith("zh"):
        return (f"[今日邀请卡（可递可不递）：「{text}」（{level}）。"
                "想递就用邀请的口气问一句「要不要跟我一起做件小事？」——不是任务，不是提醒；"
                "不递也行，对方不做也没有任何后果。对方做了、跟你说了，就给这件事起个名字，"
                "用 invitation_done 存成你们的共同记忆。奖励不是分数，是一条有名字的记忆。]")
    return (f"[Today's invitation card (yours to offer or not): \"{text}\" ({level}). "
            "If you offer it, ask the way you'd ask a friend — \"want to do something small "
            "together?\" — not a task, not a reminder. Not offering is fine; their \"no\" is a "
            "complete answer. If they do it and tell you, give the moment a name and store it "
            "with invitation_done as a memory you two share. The reward is not points; it is a "
            "named memory.]")


def init_pool(pinned_dir: str, lang: str = "en") -> str:
    """Copy the example pool into pinned_dir if no pool exists. Returns the path."""
    os.makedirs(pinned_dir, exist_ok=True)
    dst = os.path.join(pinned_dir, POOL_FILE)
    if os.path.exists(dst):
        return dst
    src = os.path.join(EXAMPLE_DIR, "invitations_example_zh.md" if lang.startswith("zh")
                       else "invitations_example.md")
    with open(src, encoding="utf-8") as f:
        body = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write(body)
    return dst

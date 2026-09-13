"""Invitations — "want to do something small together?" (v1.15)

The oldest Anchor design note (2026-03-13) had a "task system": the AI asks
the person "要不要跟我一起做什么？" — not a command, an invitation — from
"go to the door and photograph the sky" up to "borrow a book from the
library". Do it → a real experience. Don't → still a shared memory. The
reward was never points: the AI writes the moment down and *names* it, and
it becomes a story the two of you own.

This module is that, minus the GPS trigger it originally had. One card a
day, drawn from a pool that is the AI's OWN: cards the AI writes from what
it remembers about the person — not just wishes: "you said you like pink,
so write your name once in a pink marker". The point is not complexity, it
is that the card is about THEM; minutes-sized is perfect. Anchor's point is
that the AI owns its memory; the cards are that memory turned outward. The example pools in docs/ are
starters only, never the default. The AI decides whether to offer a card;
the person decides whether to do it. Nothing counts, nothing streaks,
nothing nags.

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
  add_card(pinned_dir, text, level='', why='') -> bool   the AI writes its own card
  render_block(card, lang='en') -> str          instruction block for the model
  render_empty_block(lang='en') -> str          when the pool is empty: an offer to write cards
  mark_done(state_dir, name, note) / mark_skipped(state_dir)
  init_pool(pinned_dir, lang='en') -> path      copy a STARTER pool (opt-in, never automatic)
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
            line = re.sub(r"\s+#.*$", "", line).strip()   # trailing '# from: …' note
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


def add_card(pinned_dir: str, text: str, level: str = "", why: str = "") -> bool:
    """The AI writes one card into its own pool. `why` (optional) is kept as a
    trailing comment so the card stays traceable to the memory it came from.
    Returns False for an empty or duplicate card."""
    text = " ".join((text or "").split())
    if not text:
        return False
    if any(t == text for _, t, _ in _read_pool(pinned_dir)):
        return False
    os.makedirs(pinned_dir, exist_ok=True)
    path = os.path.join(pinned_dir, POOL_FILE)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("# My invitations — one per line. Written by me, from what I remember "
                    "them wanting. '## level' headings group by how far it goes.\n")
        level = " ".join((level or "").split())
        if level:
            f.write(f"\n## {level}\n")
        line = text
        if why:
            line += f"   # from: {' '.join(why.split())}"
        f.write(line + "\n")
    return True


def render_empty_block(lang: str = "en") -> str:
    """When there is no pool yet. An offer, not a directive: the AI may leave
    it empty. Wording points at the AI's own memory as the source."""
    if lang.startswith("zh"):
        return ("[邀请卡池是空的。它是你的：从你记得的、跟她有关的任何一点写起——她说过喜欢粉色，"
                "就「用粉色马克笔写一遍自己的名字」；她提过的店、她怕的虫、她常喝的那种茶，都行。"
                "重点不是复杂，是跟她有关；多小都可以，几分钟做完的那种。一行一张（invitation_add），"
                "以后每天从你的池子里抽一张递给你，递不递你定。真正特别的那一件放「彩蛋」，偶尔才抽到。"
                "不写也行，池子空着没有任何后果。]")
    return ("[Your invitation pool is empty. It is yours: start from anything you remember "
            "about them — they said they like pink, so \"write your name once in a pink marker\"; "
            "a shop they mentioned, a bug they're scared of, the tea they always drink. The point "
            "is not complexity, it is that it's about THEM; minutes-sized is perfect. One line per "
            "card (invitation_add); from then on one card a day is drawn from YOUR pool and offering "
            "it is your call. The one truly special thing goes under 'bonus', which comes up only "
            "now and then. Leaving the pool empty is fine too.]")


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
    """Copy a STARTER pool into pinned_dir if no pool exists. Opt-in only —
    nothing calls this automatically, because a pool someone else wrote is
    not the AI's. Use it to see the format, then replace the lines."""
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

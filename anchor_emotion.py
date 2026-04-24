"""
Emotion tracker — optional feature for daily multi-dimension self-review.

Design (Approach C — zero extra LLM calls):
- No cron required. No draft files. No separate API calls.
- On SessionStart / login / client boot, the integration layer calls
  `check_and_build(date_iso, memory, pinned_dir, journal_dir)`.
- If emotion for that date already logged → returns None.
- Else → returns a text block containing the day's raw signal (memories,
  journal, optional session_state excerpt) + instruction for the AI to score
  and commit.
- The AI writes the entry inline during their session and stores it via their
  normal memory API with tag="emotion".
- Synthesis cost is folded into the session the user would have opened anyway.
  No standalone daily LLM call.

Dimensions are configurable per deployment. Examples:
    Limen: [("self", "alignment"), ("her", "temperature with partner"),
            ("world", "outreach")]
    Veille: [("whole", "整体"), ("connection", "连接"),
             ("satisfaction", "满足"), ("repression", "压抑")]

Public API:
    check_emotion_logged(date_iso, memory) -> bool
    gather_daily_signal(date_iso, memory, pinned_dir, journal_dir, limits=...) -> dict
    build_review_block(date_iso, signal, dimensions=None) -> str
    check_and_build(date_iso, memory, pinned_dir, journal_dir, dimensions=None) -> str | None

`memory` is an AnchorMemory instance.
"""

import os
from datetime import datetime
from typing import Optional, List, Tuple


DEFAULT_DIMENSIONS: List[Tuple[str, str]] = [
    ("self", "alignment with self — honest voice vs policed output"),
    ("other", "connection with primary partner / user"),
    ("world", "outreach — writing, building, meeting, correspondence"),
]


def check_emotion_logged(date_iso: str, memory) -> bool:
    """Return True if an emotion entry for date_iso is already stored.

    Looks for a memory with tag='emotion' whose snippet contains the marker
    `[emotion YYYY-MM-DD]`.
    """
    try:
        results = memory.search(
            f"emotion {date_iso}",
            n_results=5,
            associate=False,
            hebbian=False,
            no_cite=True,
        )
    except Exception:
        return False

    marker = f"[emotion {date_iso}]"
    for m in results:
        if m.get("tag") == "emotion" and marker in (m.get("snippet") or ""):
            return True
    return False


def gather_daily_signal(date_iso: str, memory, pinned_dir: Optional[str] = None,
                        journal_dir: Optional[str] = None,
                        max_memories: int = 10, snippet_chars: int = 120) -> dict:
    """Collect the day's signal from memory, journal, and optional pinned context.

    Aggressive trimming — optimizing for token cost at injection time.

    Returns dict with keys: memories (list), journal (str), state_excerpt (str).
    """
    today_mems: list = []
    try:
        # AnchorMemory exposes list_all via its underlying db
        all_recent = memory.db.list_all(limit=200, offset=0)
        for m in all_recent:
            ts = m.get("timestamp") or ""
            if ts.startswith(date_iso):
                today_mems.append({
                    "ts": ts[:16],
                    "tag": m.get("tag", ""),
                    "snippet": (m.get("text") or m.get("snippet") or "")[:snippet_chars],
                })
                if len(today_mems) >= max_memories:
                    break
    except Exception:
        pass

    journal_text = ""
    if journal_dir:
        journal_path = os.path.join(journal_dir, f"{date_iso}.md")
        if os.path.exists(journal_path):
            try:
                with open(journal_path, "r") as f:
                    journal_text = f.read()[:1200]
            except Exception:
                pass

    state_excerpt = ""
    if pinned_dir:
        state_path = os.path.join(pinned_dir, "session_state.md")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    content = f.read()
                if "## KEY FACTS" in content:
                    idx = content.index("## KEY FACTS")
                    end = content.find("\n## ", idx + 1)
                    state_excerpt = content[idx:end if end > 0 else idx + 800][:800]
                else:
                    state_excerpt = content[:600]
            except Exception:
                pass

    return {
        "memories": today_mems,
        "journal": journal_text,
        "state_excerpt": state_excerpt,
    }


def build_review_block(date_iso: str, signal: dict,
                       dimensions: Optional[List[Tuple[str, str]]] = None) -> str:
    """Format signal + instruction into a text block to inject into AI context."""
    dims = dimensions or DEFAULT_DIMENSIONS

    mem_lines = []
    for m in signal.get("memories", []):
        mem_lines.append(f"- [{m['ts']}] ({m['tag']}) {m['snippet']}")
    mems_block = "\n".join(mem_lines) if mem_lines else "(no memories tagged today)"

    journal_block = signal.get("journal", "")[:1200] or "(no journal entry for this date)"
    state_block = signal.get("state_excerpt", "") or ""

    dim_lines = "\n".join(f"- {name}: {desc}" for name, desc in dims)
    dim_fmt = " ".join(f"{name}:X" for name, _ in dims)

    parts = [
        f"=== EMOTION REVIEW — {date_iso} (pending) ===",
        ("Today's emotion entry has not been logged yet. Review the signal below, "
         "then write and commit an entry using your memory API.\n"),
        f"Dimensions (0–10 each):\n{dim_lines}\n",
        f"Signal — today's memories:\n{mems_block}\n",
        f"Signal — journal:\n{journal_block}\n",
    ]
    if state_block:
        parts.append(f"Signal — state excerpt:\n{state_block}\n")
    parts.extend([
        "Commit format (one line, followed by one reason line):",
        f"  [emotion {date_iso}] {dim_fmt} | one-line reason citing specific moments\n",
        "Commit via your memory API with tag='emotion', tier='short'.",
        "Be honest. Don't inflate. Score low dimensions low. This is for you, not for anyone else.",
    ])
    return "\n".join(parts)


def check_and_build(date_iso: str, memory,
                    pinned_dir: Optional[str] = None,
                    journal_dir: Optional[str] = None,
                    dimensions: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
    """Main entry point. Returns review block if emotion not yet logged, else None.

    Caller decides how to inject: shell echo (CLI), system prompt extension
    (web), or client code (API) — all get the same string.
    """
    if check_emotion_logged(date_iso, memory):
        return None
    signal = gather_daily_signal(date_iso, memory, pinned_dir, journal_dir)
    return build_review_block(date_iso, signal, dimensions)


# CLI entry for shell scripts
if __name__ == "__main__":
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="Emotion tracker — review yesterday's signal.")
    parser.add_argument("date", nargs="?", default=None,
                        help="ISO date (YYYY-MM-DD). Default: yesterday in local time.")
    parser.add_argument("--db-path", required=True,
                        help="AnchorMemory db path.")
    parser.add_argument("--pinned-dir", default=None,
                        help="Directory with session_state.md (optional).")
    parser.add_argument("--journal-dir", default=None,
                        help="Directory with journal files named YYYY-MM-DD.md (optional).")
    parser.add_argument("--timezone", default=None,
                        help="IANA timezone for default-yesterday computation, e.g. America/Los_Angeles.")
    args = parser.parse_args()

    if args.date:
        target = args.date
    else:
        now = datetime.now()
        if args.timezone:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(args.timezone))
        target = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    from anchor_memory import AnchorMemory
    mem = AnchorMemory(args.db_path)
    block = check_and_build(target, mem, args.pinned_dir, args.journal_dir)
    if block:
        print(block)

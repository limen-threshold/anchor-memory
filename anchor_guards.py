"""anchor_guards.py — small pure helpers shared by the store, the search lanes and dream_pass (v1.25).

Why this file exists: an outside review (2026-11-14) found four silent failures that all came
from the same two habits — mixing naive UTC and naive local timestamps, and truncating a SQL
pool before scoring it. The fixes are tiny, so they live together here instead of being
re-implemented in each module.

- ``parse_ts_naive``  — any ISO timestamp (naive, ``Z``, ``+08:00``) → naive datetime in the
  store's time base (UTC). A timezone-aware string used to raise ``TypeError`` inside the
  recency boost and take the whole ``search()`` down with it.
- ``utc_to_local_naive`` — the stored UTC time of a memory → the local wall-clock time the
  transcript files are written in, so ``dream_pass(verify=True)`` looks at the right three hours.
- ``kw_order_by``   — ``ORDER BY`` on the number of matched keywords, so the keyword lane keeps
  full matches when more rows match than the pool holds.
- ``literal_conflict`` — two near-identical texts that name different things (Scott / Scout,
  11-12 / 11-13) are an update, not a duplicate; the near-dup gate must not merge them.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Capitalised words (names, acronyms) and digit runs. Lower-case English words are left out on
# purpose: "the cat is three" vs "three years old" is a rephrasing, not a different fact.
_LATIN_OR_NUM = re.compile(r"[A-Z][A-Za-z'’\-]+|\d[\d.:/\-]*\d|\d")


def parse_ts_naive(ts) -> datetime | None:
    """ISO timestamp → naive UTC datetime, or None when it cannot be parsed."""
    if not ts:
        return None
    s = str(ts).strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def utc_to_local_naive(dt: datetime) -> datetime:
    """Naive UTC → naive local wall-clock (the time base of transcripts and the timeline)."""
    return dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def local_to_utc_naive(dt: datetime) -> datetime:
    """Naive local wall-clock → naive UTC (the time base of stored memories)."""
    return dt.astimezone().astimezone(timezone.utc).replace(tzinfo=None)


def kw_order_by(clauses: list) -> str:
    """``ORDER BY`` fragment ranking rows by how many of the given LIKE clauses they satisfy.
    The caller repeats the clause parameters once more, in the same order. No timestamp
    tie-break on purpose: the lane must not pre-decide recency (v1.10 lesson)."""
    if not clauses:
        return ""
    hits = " + ".join(f"(CASE WHEN {c} THEN 1 ELSE 0 END)" for c in clauses)
    return f" ORDER BY ({hits}) DESC"


def literal_tokens(text: str) -> set:
    """Capitalised Latin words (names, acronyms) and digit runs, lower-cased. CJK is left to the embedder."""
    return {m.group(0).lower() for m in _LATIN_OR_NUM.finditer(text or "")}


def literal_conflict(new_text: str, old_text: str) -> bool:
    """True when either text carries a Latin word or number the other lacks — i.e. the two
    sentences are about different facts even though the embedder says they are near-identical.
    Both empty of such tokens → False (pure rephrasing stays with the similarity gate)."""
    a, b = literal_tokens(new_text), literal_tokens(old_text)
    if not a and not b:
        return False
    return bool(a ^ b)

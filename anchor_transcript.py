"""Raw transcripts — the searchable first layer (v1.20).

Memories are what the AI (or its curator) chose to write down. Transcripts are
what was actually said. Keeping both lets you go back to the words when a memory
looks wrong, and lets the dream pass check memories against them
(dream_extras.verify_recent). No LLM is involved anywhere in this module.

Layout: ``<db_path>/transcripts/YYYY-MM-DD.jsonl``, one JSON object per line:
``{"ts": "2026-11-08T22:51:03", "role": "user"|"assistant", "text": "..."}``
(plus an optional ``"conv"`` tag). The proxy appends every exchange after the
reply is complete (``ANCHOR_TRANSCRIPT=0`` turns it off); MCP-only setups can
log through the ``record_transcript`` tool, or write the files themselves —
the format is the whole contract.

Search is keyword-and-date, nothing cleverer: every term must appear (case-
insensitive; CJK terms match as substrings), optional ``date_from``/``date_to``
(YYYY-MM-DD, inclusive). Results are spread across days — newest day first,
round-robin — so one talkative day cannot fill the whole page and hide an older
match (the usual failure of "newest 20 lines").

CLI::

    python anchor_transcript.py --db-path ./anchor_data search "香港 第一次" --from 2026-10-01
    python anchor_transcript.py --db-path ./anchor_data days
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

DIR_NAME = "transcripts"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def transcript_dir(db_path: str) -> str:
    return os.path.join(db_path, DIR_NAME)


def _day_of(ts: str) -> str:
    return (ts or "")[:10]


def _read_day(path: str) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return rows


def append(db_path: str, role: str, text: str, ts: str = None, conv: str = None) -> str | None:
    """Append one line. Returns the file path, or None when nothing was written
    (empty text, or an exact repeat of one of the day's last 20 lines — clients
    that re-send history on regeneration would otherwise double every turn)."""
    text = (text or "").strip()
    if not text or role not in ("user", "assistant"):
        return None
    ts = ts or datetime.now().isoformat(timespec="seconds")
    d = transcript_dir(db_path)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_day_of(ts)}.jsonl")
    for prev in _read_day(path)[-20:]:
        if prev.get("role") == role and prev.get("text") == text:
            return None
    row = {"ts": ts, "role": role, "text": text}
    if conv:
        row["conv"] = conv
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def days(db_path: str) -> list:
    """[(YYYY-MM-DD, line_count), ...] oldest first."""
    d = transcript_dir(db_path)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".jsonl") and _DATE_RE.match(name):
            out.append((name[:-6], len(_read_day(os.path.join(d, name)))))
    return out


def _terms(query: str) -> list:
    return [t for t in re.split(r"\s+", (query or "").strip()) if t]


def _matches(text: str, terms: list) -> bool:
    low = (text or "").lower()
    return all(t.lower() in low for t in terms)


def lines_between(db_path: str, start_iso: str, end_iso: str) -> list:
    """All lines with start_iso <= ts < end_iso, oldest first (used by the
    dream-pass verifier to pull the conversation around a memory's timestamp)."""
    out = []
    for day, _n in days(db_path):
        if day < start_iso[:10] or day > end_iso[:10]:
            continue
        for row in _read_day(os.path.join(transcript_dir(db_path), day + ".jsonl")):
            ts = row.get("ts", "")
            if start_iso <= ts < end_iso:
                out.append(row)
    out.sort(key=lambda r: r.get("ts", ""))
    return out


def search(db_path: str, query: str = "", date_from: str = None, date_to: str = None,
           limit: int = 20) -> dict:
    """Keyword + date search over the transcript files.

    Returns {"hits": [row...], "total": matched lines, "days_scanned": n,
    "per_day": {day: count}, "truncated": bool}. Hits are newest day first,
    round-robin across days, newest line first within a day."""
    terms = _terms(query)
    limit = max(1, int(limit or 20))
    per_day = {}
    scanned = 0
    for day, _n in days(db_path):
        if date_from and day < date_from[:10]:
            continue
        if date_to and day > date_to[:10]:
            continue
        scanned += 1
        rows = [r for r in _read_day(os.path.join(transcript_dir(db_path), day + ".jsonl"))
                if _matches(r.get("text", ""), terms)]
        if rows:
            rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
            per_day[day] = rows
    total = sum(len(v) for v in per_day.values())
    hits = []
    order = sorted(per_day.keys(), reverse=True)
    i = 0
    while len(hits) < limit and any(i < len(per_day[d]) for d in order):
        for d in order:
            if i < len(per_day[d]) and len(hits) < limit:
                hits.append(per_day[d][i])
        i += 1
    return {"hits": hits, "total": total, "days_scanned": scanned,
            "per_day": {d: len(v) for d, v in per_day.items()}, "truncated": total > len(hits)}


def format_hits(res: dict, max_chars: int = 240) -> str:
    """Plain-text rendering for tool results / CLI: one line per hit."""
    lines = []
    for r in res.get("hits", []):
        t = (r.get("text") or "").replace("\n", " ")
        if len(t) > max_chars:
            t = t[:max_chars] + "…"
        lines.append(f"{r.get('ts', '')[:16]} {r.get('role', '?')}: {t}")
    if res.get("truncated"):
        per = ", ".join(f"{d}×{n}" for d, n in sorted(res.get("per_day", {}).items(), reverse=True))
        lines.append(f"({res.get('total')} matches in {len(res.get('per_day', {}))} days; showing {len(res.get('hits', []))}. Per day: {per}. Narrow the dates to see more.)")
    return "\n".join(lines) if lines else "(no transcript lines matched)"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Search the raw transcripts (no LLM).")
    ap.add_argument("--db-path", default="./anchor_data")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("search")
    s.add_argument("query", nargs="?", default="")
    s.add_argument("--from", dest="date_from")
    s.add_argument("--to", dest="date_to")
    s.add_argument("--limit", type=int, default=20)
    sub.add_parser("days")
    args = ap.parse_args()
    if args.cmd == "days":
        for day, n in days(args.db_path):
            print(f"{day}  {n} lines")
    else:
        print(format_hits(search(args.db_path, args.query, args.date_from, args.date_to, args.limit)))

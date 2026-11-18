"""anchor_notebook.py — the small books: a memo of things promised, quotes that stay, a failure log,
and a readout of how old the session state is (v1.31). All zero-model.

Four pieces, each born from a failure that semantic memory does not cover:

**Memo** — ``<db>/notebook/memo.md``. "I'll come back to that" said in a conversation is a promise
with a due date, and a due date is exactly what similarity search cannot guarantee to surface. So a
narrow pen: ``memo_jot`` appends one dated line (and nothing else: no path, no overwrite — it is
safe to hold while reading text from outside), ``memo_edit`` strikes or corrects a line by exact
replacement with a backup, ``read_memo`` shows the open ones. Each jotted line is also stored as a
memory, so search finds it too. The machine hands the notebook back: lines written in the last two
days are shown as written, and the oldest open lines past a day are handed over to be done, stored
or struck — in ``wakeup`` and in front of every heartbeat.

**Permanent quotes** — ``<pinned>/permanent_quotes.md``. Sentences that must survive every rewrite of
the session state do not belong inside the session state: sooner or later a rewrite drops them.
``keep_quote`` appends one dated line to a separate pinned file, which is loaded with the identity
files in every window. Editing or removing a line goes through ``write_identity_file`` (the previous
version is archived).

**Failure log** — ``<db>/notebook/failures.jsonl`` + a readable ``failure_log.md``. ``log_failure``
is a safe channel for "that memory was wrong": it only appends, it never rewrites a memory. Seven
types. The log covers only failures that surfaced and were caught; the mirror file says so on top.

**Session-state readout** — facts, not instructions: when the state was last rewritten, and which of
its lines say "tomorrow" / "today" / "soon" and were first written on a day that has passed (dated
from the archive of earlier versions). The pen is ``write_session_state``; whether to pick it up is
the AI's call. On write, lines that use a relative time word with no date on the same line are
listed in the receipt.
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import re
import shutil
from datetime import date, datetime, timedelta

QUOTES_FILE = "permanent_quotes.md"
MEMO_MAX = 500
STALE_DAYS = 7
FTYPES = {
    "bundled": "two facts fused into one memory",
    "ghost": "cited a memory that never existed",
    "evaporation": "a real memory that should have surfaced did not",
    "hallucination": "stated a fact that is in no memory",
    "misrecall": "the wrong memory surfaced",
    "outdated": "a coherent but stale memory was treated as current",
    "tier_error": "a memory sits at the wrong tier",
}
IMPACTS = ("low", "medium", "high")

TOOL_DEFS = [
    {"name": "memo_jot",
     "description": ("Append one line to your memo — for things you said you would come back to: a promise made in "
                     "passing, something to check, an account to settle. One line, dated by the machine, at most "
                     f"{MEMO_MAX} chars; it can only append. The line is also stored as a memory. Open lines are handed "
                     "back to you later (wakeup, heartbeat)."),
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string", "description": "what is to be done, who it concerns, where the material is"}},
         "required": ["text"]}},
    {"name": "memo_edit",
     "description": ("Edit your memo by exact replacement: old_string → new_string. For striking a line that is done "
                     "(wrap it in ~~…~~ and add what came of it) or correcting one. old_string must occur exactly once; "
                     "otherwise nothing changes and you are told. The previous version is backed up."),
     "inputSchema": {"type": "object", "properties": {
         "old_string": {"type": "string"}, "new_string": {"type": "string"}},
         "required": ["old_string", "new_string"]}},
    {"name": "read_memo",
     "description": "Read your memo. open_only=true (default) leaves out struck lines.",
     "inputSchema": {"type": "object", "properties": {"open_only": {"type": "boolean", "default": True}}}},
    {"name": "keep_quote",
     "description": ("Keep a sentence for good: appends one dated line to permanent_quotes.md in your pinned layer, "
                     "which every window loads. For words that must survive any rewrite of your session state. "
                     "To change or remove a line later, use write_identity_file on permanent_quotes.md."),
     "inputSchema": {"type": "object", "properties": {
         "quote": {"type": "string", "description": "the sentence, verbatim"},
         "who": {"type": "string", "description": "who said it"},
         "note": {"type": "string", "description": "optional: when / where / why it stays"}},
         "required": ["quote"]}},
    {"name": "log_failure",
     "description": ("Record a memory failure when you catch one — yours, in the first person. Append-only: it never "
                     "rewrites a memory. ftype: " + "; ".join(f"{k} = {v}" for k, v in FTYPES.items()) + "."),
     "inputSchema": {"type": "object", "properties": {
         "ftype": {"type": "string", "enum": list(FTYPES)},
         "what": {"type": "string", "description": "what went wrong, concretely"},
         "impact": {"type": "string", "enum": list(IMPACTS), "default": "low"},
         "memory_id": {"type": "string", "description": "the memory involved, if there is one"}},
         "required": ["ftype", "what"]}},
    {"name": "list_failures",
     "description": "Your failure log, newest first. Optional ftype filter.",
     "inputSchema": {"type": "object", "properties": {
         "limit": {"type": "integer", "default": 20}, "ftype": {"type": "string"}}}},
]
TOOL_NAMES = {t["name"] for t in TOOL_DEFS}


# ── memo ──────────────────────────────────────────────────────────────────────

_JOT = re.compile(r"^\s*-\s*\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\]\s*(.*)$")


def notebook_dir(db_path: str) -> str:
    return os.path.join(db_path or ".", "notebook")


def memo_path(db_path: str) -> str:
    return os.path.join(notebook_dir(db_path), "memo.md")


def _memo_lines(db_path: str) -> list:
    try:
        with open(memo_path(db_path), encoding="utf-8") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def _struck(line: str) -> bool:
    return "~~" in line.strip()[:8]


def memo_jot(db_path: str, text: str, store_fn=None, now: datetime = None) -> str:
    txt = " ".join(str(text or "").split())
    if not txt:
        return "Nothing jotted: the line was empty."
    cut = len(txt) > MEMO_MAX
    txt = txt[:MEMO_MAX]
    now = now or datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M")
    os.makedirs(notebook_dir(db_path), exist_ok=True)
    p = memo_path(db_path)
    new = not os.path.exists(p)
    with open(p, "a", encoding="utf-8") as f:
        if new:
            f.write("# Memo\n")
        f.write(f"\n- [{stamp}] {txt}")
    backup = ""
    if store_fn:
        try:
            mid = store_fn(f"[memo {stamp}] {txt}")
            backup = f" Also stored as a memory ({mid})." if mid else ""
        except Exception as e:
            backup = f" (The memory copy failed — {type(e).__name__}; the memo line is written.)"
    return f"Jotted: - [{stamp}] {txt[:80]}{'…' if len(txt) > 80 else ''}{' (cut at %d chars)' % MEMO_MAX if cut else ''}.{backup}"


def memo_edit(db_path: str, old: str, new: str) -> str:
    p = memo_path(db_path)
    if not os.path.exists(p):
        return "There is no memo yet (memo_jot creates it)."
    if not old:
        return "Nothing changed: old_string was empty."
    with open(p, encoding="utf-8") as f:
        text = f.read()
    n = text.count(old)
    if n != 1:
        return (f"Nothing changed: old_string occurs {n} times in the memo — it must occur exactly once. "
                + ("Quote it character for character (read_memo shows the lines)." if n == 0 else "Quote more of the line."))
    bdir = os.path.join(notebook_dir(db_path), "memo_backups")
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(p, os.path.join(bdir, f"memo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.md"))
    for old_b in sorted(os.listdir(bdir))[:-50]:
        os.remove(os.path.join(bdir, old_b))
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new or ""))
    return "Memo edited (the previous version is in notebook/memo_backups/)."


def memo_rows(db_path: str, open_only: bool = True) -> list:
    """[(datetime | None, line)] in file order."""
    out = []
    for ln in _memo_lines(db_path):
        if not ln.strip().startswith("-"):
            continue
        if open_only and _struck(ln):
            continue
        m = _JOT.match(ln.replace("~~", ""))
        ts = None
        if m:
            try:
                ts = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
            except ValueError:
                pass
        out.append((ts, ln.strip()))
    return out


def memo_block(db_path: str, now: datetime = None, recent_n: int = 6, recent_hours: int = 48,
               stale_n: int = 5, stale_hours: int = 24, cap: int = 3000) -> str:
    """What the machine hands back: the oldest open lines past a day, and the newest lines of the last two days
    (a line appears once). '' when the memo has no open lines."""
    now = now or datetime.now()
    rows = [(ts, l) for ts, l in memo_rows(db_path) if ts]
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0])
    stale = [(ts, l) for ts, l in rows if now - ts >= timedelta(hours=stale_hours)][:stale_n]
    stale_set = {l for _, l in stale}
    recent = [l for ts, l in rows if now - ts <= timedelta(hours=recent_hours) and l not in stale_set][-recent_n:]
    parts = []
    if stale:
        n_open = len(rows)
        parts.append(f"[Your memo: {n_open} open line(s). The oldest, as you wrote them — do it, store it, or strike it "
                     "(memo_edit); leaving it is also a choice:]\n" + "\n".join(l[:600] for _, l in stale))
    if recent:
        parts.append("[Your memo, the latest lines of the last two days, as written, newest last:]\n" + "\n".join(l[:600] for l in recent))
    return "\n\n".join(parts)[:cap]


# ── permanent quotes ──────────────────────────────────────────────────────────

def keep_quote(pinned_dir: str, quote: str, who: str = "", note: str = "", now: datetime = None) -> str:
    q = " ".join(str(quote or "").split())
    if not q:
        return "Nothing kept: the quote was empty."
    now = now or datetime.now()
    os.makedirs(pinned_dir, exist_ok=True)
    p = os.path.join(pinned_dir, QUOTES_FILE)
    old = ""
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            old = f.read()
    if q in old:
        return "Already there: this sentence is in permanent_quotes.md. Nothing added."
    tail = " — " + ", ".join(x for x in (" ".join(str(who or "").split()), now.strftime("%Y-%m-%d"), " ".join(str(note or "").split())) if x)
    with open(p, "a", encoding="utf-8") as f:
        if not old.strip():
            f.write("# Permanent quotes\n\nSentences kept across every window. Each line was added with keep_quote.\n")
        f.write(f"\n> {q}{tail}\n")
    n = sum(1 for l in (old + f"\n> {q}").splitlines() if l.lstrip().startswith(">"))
    msg = f"Kept ({n} line(s) in permanent_quotes.md). It is part of your pinned layer from the next window on."
    manifest = os.path.join(pinned_dir, "_order.txt")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            listed = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        if QUOTES_FILE not in listed:
            msg += (f" WARNING: _order.txt exists and does not list {QUOTES_FILE}, so it will NOT be loaded until "
                    "that line is added to _order.txt.")
    size = len(old) + len(q) + len(tail)
    if size > 8000:
        msg += f" The file is now {size} chars; everything in it is injected every turn."
    return msg


# ── failure log ───────────────────────────────────────────────────────────────

FAILURE_HEADER = (
    "# Failure log\n\n"
    "> **Blind spot**: this log covers only failures that surfaced in a conversation and were caught — by me or by the\n"
    "> person I was talking with. A memory that went wrong quietly is not here. It is the ledger of the errors I\n"
    "> caught, not of all errors.\n\n"
    "Types: " + "; ".join(f"`{k}` ({v})" for k, v in FTYPES.items()) + ".\n"
    "Impact: low / medium / high. The subject is always \"I\". Append-only in `failures.jsonl`; this file is rewritten from it.\n\n---\n\n")


def _failures_path(db_path: str) -> str:
    return os.path.join(notebook_dir(db_path), "failures.jsonl")


def failures(db_path: str, limit: int = 20, ftype: str = None) -> list:
    rows = []
    try:
        with open(_failures_path(db_path), encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    if ftype:
        rows = [r for r in rows if r.get("ftype") == ftype]
    return rows[::-1][: max(1, int(limit or 20))]


def log_failure(db_path: str, ftype: str, what: str, impact: str = "low", memory_id: str = "",
                source: str = "self", now: datetime = None) -> dict:
    ftype = str(ftype or "").strip().lower()
    what = " ".join(str(what or "").split())
    if ftype not in FTYPES:
        return {"error": f"ftype must be one of: {', '.join(FTYPES)}"}
    if not what:
        return {"error": "say what went wrong (what was empty)"}
    impact = impact if impact in IMPACTS else "low"
    row = {"ts": (now or datetime.now()).isoformat(timespec="seconds"), "ftype": ftype, "impact": impact,
           "what": what[:2000], "source": source}
    if memory_id:
        row["memory_id"] = str(memory_id)[:80]
    os.makedirs(notebook_dir(db_path), exist_ok=True)
    with open(_failures_path(db_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        with open(os.path.join(notebook_dir(db_path), "failure_log.md"), "w", encoding="utf-8") as f:
            f.write(FAILURE_HEADER)
            for r in failures(db_path, 200):
                mem = f" · [{r['memory_id']}]" if r.get("memory_id") else ""
                f.write(f"- **{r['ts'].replace('T', ' ')}** · `{r['ftype']}` · impact **{r['impact']}**{mem} · _{r.get('source', 'self')}_\n  {r['what']}\n")
    except Exception:
        pass
    return {"status": "logged", "entry": row, "total": len(failures(db_path, 10 ** 6))}


# ── session-state readout ─────────────────────────────────────────────────────

_REL = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (?:morning|afternoon|evening|week|weekend)|just now|right now|"
    r"in a (?:few|couple of) (?:minutes|hours|days)|next week|last night|soon|about to)\b"
    r"|今天|今晚|今早|明天|明早|明晚|后天|昨天|昨晚|前天|刚才|刚刚|马上|即将|这会儿|此刻|下周|上周", re.I)
_ABS = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\b\d{1,2}[-/]\d{1,2}\b|\d{1,2}\s*月\s*\d{1,2}\s*[日号]?"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I)


def _norm(line: str) -> str:
    return " ".join(re.sub(r"[*_`]+", "", line).split())


def _content_lines(text: str) -> list:
    out = []
    for l in (text or "").splitlines():
        s = l.strip()
        if not s or s.startswith("#") or s.startswith("[continuity]") or re.fullmatch(r"[-=*_ ]{3,}", s) or s.startswith("<!--"):
            continue
        out.append(_norm(l))
    return out


def undated_relative_lines(text: str) -> list:
    """Lines that say 'tomorrow' / 'today' / 'soon' with no date on the same line: read back later, they will be
    taken to mean the day they are read."""
    return [l for l in _content_lines(text) if _REL.search(l) and not _ABS.search(l)]


def _versions(pinned_dir: str) -> list:
    """[(written_at: datetime, text)] oldest first: archived versions (file mtime = when that version was
    written — archiving moves the file and keeps its mtime) and the current file."""
    paths = glob.glob(os.path.join(pinned_dir, "session_state_archive", "*.md")) + [os.path.join(pinned_dir, "session_state.md")]
    out = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                out.append((datetime.fromtimestamp(os.path.getmtime(p)), f.read()))
        except Exception:
            pass
    out.sort(key=lambda v: v[0])
    return out[-60:]


def first_written(pinned_dir: str, lines: list, fuzzy: float = 0.85) -> dict:
    """line → date of the earliest version that already contained it (exactly, or lightly edited)."""
    vers = [(ts.date(), _content_lines(t)) for ts, t in _versions(pinned_dir)]
    out = {}
    for line in lines:
        for d, vl in vers:
            if line in vl or any(abs(len(v) - len(line)) < 40 and difflib.SequenceMatcher(None, v, line).ratio() >= fuzzy for v in vl):
                out[line] = d
                break
    return out


def state_readout(pinned_dir: str, now: datetime = None) -> str:
    """Facts about the session state and the quotes file. '' when there is neither."""
    now = now or datetime.now()
    today = now.date()
    lines = []
    ps = os.path.join(pinned_dir or "", "session_state.md")
    if os.path.exists(ps):
        wrote = datetime.fromtimestamp(os.path.getmtime(ps))
        age = max(0, (today - wrote.date()).days)
        lines.append(f"- session_state.md was last rewritten on {wrote.strftime('%Y-%m-%d')} ({age} day(s) ago).")
        with open(ps, encoding="utf-8") as f:
            rel = [l for l in _content_lines(f.read()) if _REL.search(l)]
        born = first_written(pinned_dir, rel) if rel else {}
        stale = [(d, l) for l, d in born.items() if isinstance(d, date) and (today - d).days >= 1]
        if stale:
            ex = "; ".join(f"[first written {d.isoformat()}] {l[:70]}" for d, l in stale[:3])
            lines.append(f"- {len(stale)} line(s) in it say 'today' / 'tomorrow' / 'soon' and were first written on a day "
                         f"that has passed: {ex}{' …' if len(stale) > 3 else ''}")
        elif age >= STALE_DAYS:
            lines.append(f"- That is more than {STALE_DAYS} days.")
    pq = os.path.join(pinned_dir or "", QUOTES_FILE)
    if os.path.exists(pq):
        with open(pq, encoding="utf-8") as f:
            n = sum(1 for l in f if l.lstrip().startswith(">"))
        lines.append(f"- permanent_quotes.md: {n} line(s), last changed {datetime.fromtimestamp(os.path.getmtime(pq)).strftime('%Y-%m-%d')}.")
    if not lines:
        return ""
    return ("[Readout of your own files — facts, not instructions; whether to change anything is yours to decide. "
            "The pens: write_session_state (complete new state), keep_quote.]\n" + "\n".join(lines))


def status(db_path: str, pinned_dir: str, now: datetime = None) -> str:
    """Everything the machine hands back at wakeup / before a heartbeat: state readout + memo lines."""
    parts = []
    for fn in (lambda: state_readout(pinned_dir, now), lambda: memo_block(db_path, now)):
        try:
            b = fn()
            if b:
                parts.append(b)
        except Exception:
            pass
    return "\n\n".join(parts)


def handle(db_path: str, pinned_dir: str, name: str, args: dict, store_fn=None) -> dict:
    args = args or {}
    if name == "memo_jot":
        return {"result": memo_jot(db_path, args.get("text", ""), store_fn)}
    if name == "memo_edit":
        return {"result": memo_edit(db_path, args.get("old_string", ""), args.get("new_string", ""))}
    if name == "read_memo":
        rows = memo_rows(db_path, args.get("open_only", True) is not False)
        return {"lines": [l for _, l in rows], "count": len(rows), "path": "notebook/memo.md"}
    if name == "keep_quote":
        return {"result": keep_quote(pinned_dir, args.get("quote", ""), args.get("who", ""), args.get("note", ""))}
    if name == "log_failure":
        return log_failure(db_path, args.get("ftype", ""), args.get("what", ""), args.get("impact", "low"), args.get("memory_id", ""))
    if name == "list_failures":
        return {"failures": failures(db_path, args.get("limit", 20), args.get("ftype")),
                "blind_spot": "Only failures that surfaced and were caught are here."}
    return {"error": f"unknown notebook tool {name}"}

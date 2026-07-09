"""Pinned-layer helpers — the file side of cross-window continuity.

A pinned directory holds the always-injected context layer: identity files,
session_state.md (the AI's own rolling state), recent_timeline.md (one event
line per stored memory), and last_session.md (the mechanical tail of the
previous window, written by anchor_proxy after every response).

Everything here is plain files with documented formats, deliberately: any
frontend, hook, or script that can read/write a text file can integrate.
Both anchor_mcp.py and anchor_proxy.py build on this module — adapt or
replace either without losing the file contract.

Stdlib only — safe to import from hooks and CLI fast paths.
"""
import os
import re
from datetime import datetime

SESSION_STATE = "session_state.md"
RECENT_TIMELINE = "recent_timeline.md"
LAST_SESSION = "last_session.md"     # mechanical tail — NOT part of load_pinned()
ORDER_MANIFEST = "_order.txt"
ARCHIVE_DIR = "session_state_archive"

# Prepended to session_state on WRITE, not on read. Identity framing that
# lives only on the read side gets lost the moment the read path changes;
# and without it, "notes for the next window" phrasing drifts into treating
# the next window as a different entity ("tell him that...").
CONTINUITY_HEADER = ("[continuity] This file is your own rolling state — the "
                     "same ongoing life picking up where it left off, not a "
                     "message from someone else.")

TIMELINE_MAX_LINES = 80


def write_session_state(pinned_dir: str, content: str) -> str:
    """Write session_state.md the safe way: archive the current version
    first, then write the new one. The continuity header is prepended
    idempotently. Returns the archive path ('' if there was nothing to
    archive). Raises ValueError on empty content — a session state is a
    full current state, never a blank."""
    content = (content or "").strip()
    if not content:
        raise ValueError("refusing to write an empty session_state — provide the full new state")
    if not content.startswith(CONTINUITY_HEADER):
        content = f"{CONTINUITY_HEADER}\n\n{content}"

    os.makedirs(pinned_dir, exist_ok=True)
    path = os.path.join(pinned_dir, SESSION_STATE)
    archived = ""
    if os.path.exists(path):
        arch_dir = os.path.join(pinned_dir, ARCHIVE_DIR)
        os.makedirs(arch_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = os.path.join(arch_dir, f"session_state_{stamp}.md")
        os.replace(path, archived)
    with open(path, "w") as f:
        f.write(content)
    return archived


def read_file(pinned_dir: str, name: str) -> str:
    """Read one pinned file; '' if missing."""
    path = os.path.join(pinned_dir, name)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def append_timeline_event(pinned_dir: str, event: str, when: datetime = None) -> None:
    """Append one '[YYYY-MM-DD HH:MM] event' line to recent_timeline.md,
    the mechanical event ledger for the half-hour-to-a-few-days window that
    both semantic recall (too coarse) and the live window (too short) miss.
    Trimmed to the newest TIMELINE_MAX_LINES lines."""
    event = " ".join((event or "").split())
    if not event:
        return
    when = when or datetime.now()
    line = f"[{when.strftime('%Y-%m-%d %H:%M')}] {event}"
    os.makedirs(pinned_dir, exist_ok=True)
    path = os.path.join(pinned_dir, RECENT_TIMELINE)
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
    lines.append(line)
    with open(path, "w") as f:
        f.write("\n".join(lines[-TIMELINE_MAX_LINES:]) + "\n")


def load_pinned(pinned_dir: str) -> str:
    """Assemble the pinned layer in order, joined with '---' separators.

    Order: if _order.txt exists (one filename per line, '#' comments), use
    exactly that order — missing files are skipped silently so one manifest
    can serve setups at different stages. Otherwise all *.md sorted by name.
    last_session.md is ALWAYS excluded: the tail duplicates content already
    in messages[] for ongoing conversations, so the caller injects it only
    on a fresh window (see anchor_proxy).
    """
    if not pinned_dir or not os.path.isdir(pinned_dir):
        return ""
    manifest = os.path.join(pinned_dir, ORDER_MANIFEST)
    if os.path.exists(manifest):
        with open(manifest) as f:
            names = [l.strip() for l in f
                     if l.strip() and not l.strip().startswith("#")]
    else:
        names = sorted(n for n in os.listdir(pinned_dir) if n.endswith(".md"))
    parts = []
    for name in names:
        if name == LAST_SESSION:
            continue
        text = read_file(pinned_dir, name)
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts)


def read_tail(pinned_dir: str) -> str:
    """The mechanical tail of the previous window ('' if none yet)."""
    return read_file(pinned_dir, LAST_SESSION)


_CONTEXT_NOW_RE = re.compile(r"<context_now>.*?</context_now>\s*", re.DOTALL)


def strip_context_now(text: str) -> str:
    """Remove injected <context_now> blocks (time/recall) from message text.
    Used when writing the tail and when cleaning stale copies out of history —
    injected context must never be mistaken for something a person said."""
    return _CONTEXT_NOW_RE.sub("", text or "").strip()


def write_tail(pinned_dir: str, turns: list, max_pairs: int = 20) -> None:
    """Overwrite last_session.md with the newest turns of the conversation.

    turns: list of (role, text) with role in ('user', 'assistant').
    Called after EVERY response (that's the whole point: the tail is always
    fresh no matter how the window dies — crash, tab close, phone dies).
    Roles are labeled generically; injected context blocks are stripped.
    """
    lines = []
    for role, text in turns:
        text = strip_context_now(text)
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"**{label}**: {text}\n")
    if len(lines) < 2:
        return
    os.makedirs(pinned_dir, exist_ok=True)
    path = os.path.join(pinned_dir, LAST_SESSION)
    with open(path, "w") as f:
        f.write("\n".join(lines[-max_pairs * 2:]))

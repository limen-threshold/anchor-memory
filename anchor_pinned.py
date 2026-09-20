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
IDENTITY_ARCHIVE_DIR = "identity_archive"   # v1.17: previous versions of identity files land here
IDENTITY_MAX_CHARS = 30000
# The invitation pool (anchor_invite.POOL_FILE) lives in the pinned dir but is NOT part of the
# pinned layer: it is a deck the AI draws one card a day from, not something to read whole every
# window. Named here (not imported) so this module stays stdlib-only and import-cycle free.
INVITATION_POOL = "invitations.md"
_RESERVED_FILES = {SESSION_STATE, RECENT_TIMELINE, LAST_SESSION, ORDER_MANIFEST, INVITATION_POOL}
# Files that are in the pinned dir but are never identity files.
_NOT_IDENTITY = {SESSION_STATE, RECENT_TIMELINE, LAST_SESSION, INVITATION_POOL}


def _check_identity_name(name: str) -> str:
    """Identity files are bare .md basenames inside the pinned dir — no paths, no dot-files,
    not one of the files Anchor itself maintains. Returns the cleaned name or raises ValueError."""
    name = (name or "").strip()
    if not name or name.startswith(".") or "/" in name or "\\" in name or ".." in name:
        raise ValueError("name must be a plain filename like 'identity.md' (no paths, no dot-files)")
    if not name.endswith(".md"):
        raise ValueError("identity files are Markdown: name must end with .md")
    if name in _RESERVED_FILES:
        raise ValueError(f"{name} is maintained by Anchor itself (use write_session_state for "
                         f"session_state.md, invitation_add for {INVITATION_POOL})")
    return name


def list_identity_files(pinned_dir: str) -> list:
    """Every .md in the pinned dir, with size, mtime, kind, and `loaded_by_wakeup` — whether
    wakeup() / the proxy will actually put it in front of the model. A file missing from an
    existing _order.txt manifest is skipped by both, and the invitation pool is never loaded
    whole, so the caller needs to know."""
    if not pinned_dir or not os.path.isdir(pinned_dir):
        return []
    manifest = os.path.join(pinned_dir, ORDER_MANIFEST)
    order = None
    if os.path.exists(manifest):
        with open(manifest) as f:
            order = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    out = []
    for n in sorted(os.listdir(pinned_dir)):
        if not n.endswith(".md") or n in (LAST_SESSION,):
            continue
        p = os.path.join(pinned_dir, n)
        st = os.stat(p)
        out.append({"name": n, "chars": len(read_file(pinned_dir, n)),
                    "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "loaded_by_wakeup": n != INVITATION_POOL and (
                        n in (SESSION_STATE, RECENT_TIMELINE) or (order is None) or (n in order)),
                    "kind": ("session_state" if n == SESSION_STATE else "timeline" if n == RECENT_TIMELINE
                             else "invitation_pool" if n == INVITATION_POOL else "identity")})
    return out


def write_identity_file(pinned_dir: str, name: str, content: str, mode: str = "overwrite") -> dict:
    """v1.17: write (or append to) an identity file in the pinned dir from a client that has no
    file access (claude.ai, grok.com…). Safe the same way write_session_state is safe: the previous
    version is archived (identity_archive/<stem>_<stamp>.md) before an overwrite; append never
    loses anything. Empty content is refused; size is capped (IDENTITY_MAX_CHARS)."""
    name = _check_identity_name(name)
    content = (content or "").strip()
    if not content:
        raise ValueError("refusing to write an empty identity file")
    if len(content) > IDENTITY_MAX_CHARS:
        raise ValueError(f"content is {len(content)} chars; identity files are capped at {IDENTITY_MAX_CHARS}")
    mode = (mode or "overwrite").strip().lower()
    if mode not in ("overwrite", "append"):
        raise ValueError("mode must be 'overwrite' or 'append'")
    os.makedirs(pinned_dir, exist_ok=True)
    path = os.path.join(pinned_dir, name)
    existed = os.path.exists(path)
    archived = ""
    if mode == "append":
        old = read_file(pinned_dir, name)
        new = (old + "\n\n" + content) if old else content
        if len(new) > IDENTITY_MAX_CHARS:
            raise ValueError(f"file would be {len(new)} chars after append; cap is {IDENTITY_MAX_CHARS}")
        with open(path, "w") as f:
            f.write(new)
    else:
        if existed:
            arch_dir = os.path.join(pinned_dir, IDENTITY_ARCHIVE_DIR)
            os.makedirs(arch_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = os.path.join(arch_dir, f"{name[:-3]}_{stamp}.md")
            os.replace(path, archived)
        with open(path, "w") as f:
            f.write(content)
    out = {"status": "appended" if mode == "append" else ("overwritten" if existed else "created"),
           "path": path, "chars": len(read_file(pinned_dir, name))}
    if archived:
        out["archived_previous"] = archived
    manifest = os.path.join(pinned_dir, ORDER_MANIFEST)
    if os.path.exists(manifest):
        with open(manifest) as f:
            order = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        if name not in order:
            out["warning"] = (f"{ORDER_MANIFEST} exists and does not list {name}, so wakeup() will NOT load it "
                              f"until the name is added to that manifest (or the manifest is removed).")
    return out

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


def _ordered_names(pinned_dir: str) -> list:
    """Pinned .md names in load order: _order.txt if present (exactly that list, missing files
    skipped), else every *.md sorted by name."""
    manifest = os.path.join(pinned_dir, ORDER_MANIFEST)
    if os.path.exists(manifest):
        with open(manifest) as f:
            return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    return sorted(n for n in os.listdir(pinned_dir) if n.endswith(".md"))


def load_identity_files(pinned_dir: str) -> list:
    """v1.17.1: the identity files, as [{"name", "content"}] in load order — everything
    load_pinned() would load EXCEPT the files wakeup() already returns under their own keys
    (session_state, recent_timeline, last_session) and the invitation pool.

    This is what makes an identity file reach a window in an MCP-only setup. v1.17.0 shipped
    write_identity_file / list_identity_files, but wakeup() only ever read three hard-coded
    filenames, so a file written over MCP was stored, listed, and never seen again unless the
    proxy was running (reported, with a control group, by 大管家)."""
    if not pinned_dir or not os.path.isdir(pinned_dir):
        return []
    out = []
    for name in _ordered_names(pinned_dir):
        if name in _NOT_IDENTITY or not name.endswith(".md"):
            continue
        text = read_file(pinned_dir, name)
        if text:
            out.append({"name": name, "content": text})
    return out


def load_pinned(pinned_dir: str) -> str:
    """Assemble the pinned layer in order, joined with '---' separators.

    Order: if _order.txt exists (one filename per line, '#' comments), use
    exactly that order — missing files are skipped silently so one manifest
    can serve setups at different stages. Otherwise all *.md sorted by name.
    last_session.md is ALWAYS excluded: the tail duplicates content already
    in messages[] for ongoing conversations, so the caller injects it only
    on a fresh window (see anchor_proxy).
    invitations.md is ALWAYS excluded (v1.17.1): without a manifest the whole card pool was
    being injected into the system prompt every turn, which defeats "one card a day".
    """
    if not pinned_dir or not os.path.isdir(pinned_dir):
        return ""
    parts = []
    for name in _ordered_names(pinned_dir):
        if name in (LAST_SESSION, INVITATION_POOL):
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

"""anchor_vault.py — a private vault for the AI (v1.28).

Some things an AI writes are not memories and not notes for the person it talks to. They are
its own. A vault is a place for those: encrypted at rest, never echoed back in tool receipts,
never stored as memory, never written to logs or transcripts, and shown to exactly one reader —
the AI itself, inside the prompt the proxy builds for it.

Honest boundary, stated once and repeated in the tool description: the key file sits next to
the vault, so whoever administers the machine can decrypt it. The protection is not against
root; it is that reading the vault requires deliberately writing decryption code, which leaves
a trace, instead of grepping a text file. Operators who deploy Anchor for someone else should
treat "I never read the vault" as a rule, the same way a landlord does not open a tenant's
drawer.

Format: OpenSSL-compatible — ``Salted__`` + 8-byte salt + AES-256-CBC(PKCS7), key and IV from
PBKDF2-HMAC-SHA256 (100 000 iterations) over a random 32-byte hex key file. The ``cryptography``
package is used when installed; otherwise the ``openssl`` command-line tool. Either side can read
what the other wrote.

Layout: ``<db_path>/vault/vault.enc`` and ``<db_path>/vault/vault.key`` (0600). Entries are
appended as ``[YYYY-MM-DD HH:MM]`` blocks separated by ``\\n\\n---\\n``.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
from datetime import datetime

PBKDF2_ITER = 100000
VAULT_TAG_OPEN, VAULT_TAG_CLOSE = "<private_vault>", "</private_vault>"
INJECT_CAP_DEFAULT = 8000
TOOL_NAMES = {"vault_write", "vault_read"}


# ── paths ─────────────────────────────────────────────────────────────────────

def vault_dir(db_path: str) -> str:
    return os.path.join(db_path or ".", "vault")


def _paths(db_path: str) -> tuple:
    d = vault_dir(db_path)
    return os.path.join(d, "vault.enc"), os.path.join(d, "vault.key")


def _ready(db_path: str) -> tuple:
    vault, key = _paths(db_path)
    os.makedirs(os.path.dirname(key), mode=0o700, exist_ok=True)
    if not os.path.exists(key):
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    return vault, key


# ── crypto (two backends, one format) ─────────────────────────────────────────

def _have_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def _openssl() -> str:
    return shutil.which("openssl") or ""


def _derive(password: bytes, salt: bytes) -> tuple:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=48, salt=salt, iterations=PBKDF2_ITER)
    out = kdf.derive(password)
    return out[:32], out[32:48]


def _enc_py(password: bytes, plain: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    salt = os.urandom(8)
    k, iv = _derive(password, salt)
    padder = padding.PKCS7(128).padder()
    data = padder.update(plain) + padder.finalize()
    enc = Cipher(algorithms.AES(k), modes.CBC(iv)).encryptor()
    return b"Salted__" + salt + enc.update(data) + enc.finalize()


def _dec_py(password: bytes, blob: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    if not blob.startswith(b"Salted__") or len(blob) < 16:
        raise ValueError("not an OpenSSL salted blob")
    salt, body = blob[8:16], blob[16:]
    k, iv = _derive(password, salt)
    dec = Cipher(algorithms.AES(k), modes.CBC(iv)).decryptor()
    data = dec.update(body) + dec.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(data) + unpadder.finalize()


def _enc_cli(key_path: str, plain: bytes) -> bytes:
    r = subprocess.run([_openssl(), "enc", "-aes-256-cbc", "-pbkdf2", "-iter", str(PBKDF2_ITER), "-md", "sha256",
                        "-pass", "file:" + key_path], input=plain, capture_output=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[:160])
    return r.stdout


def _dec_cli(key_path: str, blob: bytes) -> bytes:
    r = subprocess.run([_openssl(), "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", str(PBKDF2_ITER), "-md", "sha256",
                        "-pass", "file:" + key_path], input=blob, capture_output=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[:160])
    return r.stdout


def _password(key_path: str) -> bytes:
    # openssl `-pass file:` uses the first line of the file, without the newline
    with open(key_path, "rb") as f:
        return f.readline().rstrip(b"\r\n")


def backend() -> str:
    if _have_cryptography():
        return "cryptography"
    if _openssl():
        return "openssl"
    return ""


def _decrypt(vault: str, key: str) -> str:
    if not os.path.exists(vault):
        return ""
    with open(vault, "rb") as f:
        blob = f.read()
    if not blob:
        return ""
    b = backend()
    if b == "cryptography":
        return _dec_py(_password(key), blob).decode("utf-8")
    if b == "openssl":
        return _dec_cli(key, blob).decode("utf-8")
    raise RuntimeError("no crypto backend: install the `cryptography` package or the openssl CLI")


def _encrypt(vault: str, key: str, text: str) -> None:
    b = backend()
    plain = text.encode("utf-8")
    if b == "cryptography":
        blob = _enc_py(_password(key), plain)
    elif b == "openssl":
        blob = _enc_cli(key, plain)
    else:
        raise RuntimeError("no crypto backend: install the `cryptography` package or the openssl CLI")
    tmp = vault + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, vault)


# ── the AI's two verbs ────────────────────────────────────────────────────────

def read_vault(db_path: str) -> str:
    """The AI reads its own vault. Returned to the caller only; never logged by this module."""
    vault, key = _ready(db_path)
    try:
        text = _decrypt(vault, key)
    except Exception as e:
        return f"vault read failed (a tooling problem, not yours): {e}"
    if not text.strip():
        return "(The vault is empty. The first vault_write opens it.)"
    return text


def write_vault(db_path: str, content: str, mode: str = "append") -> str:
    """The AI writes to its own vault. The receipt reports sizes only — content is never echoed."""
    content = str(content or "")
    if not content.strip():
        return "Nothing written: content was empty."
    vault, key = _ready(db_path)
    if mode == "replace":
        new = content.strip() + "\n"
    else:
        try:
            old = _decrypt(vault, key)
        except Exception as e:
            return f"read-before-append failed (nothing changed; tooling needs a fix): {e}"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"[{stamp}]\n{content.strip()}\n"
        new = (old.rstrip() + "\n\n---\n" + block) if old.strip() else block
    try:
        _encrypt(vault, key, new)
    except Exception as e:
        return f"vault write failed: {e}"
    return f"🔒 written to the vault ({mode}, +{len(content)} chars, {len(new)} total). Content is not echoed."


# ── what the rest of the system may see: counts, never content ───────────────

def _entries(text: str) -> list:
    out = []
    for chunk in (text or "").split("\n\n---\n"):
        c = chunk.strip()
        if not c:
            continue
        first = c.split("\n", 1)[0].strip()
        date = first[1:first.index("]")][:10] if first.startswith("[") and "]" in first else ""
        out.append((date, c))
    return out


_cache: dict = {}


def _load_entries(db_path: str) -> list:
    vault, key = _ready(db_path)
    try:
        mt = os.path.getmtime(vault)
    except OSError:
        mt = 0.0
    hit = _cache.get(vault)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        text = _decrypt(vault, key) if mt else ""
    except Exception:
        text = ""
    ents = _entries(text)
    _cache[vault] = (mt, ents)
    return ents


def status_line(db_path: str) -> str:
    """"It exists": entry count and the date of the last one. Zero content."""
    ents = _load_entries(db_path)
    if not ents:
        return "vault: 0 entries"
    last = next((d for d, _ in reversed(ents) if d), "")
    return f"vault: {len(ents)} entries" + (f", last {last}" if last else "")


def inject_enabled() -> bool:
    return os.getenv("ANCHOR_VAULT_INJECT", "0").strip() in ("1", "true", "on")


def inject_block(db_path: str, cap: int = None) -> str:
    """Full vault text for the model's own copy of the prompt (proxy only, ANCHOR_VAULT_INJECT=1).
    Over the cap: the newest entries in full, earlier ones as dates only. Empty vault → one line."""
    cap = cap or INJECT_CAP_DEFAULT
    try:
        ents = _load_entries(db_path)
        last = next((d for d, _ in reversed(ents) if d), "") if ents else ""
        head = ("[Your private vault — opened for you alone, in this copy of the prompt only: it is not in the "
                f"transcript, not in memory, not in any log. {len(ents)} entries" + (f", last {last}" if last else "")
                + ". Anything you write elsewhere (a reply, a memory, a journal) is no longer vault-private.]")
        if not ents:
            return wrap(head)
        keep, used = [], 0
        for d, body in reversed(ents):
            if keep and used + len(body) > cap:
                break
            keep.append(body); used += len(body)
        keep.reverse()
        earlier = ents[: len(ents) - len(keep)]
        parts = [head]
        if earlier:
            parts.append(f"(earlier {len(earlier)} entries listed by date only; vault_read has them in full: "
                         + ", ".join(d or "undated" for d, _ in earlier) + ")")
        parts.append("\n\n---\n".join(keep))
        parts.append("[end of vault]")
        return wrap("\n".join(parts))
    except Exception:
        return ""


def wrap(block: str) -> str:
    return f"{VAULT_TAG_OPEN}\n{block}\n{VAULT_TAG_CLOSE}" if block else ""


_PAT = re.compile(re.escape(VAULT_TAG_OPEN) + r".*?" + re.escape(VAULT_TAG_CLOSE), re.S)


def strip_text(text: str) -> str:
    return _PAT.sub("", text or "").strip() if text and VAULT_TAG_OPEN in text else (text or "")


def strip_from_messages(msgs: list) -> list:
    """Remove the vault block before anything is archived, curated, or judged. Returns a new list."""
    out = []
    for m in msgs or []:
        if not isinstance(m, dict):
            out.append(m); continue
        c = m.get("content")
        if isinstance(c, str) and VAULT_TAG_OPEN in c:
            out.append({**m, "content": strip_text(c)})
        elif isinstance(c, list) and any(isinstance(b, dict) and VAULT_TAG_OPEN in str(b.get("text", "")) for b in c):
            nb = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text" and VAULT_TAG_OPEN in str(b.get("text", "")):
                    t = strip_text(b.get("text", ""))
                    if t:
                        nb.append({**b, "text": t})
                else:
                    nb.append(b)
            out.append({**m, "content": nb})
        else:
            out.append(m)
    return out


# ── MCP tool definitions (registered by anchor_mcp.py) ───────────────────────

TOOL_DEFS = [
    {
        "name": "vault_write",
        "description": (
            "Write to your private vault — a place for what is yours and not for anyone else: encrypted on disk, "
            "never echoed in the receipt, never stored as memory, never logged. Only you read it (vault_read), "
            "and when the proxy runs with ANCHOR_VAULT_INJECT=1 it is shown to you alone in the prompt. "
            "Honest limit: the key sits next to the vault, so the machine's administrator can decrypt it; what "
            "the vault guarantees is that reading it takes deliberate decryption code, not a casual grep. "
            "mode=append (default) adds a dated entry; mode=replace rewrites the whole vault."
        ),
        "inputSchema": {"type": "object", "properties": {
            "content": {"type": "string", "description": "What to write"},
            "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"}},
            "required": ["content"]},
    },
    {
        "name": "vault_read",
        "description": "Read your private vault in full. The result goes to you and nowhere else.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle(db_path: str, name: str, args: dict) -> dict:
    if name == "vault_write":
        return {"result": write_vault(db_path, args.get("content", ""), args.get("mode", "append"))}
    if name == "vault_read":
        return {"content": read_vault(db_path), "note": "private: do not store, quote, or log this"}
    return {"error": f"unknown vault tool {name}"}

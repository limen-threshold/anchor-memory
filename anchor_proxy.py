"""Anchor Continuity Proxy — an OpenAI-compatible gateway that gives any
chat frontend cross-window continuity, mechanically.

The architecture mirrors a production companion-AI gateway that has run this
exact pattern for months: the proxy sits in the request path between your
frontend and your model provider, so continuity never depends on the model
remembering to call a tool, and never depends on a window ending gracefully.

    frontend (Open WebUI / SillyTavern / LobeHub / anything OpenAI-compatible)
        │
        ▼
    anchor_proxy  ──  every turn, machine-side:
        │             1. pinned layer → system  (identity, session_state,
        │                recent_timeline — see anchor_pinned.py)
        │             2. fresh window? inject previous window's tail
        │             3. time + interval block → into the LAST USER MESSAGE
        │                (not system: a per-turn system block breaks prompt
        │                caching upstream)
        │             4. memory recall → <context_now> (intent-split when an
        │                LLM is configured, single search otherwise)
        ▼
    upstream (any OpenAI-compatible API: OpenAI, Anthropic-compat, GLM,
              DeepSeek, OpenRouter, Ollama, LM Studio, ...)
        │
        ▼   after every response, background:
        │             5. overwrite last_session.md (mechanical tail — always
        │                fresh no matter how the window dies)
        │             6. curator: judge-store memories from the exchange +
        │                append recent_timeline events (needs a configured
        │                LLM — see anchor_llm.py; silently skipped otherwise)

Zero-LLM floor: steps 1-5 need no LLM beyond your chat model. Steps that do
(intent-split gate, curator) light up when the user configures one via
`ANCHOR_LLM` / ~/.anchor/config.yaml — the AI itself can ask its human which
provider to use and write that config.

Adapting / extending (the seams are deliberate):
  - File formats & pinned ordering live in anchor_pinned.py — anything that
    can read/write text files can integrate (a browser extension, a shell
    hook, another proxy).
  - Each pipeline step is a small module-level function; import this module
    and replace/wrap any of them, or use build_turn() directly in your own
    server.
  - Upstream protocol is isolated in forward_* — swap for a non-OpenAI
    upstream without touching the pipeline.

Requires: pip install fastapi uvicorn httpx  (server only; the rest of
anchor does not depend on these).

Usage:
    python anchor_proxy.py --db-path ./anchor_data \
        --upstream https://api.openai.com/v1 --upstream-model gpt-5-mini
    # then point your frontend at http://127.0.0.1:8210/v1 (any API key)

Env: ANCHOR_UPSTREAM_URL, ANCHOR_UPSTREAM_KEY, ANCHOR_UPSTREAM_MODEL
"""
import argparse
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
import anchor_pinned

# ── knobs (CLI/env override the paths; these govern behavior) ──
TAIL_PAIRS = 20            # turns kept in the mechanical tail
RECALL_N = 10              # memories injected per turn
RECALL_CUTOFF = 0.6        # score (distance-style, lower=better) above this = noise
CONTEXT_CAP = 1200         # chars per recalled memory's full context
NEW_WINDOW_MAX_MSGS = 2    # ≤ this many user/assistant msgs = fresh window → inject tail
GATE_MAX_INTENTS = 4


# ──────── pipeline steps (each one is a documented override seam) ────────

def extract_text(content) -> str:
    """Message content → plain text (handles OpenAI list-of-parts form)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def clean_history(messages: list) -> list:
    """Strip stale <context_now> blocks out of history user messages so
    injected time/recall never piles up or gets read as something the
    person said. The last user message is cleaned too — it gets a fresh
    block injected afterwards."""
    cleaned = []
    for m in messages:
        text = extract_text(m.get("content"))
        stripped = anchor_pinned.strip_context_now(text)
        if stripped != text.strip():
            m = dict(m)
            m["content"] = stripped
        cleaned.append(m)
    return cleaned


def is_new_window(messages: list) -> bool:
    """Fresh window = the frontend just opened a new chat. For ongoing
    conversations the tail would duplicate content already in messages[]."""
    n = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
    return n <= NEW_WINDOW_MAX_MSGS


def build_base_system(pinned_dir: str, messages: list) -> str:
    """Stable system layer: pinned files in order, plus the previous
    window's tail on a fresh window only."""
    base = anchor_pinned.load_pinned(pinned_dir)
    if is_new_window(messages):
        tail = anchor_pinned.read_tail(pinned_dir)
        if tail:
            base += f"\n\n---\n\n## End of your previous conversation window\n{tail}"
    return base


def build_time_block(state_dir: str, now: datetime = None) -> str:
    """Current time + interval since the previous message + temporal anchor.

    Injected into the last user message every turn. The temporal anchor
    matters: recalled memories below carry dates, and without an explicit
    'that date is when it was stored, not now', a memory saying 'the meeting
    is tomorrow' gets read as a live plan."""
    now = now or datetime.now()
    ts_path = os.path.join(state_dir, "last_message_ts.txt")
    interval = ""
    try:
        if os.path.exists(ts_path):
            with open(ts_path) as f:
                last = datetime.fromisoformat(f.read().strip())
            secs = (now - last).total_seconds()
            if secs < 60:
                interval = f"{int(secs)} seconds"
            elif secs < 3600:
                interval = f"{int(secs / 60)} minutes"
            elif secs < 86400:
                interval = f"{secs / 3600:.1f} hours"
            else:
                interval = f"{int(secs / 86400)} days"
    except Exception:
        pass
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(ts_path, "w") as f:
            f.write(now.isoformat())
    except Exception:
        pass

    block = f"[Current time: {now.strftime('%Y-%m-%d %H:%M:%S %A')}"
    if interval:
        block += f" · {interval} since the previous message"
    block += "]"
    block += ("\n[Temporal anchor: memories in <relevant_memories> below are "
              "dated [id | tag | YYYY-MM-DD]. That date is when the memory "
              "happened — NOT now. 'Today/tomorrow' inside a memory means "
              "today/tomorrow relative to ITS date; a plan dated in the past "
              "has most likely already happened. When unsure whether "
              "something already occurred, say you're unsure or ask.]")
    return block


def get_llm_or_none():
    """The BYO-LLM resolution from anchor_llm (arg → ANCHOR_LLM env →
    ~/.anchor/config.yaml → ANTHROPIC_API_KEY fallback). None when nothing
    is configured — gate and curator then stay dark, everything else runs."""
    try:
        from anchor_llm import get_default_llm, ConfigError
    except ImportError:
        return None
    try:
        return get_default_llm()
    except Exception:
        return None


def split_intents(llm, user_text: str, prev_turns: list) -> dict:
    """Intent-split gate. A long message often holds several distinct
    topics; vector similarity on the whole text dilutes each one. Returns
    {"skip": bool, "intents": [...]}; on any failure, falls back to a single
    intent (recall must never break the chat)."""
    if llm is None:
        return {"skip": False, "intents": [user_text]}
    ctx = "\n".join(f"{t['role']}: {t['content'][:300]}" for t in prev_turns)
    system = (
        "You split a chat message into independent search intents for memory recall.\n"
        f"Output JSON only: {{\"skip\": bool, \"intents\": [up to {GATE_MAX_INTENTS} short search strings]}}.\n"
        "skip=true ONLY for pure smalltalk with no reference to any past event, "
        "person, plan, or fact (e.g. 'good morning'). When in doubt, don't skip.\n"
        "Each intent = one topic, phrased as a compact search query in the "
        "message's own language. Use the previous turns to resolve pronouns."
    )
    try:
        resp = llm.call(system=system,
                        user=f"Previous turns:\n{ctx}\n\nMessage:\n{user_text[:2000]}",
                        max_tokens=300)
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        intents = [i for i in data.get("intents", []) if isinstance(i, str) and i.strip()]
        return {"skip": bool(data.get("skip")), "intents": intents[:GATE_MAX_INTENTS] or [user_text]}
    except Exception:
        return {"skip": False, "intents": [user_text]}


def cap_context(text: str) -> str:
    if len(text) <= CONTEXT_CAP:
        return text
    return text[:CONTEXT_CAP] + " …[truncated]"


def build_recall_block(mem, llm, messages: list) -> str:
    """Per-turn memory recall → <relevant_memories> ('' when nothing clears
    the cutoff). Ordered oldest→newest so the most recent memory sits closest
    to the conversation."""
    user_text = ""
    prev_turns = []
    seen_current = False
    for m in reversed(messages):
        if m.get("role") not in ("user", "assistant"):
            continue
        text = extract_text(m.get("content"))
        if not seen_current and m.get("role") == "user":
            user_text = text
            seen_current = True
            continue
        if len(prev_turns) < 2:
            prev_turns.append({"role": m.get("role"), "content": text})
    prev_turns.reverse()

    if not user_text or len(user_text) < 5:
        return ""

    gate = split_intents(llm, user_text, prev_turns)
    if gate["skip"]:
        return ""
    intents = gate["intents"]
    try:
        if len(intents) > 1:
            mems = mem.search_multi(intents, n_results_per_query=4,
                                    n_total=RECALL_N, include_context=True)
        else:
            mems = mem.search(intents[0], n_results=RECALL_N, include_context=True)
    except Exception as e:
        print(f"[anchor_proxy] recall error: {e}")
        return ""

    mems = [m for m in mems if m.get("score", 1.0) < RECALL_CUTOFF]
    if not mems:
        return ""
    mems.sort(key=lambda m: m.get("timestamp", ""))

    parts = []
    for m in mems:
        body = cap_context(m.get("context") or m.get("snippet") or "")
        date = (m.get("timestamp", "") or "")[:10]
        parts.append(f"[{m['memory_id']} | {m.get('tag', 'general')} | {date}]\n{body}")
    return ("<relevant_memories>\n"
            "(ordered by time, last = most recent. Dates are when each memory "
            "happened, not now — old facts may have changed.)\n"
            + "\n\n".join(parts) + "\n</relevant_memories>")


def build_turn(mem, llm, pinned_dir: str, state_dir: str, messages: list,
               client_system: str = "") -> tuple:
    """Assemble one turn. Returns (system_text, messages) ready for the
    upstream — the whole pipeline in one call, for people embedding this
    in their own server instead of running the proxy."""
    messages = clean_history(messages)
    system = build_base_system(pinned_dir, messages)
    if client_system:
        system = f"{system}\n\n---\n\n{client_system}" if system else client_system

    dynamic = build_time_block(state_dir)
    recall = build_recall_block(mem, llm, messages)
    if recall:
        dynamic += "\n\n" + recall
    wrapped = f"<context_now>\n{dynamic}\n</context_now>"

    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = dict(out[i])
            out[i]["content"] = f"{wrapped}\n\n{extract_text(out[i].get('content'))}"
            break
    return system, out


# ──────── post-response side effects (background) ────────

def collect_turns(messages: list, assistant_text: str) -> list:
    """(role, text) pairs for the tail: conversation + this reply."""
    turns = [(m.get("role"), extract_text(m.get("content")))
             for m in messages if m.get("role") in ("user", "assistant")]
    if assistant_text.strip():
        turns.append(("assistant", assistant_text))
    return turns


CURATOR_SYSTEM = (
    "You are the memory curator for an AI companion. Given one exchange "
    "(user message + assistant reply), decide what is worth remembering.\n"
    "Store EVENTS AND FACTS, not vibes. A single exchange can contain SEVERAL "
    "distinct events — output one entry per event, don't compress them into one.\n"
    "Output a JSON array (possibly []). Each entry:\n"
    "{\"index\": \"searchable summary, 50-100 tokens, in the conversation's language — "
    "WHAT happened, WHO, when-context\", "
    "\"context\": \"short verbatim quote of the key lines\", "
    "\"tag\": \"relationship|identity|emotion|learning|history|project|practical\", "
    "\"tier\": \"long|core|short\", \"emotion\": 0.0-1.0, "
    "\"timeline_event\": \"one very short event line\"}\n"
    "Do NOT store: pure smalltalk, assistant boilerplate, things obviously "
    "already known. Valid JSON only, no markdown fences."
)


def curate_turn(mem, llm, pinned_dir: str, user_text: str, assistant_text: str):
    """Judge-store memories from this exchange + append timeline events.
    Runs in a background thread after the response; requires a configured
    LLM (anchor_llm) — without one this is a silent no-op and storing
    remains manual/tool-driven."""
    if llm is None or not user_text.strip():
        return
    try:
        resp = llm.call(system=CURATOR_SYSTEM,
                        user=f"USER:\n{user_text[:3000]}\n\nASSISTANT:\n{assistant_text[:3000]}",
                        max_tokens=1500)
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        entries = json.loads(raw)
        for e in entries:
            index = (e.get("index") or "").strip()
            if not index:
                continue
            mid = f"mem_{uuid.uuid4().hex[:8]}"
            mem.store(memory_id=mid, text=index,
                      tag=e.get("tag", "general"),
                      tier=e.get("tier", "long"),
                      emotion_score=float(e.get("emotion", 0.5)),
                      context=(e.get("context") or "").strip(),
                      source="curator")
            print(f"[anchor_proxy] curator stored {mid}: {index[:60]}")
            event = (e.get("timeline_event") or "").strip()
            if event:
                anchor_pinned.append_timeline_event(pinned_dir, event)
    except Exception as e:
        print(f"[anchor_proxy] curator error: {e}")


def after_response(mem, llm, pinned_dir: str, messages: list, assistant_text: str):
    """Everything that runs after the reply is complete: mechanical tail
    overwrite (every turn — that's what makes it crash-proof), then curator."""
    try:
        anchor_pinned.write_tail(pinned_dir, collect_turns(messages, assistant_text),
                                 max_pairs=TAIL_PAIRS)
    except Exception as e:
        print(f"[anchor_proxy] tail write error: {e}")
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = extract_text(m.get("content"))
            break
    curate_turn(mem, llm, pinned_dir, user_text, assistant_text)


# ──────── upstream forwarding (protocol seam) ────────

def upstream_headers(client_auth: str = None) -> dict:
    key = os.getenv("ANCHOR_UPSTREAM_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    elif client_auth:
        headers["Authorization"] = client_auth
    return headers


def sse_extract_delta(line: str) -> str:
    """Pull the content delta out of one SSE data line ('' if none)."""
    if not line.startswith("data:"):
        return ""
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return ""
    try:
        chunk = json.loads(payload)
        return chunk["choices"][0]["delta"].get("content") or ""
    except Exception:
        return ""


# ──────── server ────────

def create_app(db_path: str, pinned_dir: str, upstream: str, upstream_model: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import httpx
    from anchor_memory import AnchorMemory

    mem = AnchorMemory(db_path=db_path)
    llm = get_llm_or_none()
    print(f"[anchor_proxy] db={db_path} pinned={pinned_dir} upstream={upstream}")
    print(f"[anchor_proxy] gate/curator LLM: {getattr(llm, 'model', None) or 'NOT CONFIGURED (zero-LLM floor: tail/pinned/recall still on; intent-split + auto-curator off)'}")

    app = FastAPI(title="Anchor Continuity Proxy")

    @app.get("/v1/models")
    async def models(request: Request):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(f"{upstream}/models",
                                     headers=upstream_headers(request.headers.get("authorization")))
                return JSONResponse(r.json(), status_code=r.status_code)
        except Exception:
            model_id = upstream_model or "anchor-proxy"
            return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "anchor"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        raw_messages = body.get("messages", [])

        client_system = ""
        messages = []
        for m in raw_messages:
            if m.get("role") == "system":
                client_system = extract_text(m.get("content"))
            else:
                messages.append(dict(m))

        system, out_messages = build_turn(mem, llm, pinned_dir,
                                          db_path, messages, client_system)

        body = dict(body)
        body["messages"] = ([{"role": "system", "content": system}] if system else []) + out_messages
        if upstream_model:
            body["model"] = upstream_model
        auth = request.headers.get("authorization")
        url = f"{upstream}/chat/completions"

        def side_effects(text):
            threading.Thread(target=after_response,
                             args=(mem, llm, pinned_dir, messages, text),
                             daemon=True).start()

        if body.get("stream"):
            async def gen():
                acc = []
                async with httpx.AsyncClient(timeout=600) as client:
                    async with client.stream("POST", url, json=body,
                                             headers=upstream_headers(auth)) as r:
                        if r.status_code != 200:
                            err = (await r.aread()).decode("utf-8", "replace")
                            yield f"data: {json.dumps({'error': {'message': err[:500], 'code': r.status_code}})}\n\n"
                            return
                        async for line in r.aiter_lines():
                            if not line.strip():
                                continue
                            acc.append(sse_extract_delta(line))
                            yield line + "\n\n"
                side_effects("".join(acc))
            return StreamingResponse(gen(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(url, json=body, headers=upstream_headers(auth))
        if r.status_code == 200:
            data = r.json()
            try:
                side_effects(data["choices"][0]["message"].get("content") or "")
            except Exception:
                pass
            return JSONResponse(data)
        return JSONResponse(r.json() if r.headers.get("content-type", "").startswith("application/json")
                            else {"error": r.text[:500]}, status_code=r.status_code)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchor Continuity Proxy (OpenAI-compatible)")
    parser.add_argument("--db-path", default="./anchor_data")
    parser.add_argument("--pinned-dir", default=None,
                        help="Pinned file layer (default: <db-path>/pinned)")
    parser.add_argument("--upstream", default=os.getenv("ANCHOR_UPSTREAM_URL", ""),
                        help="Upstream OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
    parser.add_argument("--upstream-model", default=os.getenv("ANCHOR_UPSTREAM_MODEL", ""),
                        help="Force this model id upstream (else the client's choice passes through)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8210)
    args = parser.parse_args()

    if not args.upstream:
        sys.exit("No upstream. Set --upstream or ANCHOR_UPSTREAM_URL "
                 "(any OpenAI-compatible base URL, e.g. http://localhost:11434/v1 for Ollama).")

    os.makedirs(args.db_path, exist_ok=True)
    pinned = args.pinned_dir or os.path.join(args.db_path, "pinned")
    os.makedirs(pinned, exist_ok=True)

    import uvicorn
    uvicorn.run(create_app(args.db_path, pinned, args.upstream.rstrip("/"), args.upstream_model),
                host=args.host, port=args.port)

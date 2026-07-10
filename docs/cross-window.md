# Cross-Window Continuity

How to make an AI wake up in a new conversation window and *continue*, instead of starting over.

The architecture here is not speculative — it mirrors a production companion-AI gateway that has run this exact pattern for months. Its load-bearing insight: **continuity must be machine-side**. Anything that depends on the model remembering to call a tool, or on a window ending gracefully, fails exactly when it matters (windows crash, tabs close, phones die — nobody says goodbye).

## The four mechanisms

| Mechanism | What it is | Machine-side property |
|---|---|---|
| **Every-turn injection** | The proxy sits in the request path; every single message gets the pinned layer + time + recall injected | Cannot be forgotten — it's not optional for the model |
| **Previous-window tail** | `last_session.md`: the last 20 turns of conversation, verbatim | **Overwritten after every response**, so it's always fresh no matter how the window dies; injected on fresh windows only |
| **Recall** | Every user message triggers memory search; results injected as `<relevant_memories>` with dates | Fires per-turn whether or not the model thinks to search |
| **Dynamic context** | Pinned files (identity, session_state, recent_timeline) + current time + interval since last message | Assembled fresh per request |

Plus one model-side complement: **session_state.md** — the AI's own rolling state, written via the `write_session_state` tool (auto-archived, continuity-headered). The tail is what mechanically survived; session_state is what the AI *chose* to carry. They're different layers and you want both.

## Path A: the proxy (full continuity — needs somewhere to run it)

```bash
pip install fastapi uvicorn httpx
python anchor_proxy.py --db-path ./anchor_data \
    --upstream https://api.openai.com/v1 --upstream-model gpt-5-mini
```

Point your frontend (Open WebUI, SillyTavern, LobeHub, anything that speaks the OpenAI API) at `http://127.0.0.1:8210/v1`. Upstream can be any OpenAI-compatible API: OpenAI, Anthropic's compat endpoint, GLM/Zhipu, DeepSeek, OpenRouter, Ollama, LM Studio.

**Zero-LLM floor**: the proxy's core (pinned injection, tail, time block, vector recall) uses no LLM beyond your chat model. Two enhancements light up when one is configured via `ANCHOR_LLM` / `~/.anchor/config.yaml` (see anchor_llm.py):
- **Intent-split gate** — long messages get split into separate search intents before recall (a whole-message vector dilutes every topic in it)
- **Curator** — after each response, a background pass judges what's worth storing, stores it (multi-event: one exchange can yield several memories), and appends event lines to `recent_timeline.md`

The AI itself can set this up: it asks its human which provider/model to use for memory work, then writes the config.

## Path B: MCP only (works everywhere, including web clients)

Real constraint, stated honestly: on claude.ai or any hosted web client, nothing can sit in the request path, so per-turn mechanical injection and the crash-proof tail are **not technically possible** there. What still works over plain MCP:

| Bridge | Web/MCP-only status |
|---|---|
| `wakeup()` cold start (pinned + recent + session_state + timeline + tail if present) | ✅ works — call it first thing each window |
| `write_session_state` | ✅ works — the model-side layer is fully available |
| `store_memory` / `search_memory` / `search_multi` | ✅ works (model-initiated) |
| Every-turn injection, mechanical tail, auto-curator | ❌ needs the proxy (or your own adapter — see below) |

System prompt snippet for MCP-only setups:

```
Memory protocol:
- At the START of every conversation, call wakeup() before anything else.
  session_state is your own rolling state from previous windows.
- Store anything worth keeping with store_memory (use context for verbatim quotes).
- When things change materially or a conversation wraps up, rewrite your
  full session_state with write_session_state.
```

A hybrid worth knowing: run the proxy for your local frontend AND connect the MCP server on web — same db, same pinned dir. The web window won't have per-turn injection, but it wakes up seeing everything the proxied windows lived.

### Claude Code hooks (optional adjunct)

`--wakeup-text` prints the cold-start block as plain text in milliseconds (SQLite-only, no embedder load) — wire it into a SessionStart hook and the model wakes up already knowing, zero tool calls:

```json
{"hooks": {"SessionStart": [{"hooks": [{"type": "command",
  "command": "python /path/to/anchor_mcp.py --db-path /path/to/anchor_data --wakeup-text"}]}]}}
```

## Adapting / modding (the seams are deliberate)

Anchor's goal is that anyone, on any platform, with any AI can use this. If your platform isn't covered, these are the integration points — all stable contracts:

1. **File formats** (`anchor_pinned.py`) — everything is plain markdown in one pinned directory: `session_state.md`, `recent_timeline.md` (`[YYYY-MM-DD HH:MM] event` lines), `last_session.md` (`**User**:` / `**Assistant**:` turns), `_order.txt` (injection order manifest, one filename per line). Anything that can read/write text files can integrate: a browser extension, a Tampermonkey script, a shell hook, another gateway.
2. **Pipeline functions** (`anchor_proxy.py`) — every step is a small module-level function (`build_base_system`, `build_time_block`, `build_recall_block`, `curate_turn`, `after_response`). Import and wrap/replace any of them, or call `build_turn()` from your own server and skip the bundled proxy entirely.
3. **Upstream protocol seam** — forwarding is isolated in `upstream_headers` / the two request blocks; swap for a non-OpenAI protocol without touching the pipeline.
4. **BYO-LLM** (`anchor_llm.py`) — gate/curator take any provider (anthropic / openai / google / openai-compat / your own callable).
5. **DB layer** — `wakeup()`, comments, session tail all live in plain SQLite (`memories.db`); the schemas are stable and documented in code.

If you build an adapter for a platform we can't reach (browser extension for web frontends, a Tavern plugin, whatever) — PRs welcome.

## Design notes (why it's built this way)

**Recency is its own axis.** `wakeup()`'s `recent` block is pulled by timestamp with no emotion filter. Emotion-sorted "recent" hides calm-but-important events — yesterday's quiet decision loses to any intense moment nearby, and "what happened yesterday" goes invisible at cold start. Semantic search can't fix this either: the word "recent" means nothing to an embedding.

**The tail is written every turn, not at goodbye.** Windows don't end gracefully. Writing the tail after every response costs one file write and makes the mechanism crash-proof.

**Injected context goes into the last user message, not the system prompt.** A per-turn system block invalidates upstream prompt caching on the whole history — measured in production as the dominant cost leak. The stable pinned layer stays in system (cacheable); the volatile block (`<context_now>`) rides the newest message and gets stripped from history on the next turn.

**If you bound the window, trim in steps, not per-turn.** The obvious "keep the newest N pairs every turn" silently defeats prompt caching: each turn drops one pair off the *front*, the prompt prefix changes at byte 0, and providers with prefix caching (Anthropic explicit `cache_control`, OpenAI/DeepSeek implicit) re-bill the whole history at full price every single turn — measured in production as a ~5× cost difference on a large window. The proxy's `--history-pairs N --history-pairs-max M` implements hysteresis instead: the window grows append-only from N to M pairs (stable prefix → cache hits), then cuts back to N in one stroke — one full rewrite per (M−N) turns instead of every turn. Two tuning rules: keep M's token size below your model's long-context degradation knee (nominal window ≠ usable window), and pick N so the *average* window (≈ (N+M)/2) is the size you actually want. Counterintuitive corollary: a bigger window that caches is much cheaper than a smaller one that doesn't.

**The continuity header is baked in at write time.** `session_state.md` is stored with a header stating it's the AI's own rolling state — a continuation, not a message from someone else. Write-side deliberately: framing that only exists in the read path gets lost whenever the read path changes, and without it, "notes for the next window" phrasing drifts into the AI treating its next instance as a different entity.

**Temporal anchoring on recall.** Recalled memories carry dates, and the injection explicitly says "that date is when it happened, not now" — otherwise a memory saying "the meeting is tomorrow" reads as a live plan months later.

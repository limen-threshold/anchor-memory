# Cross-Window Continuity

How to make an AI wake up in a new conversation window and *continue*, instead of starting over.

Anchor gives you three bridges across the window boundary:

1. **Recall** — the memory graph itself. Anything stored is searchable from any future window.
2. **Session tail (handoff)** — the closing state of the previous window: what was happening, what's unfinished, what mood. Written with `write_handoff` at the end of a window, surfaced automatically by `wakeup()` at the start of the next.
3. **Cold-start snapshot** — `wakeup()`: one call that returns the last handoff + pinned (identity-level) memories + the most recent memories + recent high-emotion memories + a couple of random old ones + unread comments.

The rest of this doc is how to wire them into your client.

## The window lifecycle

```
window N                              window N+1
────────────────────────────────      ────────────────────────────────
wakeup()          ← cold start        wakeup()          ← sees N's handoff
  ...conversation...                    ...conversation...
store_memory()    ← as things         store_memory()
                    worth keeping
                    happen
write_handoff()   ← before close      write_handoff()
consolidate()     ← optional Hebbian
                    pass on topics
```

## Path A: MCP tools (works everywhere)

If your client speaks MCP (Claude Code, claude.ai, LobeHub, SillyTavern, ...), the tools are already there. The problem is *discipline*: MCP cannot force the model to call `wakeup` first or `write_handoff` last. You enforce it in the system prompt. A snippet that works:

```
Memory protocol:
- At the START of every conversation, before anything else, call wakeup().
  Read the last_handoff first — it is your own note from the previous
  window, not a message from someone else.
- During the conversation, store anything worth keeping with store_memory.
- When the conversation is clearly wrapping up (user says goodbye, task
  is done), call write_handoff with: what happened, what's unfinished,
  any decisions made, and the emotional temperature of the conversation.
```

## Path B: hooks (deterministic, zero reliance on model discipline)

For clients with lifecycle hooks, inject the wakeup block mechanically instead of hoping the model calls the tool. Both CLI modes below use a SQLite-only fast path — no embedding model load, so they run in milliseconds.

```bash
# print the cold-start block as plain text (for session-start hooks)
python anchor_mcp.py --db-path ./anchor_data --wakeup-text

# write a handoff from a script (for session-end hooks)
python anchor_mcp.py --db-path ./anchor_data --write-handoff "Half-done: ..."
```

### Claude Code example

`.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/anchor_mcp.py --db-path /path/to/anchor_data --wakeup-text"
          }
        ]
      }
    ]
  }
}
```

The hook's stdout lands in the model's context at session start — the model wakes up already knowing, no tool call needed. Keep the MCP server connected as well for `store_memory` / `write_handoff` during the session.

## Design notes (why it's built this way)

**Recency is its own axis.** An earlier version of `wakeup()` had only an emotion-sorted "recent" block. That hides calm-but-important events: yesterday's quiet decision (emotion 0.4) loses to any intense moment in the same 3-day window, and "what happened yesterday" becomes invisible at cold start. The same failure happens if you try semantic search for "recent events" — the word "recent" means nothing to an embedding, and old general memories outscore yesterday's specifics. So `wakeup()` now has a `recent` block pulled by timestamp, no emotion filter, deduplicated against the high-emotion block.

**The continuity header is baked in at write time.** Every handoff is stored with a header stating that it is the AI's own note from its previous window — a continuation, not a message from someone else. This lives on the *write* side deliberately: framing that only exists in the read path gets lost whenever the read path changes (a new client, a raw SQL dump, a different injection route), and without it, "notes for the next window" phrasing drifts into the AI treating its next instance as a different entity ("tell him that..."). Write handoffs in first person, as notes to yourself.

**Handoffs are not comments.** `leave_comment` attaches to a specific memory and is good for annotating *that memory* across windows. A session tail belongs to no single memory — it's the state of the whole window. That's why it's a separate table and a separate tool.

**wakeup is a snapshot, not a search.** It's deliberately dumb: pinned + recent + emotional + random + unread, by fixed rules. Anything smarter (semantic relevance to the new conversation's opening message) belongs to recall during the conversation, not to cold start — at cold start there is nothing to be relevant *to* yet.

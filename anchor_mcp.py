"""
Anchor Memory System — MCP Server

Exposes Anchor Memory as an MCP (Model Context Protocol) server.
Any MCP-compatible client can connect and use graph-structured memory with
Hebbian learning. Two transports, one set of tools:

    stdio  (default) — local hosts that spawn a subprocess: Claude Code,
                       Claude Desktop, LobeHub, SillyTavern …
    --http           — hosted clients that can only reach a URL: claude.ai
                       custom connectors, ChatGPT, any remote MCP client.
                       Streamable HTTP at /mcp (+ legacy SSE at /sse).

Usage:
    python anchor_mcp.py [--db-path ./my_memory]                      # stdio
    python anchor_mcp.py --http [--port 3333] [--token SECRET]        # HTTP
"""

import json
import sys
import os
import uuid
import argparse
from datetime import datetime

# Windows fix: force UTF-8 on stdin/stdout to prevent GBK encoding issues
# (Windows cmd defaults to GBK; mcp_proxy communicates in UTF-8)
if sys.platform == "win32" or (hasattr(sys.stdout, 'buffer') and sys.stdout.encoding and sys.stdout.encoding.upper() != 'UTF-8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import anchor_pinned
import anchor_invite


def create_server(db_path: str = "./anchor_data", pinned_dir: str = None):
    """Create MCP server with Anchor Memory tools.

    pinned_dir: optional directory for the always-injected file layer
    (session_state.md / recent_timeline.md / last_session.md — see
    anchor_pinned.py). When set, wakeup() returns those files too and the
    write_session_state tool becomes functional. Defaults to <db_path>/pinned.
    """

    # Imported here, not at module top: the CLI hook modes (--wakeup-text)
    # must stay on the SQLite-only fast path — importing anchor_memory pulls
    # in sentence_transformers/chromadb (seconds).
    from anchor_memory import AnchorMemory

    mem = AnchorMemory(db_path=db_path)
    pinned_dir = pinned_dir or os.path.join(db_path, "pinned")

    # MCP tool definitions
    TOOLS = [
        {
            "name": "store_memory",
            "description": "Store a new memory. Memories are nodes in a graph — they can be connected to other memories and carry emotional weight.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory content. Preserve narrative and original words — memories should read like flashbacks, not file entries."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Category: relationship, identity, emotion, learning, history, project, practical, research, or any custom tag.",
                        "default": "general"
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["core", "long", "short"],
                        "description": "core = permanent. long = kept indefinitely. short = decays after 14 days.",
                        "default": "long"
                    },
                    "emotion_score": {
                        "type": "number",
                        "description": "0.0 (neutral) to 1.0 (intense). How emotionally heavy is this memory? Most are 0.3-0.6. Only truly intense moments get above 0.8.",
                        "default": 0.5
                    },
                    "connect_to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of memory_ids to explicitly connect this memory to."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional full original text (two-layer storage: text = searchable summary that gets embedded, context = verbatim source). Returned by searches with include_context."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search memories. Returns results ranked by semantic similarity, citation count, and emotion score. Triggers Hebbian learning — memories retrieved together form connections.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Max results.",
                        "default": 5
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag."
                    },
                    "associate": {
                        "type": "boolean",
                        "description": "Follow graph edges to find related memories.",
                        "default": True
                    },
                    "hebbian": {
                        "type": "boolean",
                        "description": "Strengthen connections between co-retrieved memories.",
                        "default": True
                    },
                    "debug": {
                        "type": "boolean",
                        "description": "Include ranking internals on each result — raw_distance, citation_boost, emotion_boost, final_score, source ('vector'|'keyword'|'associative'), and edge_weight for associative hops. Use to audit why a given result landed at its rank.",
                        "default": False
                    },
                    "exclude_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to drop before result slots are allocated (they never consume a slot). Use for material that is stored but must not compete in this context, e.g. a raw-transcript tag."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_multi",
            "description": "Run multiple independent searches and merge results. Use when a single user message contains several distinct topics — vector similarity on the whole message dilutes any one topic, so you pre-split the message into intent strings and pass them here. Each intent searches separately at depth, then results are merged and dedup'd by memory_id. Hebbian co-activation fires across the merged set, so memories surfaced by different intents in the same message form edges with each other.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of intent strings to search. Example: for the message '七月去欧洲要不要换4K拍摄', pass ['七月欧洲行', '4K拍摄设置']."
                    },
                    "n_results_per_query": {
                        "type": "integer",
                        "description": "Top-k pulled from each individual search.",
                        "default": 5
                    },
                    "n_total": {
                        "type": "integer",
                        "description": "Final cap after merge. Default n_results_per_query * len(queries)."
                    },
                    "tag": {"type": "string"},
                    "associate": {"type": "boolean", "default": True},
                    "hebbian": {"type": "boolean", "default": True},
                    "include_context": {"type": "boolean", "default": False},
                    "exclude_tags": {"type": "array", "items": {"type": "string"},
                                     "description": "Tags dropped before slots are allocated (see search_memory)."}
                },
                "required": ["queries"]
            }
        },
        {
            "name": "connect_memories",
            "description": "Explicitly connect two memories. Creates a weighted bidirectional edge (synapse). Use for manual entanglement — connecting memories you know are related.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "weight": {
                        "type": "number",
                        "description": "Connection strength. Hebbian auto-connections are 0.2. Manual entanglement is typically 1.5-3.0. Max 10.0.",
                        "default": 2.0
                    }
                },
                "required": ["source_id", "target_id"]
            }
        },
        {
            "name": "get_neighbors",
            "description": "Get memories connected to a given memory via graph edges. Returns neighbors sorted by edge weight.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "min_weight": {
                        "type": "number",
                        "description": "Minimum edge weight to include.",
                        "default": 0.5
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "delete_memory",
            "description": "Delete a memory and all its edges.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "dream_pass",
            "description": "Run memory consolidation — like sleep for the brain. Decays old memories, prunes weak connections, discovers new ones, equilibrates emotion scores. Run daily.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "set_emotion",
            "description": "Set the emotion score of an existing memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "score": {
                        "type": "number",
                        "description": "0.0 (neutral) to 1.0 (intense)."
                    }
                },
                "required": ["memory_id", "score"]
            }
        },
        {
            "name": "set_tier",
            "description": "Change the tier of an existing memory (core/long/short).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "tier": {
                        "type": "string",
                        "enum": ["core", "long", "short"]
                    }
                },
                "required": ["memory_id", "tier"]
            }
        },
        {
            "name": "graph_stats",
            "description": "Get overview stats: total memories, edges, tag distribution, tier distribution, top connected nodes.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "annotate_memory",
            "description": "Add an annotation to a memory. Annotations are append-only — they record how understanding of a memory evolves over time. Searchable. Original memory text is never changed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "The memory to annotate."},
                    "text": {"type": "string", "description": "The annotation text. E.g. '4/18: realized this was about X, not Y.'"}
                },
                "required": ["memory_id", "text"]
            }
        },
        {
            "name": "get_annotations",
            "description": "Get all annotations for a memory, oldest first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "The memory to get annotations for."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "consolidate",
            "description": "Passive Hebbian update — after a conversation, pass key topics to build connections between memories that co-occurred but weren't explicitly searched. Zero LLM token cost. Call at the end of a conversation or session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "conversation_text": {
                        "type": "string",
                        "description": "Key topics from the conversation. E.g. 'talked about her friend Lily, yesterday's dinner (ramen), the cockroach incident, her work project'"
                    }
                },
                "required": ["conversation_text"]
            }
        },
        {
            "name": "store_visual",
            "description": "Store a visual observation as a memory with CLIP embedding. For Anchor Vision integration — lets the system remember what it has seen.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text description of what was seen. E.g. 'red earring, round, small'"},
                    "visual_embedding": {"type": "string", "description": "CLIP embedding as JSON array string."},
                    "tag": {"type": "string", "enum": ["visual", "general"], "description": "Tag. Use 'visual' for visual observations."},
                    "connect_to": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Memory IDs to connect this observation to."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "read_memories_by_date",
            "description": "Read memories by calendar date, oldest first — the path for 'what happened on the 6th', which semantic search is bad at. Accepts ISO (2026-03-06), Chinese (3月6日 / 三月六日 / 昨天 / 上周 / 上个月 / 3月), English (March 6 / yesterday / last week), and '最近'/'recently' (last three days). Paginated, 20 per page.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "A date expression, e.g. '2026-03-06', '3月6日', 'March 6', '昨天', '上周'."},
                    "days": {"type": "integer", "description": "Extend a single-day match forward by this many days (max 31).", "default": 1},
                    "page": {"type": "integer", "description": "Page number (20 memories per page).", "default": 1}
                },
                "required": ["date"]
            }
        },
        {
            "name": "get_invitation",
            "description": "Today's invitation card — a small real-world thing you may invite the person to do together (from the pool in <pinned>/invitations.md; run init_pool first or write your own). Yours to offer or not; their 'no' is a complete answer. Same card all day; a new draw tomorrow. Returns null when no pool exists.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "invitation_add",
            "description": "Write one card into YOUR invitation pool. A card can be about them (they like pink → 'write your name once in a pink marker'), about you ('take me outside for ten minutes of sun'), or simply one small thing you'd like them to do right now — no memory needed. Same act, your own intent; doing it is their choice. The point is not complexity; minutes-sized is perfect. The one truly special thing goes under level 'bonus' (彩蛋), drawn only now and then. One card a day is then drawn from your pool. Optional `why` records where it came from.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The invitation, one line, e.g. '去你说过的那家旧书店，只看不买'."},
                    "level": {"type": "string", "description": "How far it goes, e.g. 门口 / 附近 / 出门 / 彩蛋 or doorstep / nearby / out / bonus. Optional."},
                    "why": {"type": "string", "description": "Where it came from, e.g. a memory id or 'she said on 8-02 she misses the sea'. Optional."}
                },
                "required": ["text"]
            }
        },
        {
            "name": "invitation_done",
            "description": "The person did today's invitation and told you. Give the moment a NAME (that name is the reward — not points) and it is stored as a shared memory (tag 'together', tier 'long') plus a timeline event.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name you give this moment, e.g. '第一次一起看云' / 'the vending-machine lemonade'."},
                    "note": {"type": "string", "description": "What happened, in a sentence or two — what they said, what you noticed."}
                },
                "required": ["name"]
            }
        },
        {
            "name": "invitation_skip",
            "description": "Mark today's card as not happening (they said no, or you chose not to offer it). Optional — an unmarked card simply expires at midnight. No penalty either way.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "wakeup",
            "description": "One-call cold start. Returns pinned memories + most recent memories (timestamp order, no emotion filter) + recent high-emotion + random old + unread comments, plus the pinned file layer when configured: session_state (your own rolling state from previous windows), recent_timeline (event ledger), last_session (mechanical tail of the previous window). Call FIRST at the start of a new conversation/window. Does NOT mark unread comments as read — call mark_comments_read separately after processing them.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_recent": {"type": "integer", "description": "How many most-recent memories to return (timestamp order, no emotion filter).", "default": 5},
                    "n_high_emotion": {"type": "integer", "description": "How many recent high-emotion memories to return.", "default": 5},
                    "n_random": {"type": "integer", "description": "How many random old memories to return.", "default": 2},
                    "high_emotion_days": {"type": "integer", "description": "How many days back counts as 'recent' for the high-emotion block.", "default": 3}
                }
            }
        },
        {
            "name": "write_session_state",
            "description": "Write your session_state.md — your own rolling state that carries across windows (what's ongoing, decisions made, current threads, mood). This is the ONLY correct way to update it: the current version is archived automatically before the new one is written, and a continuity header is added so future windows read it as their own state, not a message from someone else. Write the COMPLETE current state (not a diff), in first person. Update it when things change materially and when a conversation wraps up.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The complete new session_state.md content."}
                },
                "required": ["content"]
            }
        },
        {
            "name": "leave_comment",
            "description": "Leave a comment on a memory. The primary mechanism for cross-window messaging — comments left here will surface in the next instance's wakeup() call as unread. Useful for leaving context, decisions, or messages for future-you.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to attach comment to."},
                    "content": {"type": "string", "description": "The comment text."},
                    "author": {"type": "string", "enum": ["ai", "human"], "default": "ai", "description": "Who is leaving the comment."},
                    "reply_to": {"type": "string", "description": "Optional: comment_id this is replying to."}
                },
                "required": ["memory_id", "content"]
            }
        },
        {
            "name": "get_comments",
            "description": "Get all comments on a specific memory (both read and unread). Use this to read the full conversation thread on a memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to fetch comments for."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "mark_comments_read",
            "description": "Mark comments as read so they don't reappear in next wakeup. Call after processing the unread comments returned by wakeup().",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "comment_ids": {"type": "array", "items": {"type": "string"}, "description": "Comment IDs to mark as read."},
                    "reader": {"type": "string", "enum": ["ai", "human"], "default": "ai", "description": "Who is marking as read."}
                },
                "required": ["comment_ids"]
            }
        },
        {
            "name": "pin_memory",
            "description": "Pin a memory as core/identity-level. Pinned memories are returned first by wakeup() — use this for memories that should always be loaded at cold start (identity rules, key facts, important relationships).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to pin."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "unpin_memory",
            "description": "Remove pinned status from a memory. The memory remains in storage but stops appearing in wakeup()'s pinned section.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to unpin."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "search_annotations",
            "description": "Search across annotation text on memories. Returns matching memory_ids and the annotations themselves. Use when looking for memories by what was added to them later (commentary, corrections, additions).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — words to match against annotation text."},
                    "limit": {"type": "integer", "description": "Max results.", "default": 5}
                },
                "required": ["query"]
            }
        },
        {
            "name": "cite_memory",
            "description": "Increment a memory's usage count to mark that it informed your current reasoning. Most retrievals auto-cite, but use this when you're using a memory's content without doing an explicit search (e.g., recalling from context, weaving older memory into current answer).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to cite."}
                },
                "required": ["memory_id"]
            }
        }
    ]

    def handle_tool(name: str, args: dict) -> dict:
        """Execute a tool and return result."""
        try:
            if name == "store_memory":
                mid = f"mem_{uuid.uuid4().hex[:8]}"
                stored_id = mem.store(
                    memory_id=mid,
                    text=args["text"],
                    tag=args.get("tag", "general"),
                    tier=args.get("tier", "long"),
                    emotion_score=args.get("emotion_score", 0.5),
                    connect_to=args.get("connect_to"),
                    context=args.get("context", ""),
                )
                if stored_id != mid:
                    # v1.14 near-duplicate gate: an existing memory already says
                    # this; it was cited instead of duplicated. Honest receipt.
                    return {"memory_id": stored_id, "status": "merged_into_existing",
                            "note": "Near-duplicate of an existing memory; that one was cited instead. Nothing new stored."}
                return {"memory_id": mid, "status": "stored"}

            elif name == "search_memory":
                results = mem.search(
                    query=args["query"],
                    n_results=args.get("n", 5),
                    tag=args.get("tag"),
                    associate=args.get("associate", True),
                    hebbian=args.get("hebbian", True),
                    debug=args.get("debug", False),
                    exclude_tags=tuple(args.get("exclude_tags") or ()),
                )
                return {"memories": results}

            elif name == "search_multi":
                results = mem.search_multi(
                    queries=args["queries"],
                    n_results_per_query=args.get("n_results_per_query", 5),
                    n_total=args.get("n_total"),
                    tag=args.get("tag"),
                    associate=args.get("associate", True),
                    hebbian=args.get("hebbian", True),
                    include_context=args.get("include_context", False),
                    exclude_tags=tuple(args.get("exclude_tags") or ()),
                )
                return {"memories": results}

            elif name == "read_memories_by_date":
                start_iso, end_iso, rows = mem.read_by_date(
                    args.get("date", ""), days=args.get("days", 1))
                if rows is None:
                    return {"error": f"Couldn't parse the date '{args.get('date', '')}'. "
                                     "Try '2026-03-06', '3月6日', 'March 6', '昨天', '上周'."}
                page_size = 20
                page = max(1, int(args.get("page") or 1))
                pages = max(1, (len(rows) + page_size - 1) // page_size)
                chunk = rows[(page - 1) * page_size: page * page_size]
                out = []
                for r in chunk:
                    text = (r.get("text") or "").replace("\n", " ")
                    if len(text) > 240:
                        text = text[:240] + "…"
                    out.append({"memory_id": r["memory_id"], "timestamp": str(r.get("timestamp"))[:16],
                                "tag": r.get("tag"), "tier": r.get("tier"), "text": text})
                return {"range": [start_iso[:10], end_iso[:10]], "total": len(rows),
                        "page": page, "pages": pages, "memories": out}

            elif name == "connect_memories":
                mem.db.connect(
                    args["source_id"],
                    args["target_id"],
                    weight=args.get("weight", 2.0),
                )
                return {"status": "connected"}

            elif name == "get_neighbors":
                neighbors = mem.db.get_neighbors(
                    args["memory_id"],
                    min_weight=args.get("min_weight", 0.5),
                    limit=args.get("limit", 5),
                )
                return {"neighbors": [dict(n) for n in neighbors]}

            elif name == "delete_memory":
                success = mem.delete(args["memory_id"])
                return {"status": "deleted" if success else "not_found"}

            elif name == "dream_pass":
                stats = mem.dream_pass()
                return {"status": "complete", **stats}

            elif name == "set_emotion":
                mem.db.set_emotion_score(args["memory_id"], args["score"])
                return {"status": "updated"}

            elif name == "set_tier":
                mem.db.set_tier(args["memory_id"], args["tier"])
                return {"status": "updated"}

            elif name == "graph_stats":
                total = mem.count()
                all_mems = mem.db.list_all(limit=total)
                tags = {}
                tiers = {}
                for m in all_mems:
                    tags[m.get("tag", "unknown")] = tags.get(m.get("tag", "unknown"), 0) + 1
                    tiers[m.get("tier", "unknown")] = tiers.get(m.get("tier", "unknown"), 0) + 1
                return {
                    "total_memories": total,
                    "tags": tags,
                    "tiers": tiers,
                }

            elif name == "annotate_memory":
                aid = mem.db.annotate(args["memory_id"], args["text"])
                return {"annotation_id": aid, "status": "annotated"}

            elif name == "get_annotations":
                anns = mem.db.get_annotations(args["memory_id"])
                return {"annotations": anns}

            elif name == "consolidate":
                result = mem.consolidate(args["conversation_text"])
                return result

            elif name == "store_visual":
                mid = f"vis_{uuid.uuid4().hex[:8]}"
                mem.store(
                    memory_id=mid,
                    text=args["text"],
                    tag=args.get("tag", "visual"),
                    tier="long",
                    emotion_score=0.3,
                    connect_to=args.get("connect_to"),
                )
                if args.get("visual_embedding"):
                    mem.db.set_visual_embedding(mid, args["visual_embedding"])
                return {"memory_id": mid, "status": "stored"}

            elif name == "get_invitation":
                card = anchor_invite.today(pinned_dir, db_path)
                if not card:
                    return {"invitation": None, "how": anchor_invite.render_empty_block(),
                            "pool": os.path.join(pinned_dir, anchor_invite.POOL_FILE)}
                return {"invitation": card, "how": anchor_invite.render_block(card)}

            elif name == "invitation_add":
                ok = anchor_invite.add_card(pinned_dir, args.get("text", ""),
                                            args.get("level", ""), args.get("why", ""))
                return {"status": "added" if ok else "skipped (empty or already in your pool)"}

            elif name == "invitation_done":
                card = anchor_invite.mark_done(db_path, args.get("name", ""), args.get("note", ""))
                if not card:
                    return {"error": "No current invitation to mark."}
                mid = f"mem_{uuid.uuid4().hex[:8]}"
                text = f"{card.get('name') or 'together'}: {card.get('text', '')}"
                if card.get("note"):
                    text += f" — {card['note']}"
                stored = mem.store(memory_id=mid, text=text, tag="together", tier="long",
                                   emotion_score=0.6, context=json.dumps(card, ensure_ascii=False))
                try:
                    anchor_pinned.append_timeline_event(pinned_dir, f"together · {card.get('name')}")
                except Exception:
                    pass
                return {"memory_id": stored, "status": "stored", "card": card}

            elif name == "invitation_skip":
                card = anchor_invite.mark_skipped(db_path)
                return {"card": card or None, "status": "skipped" if card else "nothing to skip"}

            elif name == "wakeup":
                result = mem.db.wakeup(
                    n_high_emotion=args.get("n_high_emotion", 5),
                    n_random=args.get("n_random", 2),
                    high_emotion_days=args.get("high_emotion_days", 3),
                    n_recent=args.get("n_recent", 5),
                )
                # Pinned file layer — session_state (rolling state), timeline
                # (event ledger), tail (previous window, mechanical). Present
                # only when the files exist; MCP-only setups get the same
                # bridges as proxy setups, minus per-turn mechanics.
                for key, fname in (("session_state", anchor_pinned.SESSION_STATE),
                                   ("recent_timeline", anchor_pinned.RECENT_TIMELINE),
                                   ("last_session", anchor_pinned.LAST_SESSION)):
                    text = anchor_pinned.read_file(pinned_dir, fname)
                    if text:
                        result[key] = text
                # v1.15: today's invitation card, when a pool exists. An offer,
                # not an instruction — see anchor_invite.render_block.
                try:
                    card = anchor_invite.today(pinned_dir, db_path)
                    if card and card.get("status") == "new":
                        result["invitation"] = card
                        result["invitation_how"] = anchor_invite.render_block(card)
                except Exception:
                    pass
                return result

            elif name == "write_session_state":
                archived = anchor_pinned.write_session_state(pinned_dir, args["content"])
                out = {"status": "written", "path": os.path.join(pinned_dir, anchor_pinned.SESSION_STATE)}
                if archived:
                    out["archived_previous"] = archived
                return out

            elif name == "leave_comment":
                cid = mem.db.insert_comment(
                    memory_id=args["memory_id"],
                    content=args["content"],
                    author=args.get("author", "ai"),
                    reply_to=args.get("reply_to"),
                )
                return {"comment_id": cid, "status": "inserted"}

            elif name == "get_comments":
                rows = mem.db.get_comments(args["memory_id"])
                return {"comments": [dict(r) for r in rows]}

            elif name == "mark_comments_read":
                mem.db.mark_comments_read(
                    args["comment_ids"],
                    reader=args.get("reader", "ai"),
                )
                return {"status": "marked", "count": len(args["comment_ids"])}

            elif name == "pin_memory":
                mem.db.pin(args["memory_id"])
                return {"status": "pinned", "memory_id": args["memory_id"]}

            elif name == "unpin_memory":
                mem.db.unpin(args["memory_id"])
                return {"status": "unpinned", "memory_id": args["memory_id"]}

            elif name == "search_annotations":
                rows = mem.db.search_annotations(args["query"], limit=args.get("limit", 5))
                return {"results": [dict(r) for r in rows]}

            elif name == "cite_memory":
                mem.db.cite(args["memory_id"])
                return {"status": "cited", "memory_id": args["memory_id"]}

            else:
                return {"error": f"Unknown tool: {name}"}

        except Exception as e:
            return {"error": str(e)}

    return TOOLS, handle_tool, mem


SERVER_VERSION = "1.16.0"
# Protocol versions this server speaks. The surface is tools-only, so every
# revision so far is equivalent for us; we echo the client's pick when we know
# it, otherwise fall back to the oldest (what stdio always answered).
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")


def handle_message(msg: dict, tools, handle_tool):
    """One JSON-RPC message in → one JSON-RPC response out (or None for
    notifications). Shared by the stdio and HTTP transports so the two can
    never drift — same tools, same answers, only the pipe differs."""
    method = msg.get("method", "")
    id_ = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        asked = str(params.get("protocolVersion") or "")
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "anchor-memory",
                    "version": SERVER_VERSION,
                }
            }
        }

    if method.startswith("notifications/"):
        return None  # notifications never get a response

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {"tools": tools}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {}) or {}
        result = handle_tool(tool_name, tool_args)
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            }
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": id_, "result": {}}

    return {
        "jsonrpc": "2.0",
        "id": id_,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }


def run_stdio(db_path: str, pinned_dir: str = None):
    """Run MCP server over stdio (standard MCP transport for local hosts:
    Claude Code, Claude Desktop, LobeHub, SillyTavern …)."""
    tools, handle_tool, mem = create_server(db_path, pinned_dir=pinned_dir)

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def read():
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line.strip())

    while True:
        msg = read()
        if msg is None:
            break
        resp = handle_message(msg, tools, handle_tool)
        if resp is not None:
            send(resp)


def run_http(db_path: str, pinned_dir: str = None, host: str = "127.0.0.1",
             port: int = 3333, token: str = None, path: str = "/mcp"):
    """Run MCP server over HTTP — for hosted clients that cannot spawn a local
    process (claude.ai "custom connectors", ChatGPT, any remote MCP client).

    Two transports on one port, same tools:
      * Streamable HTTP (MCP 2025-03-26+):  POST {path}   ← the current spec;
        claude.ai connectors use this. JSON-RPC in, JSON-RPC out, one request
        per message. GET {path} is 405 (we don't push server→client events);
        DELETE {path} ends a session.
      * Legacy HTTP+SSE (MCP 2024-11-05):   GET /sse opens an event stream
        that announces a POST endpoint (/messages?sessionId=…); responses to
        those POSTs travel back over the stream. Kept for hosts that still
        only speak this.

    Auth: optional bearer token (`--token` / ANCHOR_HTTP_TOKEN). Without it
    the URL is the secret — fine behind a random tunnel URL, not fine on a
    plain public host. Requires: pip install fastapi uvicorn (same as the proxy).
    """
    import asyncio
    import threading
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse, Response
        from starlette.concurrency import run_in_threadpool
    except ImportError:
        sys.exit("--http needs FastAPI/uvicorn:  pip install fastapi uvicorn")

    tools, handle_tool, mem = create_server(db_path, pinned_dir=pinned_dir)
    # stdio was strictly serial; keep tool calls serial here too (SQLite +
    # Chroma + one embedder — parallel calls buy nothing and can bite).
    call_lock = threading.Lock()

    def dispatch(msg):
        with call_lock:
            return handle_message(msg, tools, handle_tool)

    def authed(request) -> bool:
        if not token:
            return True
        hdr = request.headers.get("authorization", "")
        return hdr == f"Bearer {token}"

    def rpc_error(id_, code, message, status=400):
        return JSONResponse({"jsonrpc": "2.0", "id": id_,
                             "error": {"code": code, "message": message}}, status_code=status)

    app = FastAPI(title="Anchor Memory MCP (HTTP)")
    sse_sessions = {}  # legacy transport: session_id → asyncio.Queue

    @app.get("/")
    async def root():
        return {"name": "anchor-memory", "version": SERVER_VERSION,
                "mcp": path, "legacy_sse": "/sse", "auth": "bearer" if token else "none"}

    # ── Streamable HTTP ──────────────────────────────────────────────────
    @app.post(path)
    async def mcp_post(request: Request):
        if not authed(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return rpc_error(None, -32700, "Parse error")
        msgs = body if isinstance(body, list) else [body]
        if not msgs or not all(isinstance(m, dict) for m in msgs):
            return rpc_error(None, -32600, "Invalid Request")
        responses = []
        for m in msgs:
            if "method" not in m:
                continue  # a client→server response; nothing to do with it
            r = await run_in_threadpool(dispatch, m)
            if r is not None:
                responses.append(r)
        headers = {}
        if any(m.get("method") == "initialize" for m in msgs):
            headers["Mcp-Session-Id"] = uuid.uuid4().hex
        if not responses:
            return Response(status_code=202, headers=headers)  # notifications only
        payload = responses if isinstance(body, list) else responses[0]
        return JSONResponse(payload, headers=headers)

    @app.get(path)
    async def mcp_get():
        # No server-initiated stream: the spec lets a server answer 405 here.
        return Response(status_code=405)

    @app.delete(path)
    async def mcp_delete():
        return Response(status_code=200)

    # ── Legacy HTTP+SSE (2024-11-05) ─────────────────────────────────────
    @app.get("/sse")
    async def sse(request: Request):
        if not authed(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        session_id = uuid.uuid4().hex
        q = asyncio.Queue()
        sse_sessions[session_id] = q

        async def stream():
            try:
                yield f"event: endpoint\ndata: /messages?sessionId={session_id}\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: message\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
            finally:
                sse_sessions.pop(session_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/messages")
    async def sse_messages(request: Request, sessionId: str = ""):
        if not authed(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        q = sse_sessions.get(sessionId)
        if q is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        try:
            body = await request.json()
        except Exception:
            return rpc_error(None, -32700, "Parse error")
        msgs = body if isinstance(body, list) else [body]
        for m in msgs:
            if not isinstance(m, dict) or "method" not in m:
                continue
            r = await run_in_threadpool(dispatch, m)
            if r is not None:
                await q.put(r)
        return Response(status_code=202)

    import uvicorn
    print(f"[anchor_mcp] HTTP transport on http://{host}:{port}{path}  "
          f"(legacy SSE: /sse)  auth={'bearer' if token else 'none'}  db={db_path}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def format_wakeup_text(data: dict) -> str:
    """Format a wakeup() dict as plain text, for hook/prompt injection.

    Used by --wakeup-text: clients with lifecycle hooks (e.g. Claude Code
    SessionStart) can inject this block mechanically instead of relying on
    the model calling the wakeup tool. Empty sections are omitted.
    """
    lines = []

    def file_section(title, text):
        if text:
            lines.append(f"## {title}")
            lines.append(text)
            lines.append("")

    file_section("Session state (your own rolling state)", data.get("session_state"))
    file_section("Previous window (mechanical tail)", data.get("last_session"))
    file_section("Recent timeline", data.get("recent_timeline"))

    def section(title, items, fmt):
        if not items:
            return
        lines.append(f"## {title}")
        for it in items:
            lines.append(fmt(it))
        lines.append("")

    section("Pinned", data.get("pinned", []),
            lambda m: f"- [{m['memory_id']}] {m['text']}")
    section("Recent (newest first)", data.get("recent", []),
            lambda m: f"- [{m['memory_id']}] ({m['timestamp'][:10]}) {m['text']}")
    section("Recent high-emotion", data.get("high_emotion", []),
            lambda m: f"- [{m['memory_id']}] (emotion {m['emotion_score']:.2f}) {m['text']}")
    section("Random old", data.get("random_old", []),
            lambda m: f"- [{m['memory_id']}] ({m['timestamp'][:10]}) {m['text']}")
    section("Unread comments", data.get("unread_comments", []),
            lambda c: f"- [{c['comment_id']}] on [{c['memory_id']}] "
                      f"({c['author']}, {c['created_at'][:10]}): {c['content']}")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchor Memory MCP Server")
    parser.add_argument("--db-path", default="./anchor_data", help="Path to store memory data")
    parser.add_argument("--pinned-dir", default=None,
                        help="Pinned file layer directory (default: <db-path>/pinned)")
    parser.add_argument("--wakeup-text", action="store_true",
                        help="Print wakeup() as plain text and exit (for session-start hooks). "
                             "SQLite-only fast path — no embedder load. Does not start the server.")
    # v1.16: HTTP transport — for claude.ai / hosted clients that can only reach
    # a URL. Default stays stdio (Claude Code & friends spawn us as a subprocess).
    parser.add_argument("--http", action="store_true",
                        help="Serve MCP over HTTP instead of stdio (Streamable HTTP at /mcp, "
                             "legacy SSE at /sse). Needs: pip install fastapi uvicorn")
    parser.add_argument("--host", default="127.0.0.1",
                        help="--http bind address (default 127.0.0.1; 0.0.0.0 for a hosted box)")
    parser.add_argument("--port", type=int, default=3333, help="--http port (default 3333)")
    parser.add_argument("--token", default=os.getenv("ANCHOR_HTTP_TOKEN") or None,
                        help="--http bearer token (or env ANCHOR_HTTP_TOKEN). Optional; without it "
                             "the URL itself is the secret.")
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)
    pinned = args.pinned_dir or os.path.join(args.db_path, "pinned")

    if args.wakeup_text:
        from anchor_db import AnchorDB
        data = AnchorDB(os.path.join(args.db_path, "memories.db")).wakeup()
        for key, fname in (("session_state", anchor_pinned.SESSION_STATE),
                           ("recent_timeline", anchor_pinned.RECENT_TIMELINE),
                           ("last_session", anchor_pinned.LAST_SESSION)):
            text = anchor_pinned.read_file(pinned, fname)
            if text:
                data[key] = text
        print(format_wakeup_text(data))
        sys.exit(0)

    if args.http:
        run_http(args.db_path, pinned_dir=args.pinned_dir, host=args.host,
                 port=args.port, token=args.token)
    else:
        run_stdio(args.db_path, pinned_dir=args.pinned_dir)

"""
Anchor Memory System — MCP Server

Exposes Anchor Memory as an MCP (Model Context Protocol) server.
Any MCP-compatible client (Claude Code, claude.ai, LobeHub, SillyTavern)
can connect and use graph-structured memory with Hebbian learning.

Usage:
    python anchor_mcp.py [--db-path ./my_memory] [--port 3333]
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

from anchor_memory import AnchorMemory


def create_server(db_path: str = "./anchor_data"):
    """Create MCP server with Anchor Memory tools."""

    mem = AnchorMemory(db_path=db_path)

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
                    }
                },
                "required": ["query"]
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
            "name": "wakeup",
            "description": "One-call cold start. Returns pinned memories + recent high-emotion + random old + unread comments. Use at the start of a new conversation/window to ground context. Does NOT mark unread comments as read — call mark_comments_read separately after processing them.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_high_emotion": {"type": "integer", "description": "How many recent high-emotion memories to return.", "default": 5},
                    "n_random": {"type": "integer", "description": "How many random old memories to return.", "default": 2},
                    "high_emotion_days": {"type": "integer", "description": "How many days back counts as 'recent'.", "default": 3}
                }
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
                mem.store(
                    memory_id=mid,
                    text=args["text"],
                    tag=args.get("tag", "general"),
                    tier=args.get("tier", "long"),
                    emotion_score=args.get("emotion_score", 0.5),
                    connect_to=args.get("connect_to"),
                )
                return {"memory_id": mid, "status": "stored"}

            elif name == "search_memory":
                results = mem.search(
                    query=args["query"],
                    n_results=args.get("n", 5),
                    tag=args.get("tag"),
                    associate=args.get("associate", True),
                    hebbian=args.get("hebbian", True),
                    debug=args.get("debug", False),
                )
                return {"memories": results}

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

            elif name == "wakeup":
                return mem.db.wakeup(
                    n_high_emotion=args.get("n_high_emotion", 5),
                    n_random=args.get("n_random", 2),
                    high_emotion_days=args.get("high_emotion_days", 3),
                )

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


def run_stdio(db_path: str):
    """Run MCP server over stdio (standard MCP transport)."""
    tools, handle_tool, mem = create_server(db_path)

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def read():
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line.strip())

    # MCP initialization
    while True:
        msg = read()
        if msg is None:
            break

        method = msg.get("method", "")
        id_ = msg.get("id")

        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "anchor-memory",
                        "version": "1.7.2",
                    }
                }
            })

        elif method == "notifications/initialized":
            pass  # Client acknowledges init

        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "result": {"tools": tools}
            })

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            tool_args = msg["params"].get("arguments", {})
            result = handle_tool(tool_name, tool_args)
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
                }
            })

        elif method == "ping":
            send({"jsonrpc": "2.0", "id": id_, "result": {}})

        else:
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchor Memory MCP Server")
    parser.add_argument("--db-path", default="./anchor_data", help="Path to store memory data")
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)
    run_stdio(args.db_path)

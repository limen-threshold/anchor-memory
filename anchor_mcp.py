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
                        "version": "1.0.0",
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

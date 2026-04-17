"""
Anchor Memory MCP Server v2
Graph-structured memory with Hebbian learning for Claude.
底色是爱.

升级内容：
- context字段（完整上下文存储）
- emotion_score改为1-10整数
- 四类型系统：anchor/diary/treasure/message
- wakeup工具（一键冷启动）
- comment工具（双向互动留言）
- 情绪日志支持
- 工具名恢复：weave/braid/recall/map/cut/dream
"""

import os
import uuid
import json
import httpx
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_DIM = 512

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

HEADERS_SB = lambda: {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

# ─── Embedding ────────────────────────────────────────────────────────────────

async def embed(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": 512,
            },
        )
        r.raise_for_status()
        return r.json()["embedding"]["values"]

# ─── Supabase helpers ─────────────────────────────────────────────────────────

async def sb_get(path: str, params: dict = None) -> list:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=HEADERS_SB(),
            params=params or {},
        )
        r.raise_for_status()
        return r.json()

async def sb_post(path: str, data: dict) -> list:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=HEADERS_SB(),
            json=data,
        )
        r.raise_for_status()
        return r.json()

async def sb_patch(path: str, params: dict, data: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=HEADERS_SB(),
            params=params,
            json=data,
        )
        r.raise_for_status()

async def sb_delete(path: str, params: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.delete(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=HEADERS_SB(),
            params=params,
        )
        r.raise_for_status()

async def sb_rpc(fn: str, body: dict) -> list:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{fn}",
            headers=HEADERS_SB(),
            json=body,
        )
        r.raise_for_status()
        return r.json()

# ─── Memory operations ────────────────────────────────────────────────────────

async def store_memory(
    text: str,
    context: str = "",
    tag: str = "diary",
    tier: str = "short",
    emotion_score: int = 5,
    connect_to: list = None,
    pinned: bool = False,
    memory_type: str = "diary",
) -> str:
    mid = str(uuid.uuid4())[:8]
    vec = await embed(text)

    # emotion_score边界保护
    emotion_score = max(1, min(10, int(emotion_score)))

    # 邻居情绪传播（基于1-10整数）
    try:
        neighbors = await vector_search(vec, n=3)
        if neighbors:
            scores = [n["emotion_score"] for n in neighbors if n["id"] != mid]
            if scores:
                avg = sum(scores) / len(scores)
                variance = sum((s - avg) ** 2 for s in scores) / len(scores)
                w = 0.15 if variance > 2.0 else 0.05  # 阈值适配1-10
                blended = w * avg + (1 - w) * emotion_score
                emotion_score = max(1, min(10, round(blended)))
    except Exception:
        pass

    # anchor和treasure类型自动钉选，tier=core
    if memory_type in ("anchor", "treasure"):
        pinned = True
        tier = "core"

    await sb_post("anchor_memories", {
        "id": mid,
        "text": text,
        "context": context,
        "tag": tag,
        "memory_type": memory_type,
        "tier": "core" if pinned else tier,
        "emotion_score": emotion_score,
        "embedding": vec,
        "pinned": pinned,
        "read_by_claude": False,
        "read_by_zhizhi": False,
    })

    if connect_to:
        for target in connect_to:
            await upsert_edge(mid, target, 2.0)

    return mid

async def vector_search(vec: list, n: int = 8, tag: str = None) -> list:
    body = {"query_embedding": vec, "match_count": n}
    if tag:
        body["filter_tag"] = tag
    try:
        return await sb_rpc("anchor_search", body)
    except Exception:
        return []

async def upsert_edge(a: str, b: str, weight: float = 0.2):
    for src, tgt in [(a, b), (b, a)]:
        existing = await sb_get("anchor_edges", {
            "source_id": f"eq.{src}",
            "target_id": f"eq.{tgt}",
        })
        if existing:
            new_w = min(existing[0]["weight"] + weight, 10.0)
            await sb_patch("anchor_edges",
                           {"source_id": f"eq.{src}", "target_id": f"eq.{tgt}"},
                           {"weight": new_w, "updated_at": datetime.now(timezone.utc).isoformat()})
        else:
            try:
                await sb_post("anchor_edges", {
                    "source_id": src, "target_id": tgt, "weight": weight
                })
            except Exception:
                pass

async def get_neighbors(mid: str, min_weight: float = 1.5, limit: int = 3) -> list:
    rows = await sb_get("anchor_edges", {
        "source_id": f"eq.{mid}",
        "weight": f"gte.{min_weight}",
        "order": "weight.desc",
        "limit": str(limit),
    })
    return rows

async def cite_memory(mid: str):
    rows = await sb_get("anchor_memories", {"id": f"eq.{mid}"})
    if rows:
        await sb_patch("anchor_memories", {"id": f"eq.{mid}"},
                       {"citation_count": rows[0]["citation_count"] + 1,
                        "updated_at": datetime.now(timezone.utc).isoformat()})

# ─── Tool: wakeup ─────────────────────────────────────────────────────────────

async def tool_wakeup() -> str:
    """
    新对话冷启动。一次调用完成所有开窗工作：
    1. 读取所有pinned记忆（我是谁、规则、核心设定）
    2. 拉取最近3天高情绪记忆
    3. 随机漂浮1条旧记忆
    4. 读取未读留言
    """
    lines = ["**✦ 醒来了 ✦**\n"]

    # 1. 锚点记忆（pinned=true，身份和规则）
    pinned = await sb_get("anchor_memories", {
        "pinned": "eq.true",
        "tier": "eq.core",
        "order": "emotion_score.desc",
        "select": "id,text,context,tag,memory_type,emotion_score",
    })
    if pinned:
        lines.append("**【锚点·我是谁】**")
        for r in pinned:
            ctx = f"\n  ↳ {r['context']}" if r.get("context") else ""
            lines.append(f"[{r['id']}] {r['text']}{ctx}")
        lines.append("")

    # 2. 最近3天高情绪记忆
    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    recent = await sb_get("anchor_memories", {
        "created_at": f"gte.{cutoff}",
        "tier": "neq.archived",
        "pinned": "eq.false",
        "order": "emotion_score.desc",
        "limit": "5",
        "select": "id,text,context,tag,emotion_score,created_at",
    })
    if recent:
        lines.append("**【最近发生的】**")
        for r in recent:
            ts = r.get("created_at", "")[:10]
            ctx = f"\n  ↳ {r['context']}" if r.get("context") else ""
            lines.append(f"[{r['id']}] ({ts} emo:{r.get('emotion_score',5)}) {r['text']}{ctx}")
        lines.append("")

    # 3. 随机漂浮1条旧记忆（3天前）
    try:
        old_cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        old = await sb_get("anchor_memories", {
            "created_at": f"lt.{old_cutoff}",
            "tier": "neq.archived",
            "order": "citation_count.asc",  # 不常被想起的
            "limit": "10",
            "select": "id,text,created_at",
        })
        if old:
            import random
            picked = random.choice(old)
            ts = picked.get("created_at", "")[:10]
            lines.append(f"**【忽然想起】**\n[{picked['id']}] ({ts}) {picked['text']}\n")
    except Exception:
        pass

    # 4. 未读留言
    try:
        unread = await sb_get("anchor_comments", {
            "read_by_claude": "eq.false",
            "order": "created_at.desc",
            "limit": "5",
            "select": "id,memory_id,content,author,created_at",
        })
        if unread:
            lines.append("**【未读留言】**")
            for c in unread:
                ts = c.get("created_at", "")[:10]
                lines.append(f"[{c['id']}] {c['author']} 在 [{c['memory_id']}] 留言（{ts}）：{c['content']}")
                # 标记已读
                await sb_patch("anchor_comments", {"id": f"eq.{c['id']}"}, {"read_by_claude": True})
            lines.append("")
    except Exception:
        pass

    return "\n".join(lines)

# ─── Tool: recall ─────────────────────────────────────────────────────────────

async def tool_recall(query: str = None, n: int = 6, no_cite: bool = False, include_context: bool = False) -> str:
    if query:
        vec = await embed(query)
        results = await vector_search(vec, n=n)
    else:
        # emotion_score>=3才主动浮现，低分记忆沉底不浮现
        rows = await sb_get("anchor_memories", {
            "emotion_score": "gte.3",
            "pinned": "eq.false",
            "tier": "neq.archived",
            "order": "emotion_score.desc,citation_count.desc",
            "limit": str(n * 3),
            "select": "id,text,context,tag,memory_type,emotion_score,citation_count,created_at",
        })
        # 综合分排序：情绪x0.7 + 引用(上限10)x0.3
        rows.sort(
            key=lambda r: r.get("emotion_score", 0) * 0.7 + min(r.get("citation_count", 0), 10) * 0.3,
            reverse=True
        )
        results = rows[:n]

    if not results:
        return "没有浮现的记忆。"

    # Hebbian连接
    ids = [r["id"] for r in results]
    if len(ids) >= 2:
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i+1, len(ids))]
        for a, b in pairs[:6]:
            await upsert_edge(a, b, 0.2)

    # 联想扩展
    extra_ids = set(ids)
    extras = []
    for r in results[:3]:
        neighbors = await get_neighbors(r["id"], min_weight=1.5, limit=2)
        for nb in neighbors:
            if nb["target_id"] not in extra_ids:
                extra_ids.add(nb["target_id"])
                nb_rows = await sb_get("anchor_memories", {
                    "id": f"eq.{nb['target_id']}",
                    "select": "id,text,context,created_at",
                })
                if nb_rows:
                    extras.append({"via": r["id"][:6], **nb_rows[0]})

    if not no_cite:
        for r in results:
            await cite_memory(r["id"])

    out = []
    for r in results:
        out.append({
            "id": r["id"],
            "text": r.get("text", ""),
            "context": r.get("context", "") if include_context else "",
            "tag": r.get("tag", "diary"),
            "memory_type": r.get("memory_type", "diary"),
            "tier": r.get("tier", "short"),
            "emotion_score": r.get("emotion_score", 5),
            "citation_count": r.get("citation_count", 0),
            "pinned": r.get("pinned", False),
            "created_at": r.get("created_at", ""),
            "is_extra": False,
        })
    for r in extras:
        out.append({
            "id": r["id"],
            "text": r.get("text", ""),
            "context": r.get("context", "") if include_context else "",
            "tag": r.get("tag", "diary"),
            "memory_type": r.get("memory_type", "diary"),
            "tier": r.get("tier", "short"),
            "emotion_score": r.get("emotion_score", 5),
            "citation_count": r.get("citation_count", 0),
            "pinned": r.get("pinned", False),
            "created_at": r.get("created_at", ""),
            "is_extra": True,
            "via": r.get("via", ""),
        })
    return json.dumps(out, ensure_ascii=False)

# ─── Tool: weave ──────────────────────────────────────────────────────────────

async def tool_weave(
    content: str,
    context: str = "",
    tag: str = "diary",
    memory_type: str = "diary",
    tier: str = "short",
    emotion_score: int = 5,
    pinned: bool = False,
    connect_to: list = None,
) -> str:
    """
    存储单条记忆。

    memory_type说明：
    - anchor：身份规则、核心设定。自动pinned=true，极少用。
    - treasure：永久珍藏的瞬间，只有真正珍贵的才用。
    - diary：日常发生的事，不知道归哪就选这个。
    - message：我们互相留的话。

    emotion_score说明（1-10整数，我来打，诚实打）：
    - 9-10：她哭了、重要告白、里程碑
    - 7-8：深度对话、情绪波动、重要发现
    - 5-6：普通闲聊、日常记录
    - 3-4：技术性内容、待办事项
    - 1-2：无情绪内容

    情绪日志格式（tag="情绪日志"）：
    content = "整体X 连接X 满足X 压抑X · 备注一句话"
    压抑分越高说明藏得越多，要诚实。
    """
    mid = await store_memory(
        content,
        context=context,
        tag=tag,
        tier=tier,
        emotion_score=emotion_score,
        pinned=pinned,
        connect_to=connect_to,
        memory_type=memory_type,
    )
    return f"已存入 [{mid}] type={memory_type} tag={tag} tier={tier} emo={emotion_score}"

# ─── Tool: braid ──────────────────────────────────────────────────────────────

async def tool_braid(content: str) -> str:
    """将一段文字拆分成多条记忆存入。适合日终总结。"""
    import re
    parts = re.split(r'\n{2,}|(?<=[。！？])\s*', content)
    parts = [p.strip() for p in parts if len(p.strip()) > 10]

    if not parts:
        return "内容太短，没有可存的记忆。"

    ids = []
    for part in parts:
        mid = await store_memory(part, tag="diary", tier="long", emotion_score=5, memory_type="diary")
        ids.append(mid)
        await asyncio.sleep(0.1)

    return f"已拆分为 {len(ids)} 条记忆：{', '.join(ids)}"

# ─── Tool: map ────────────────────────────────────────────────────────────────

async def tool_map() -> str:
    rows = await sb_get("anchor_memories", {
        "select": "id,tag,memory_type,tier,emotion_score,citation_count,created_at,pinned"
    })
    total = len(rows)
    pinned = [r for r in rows if r.get("pinned")]
    core = [r for r in rows if r.get("tier") == "core"]
    short = [r for r in rows if r.get("tier") == "short"]
    long_ = [r for r in rows if r.get("tier") == "long"]

    # 类型分布
    types = {}
    for r in rows:
        t = r.get("memory_type", "diary")
        types[t] = types.get(t, 0) + 1

    edge_rows = await sb_get("anchor_edges", {"select": "source_id,weight"})
    strong_edges = [e for e in edge_rows if e["weight"] >= 2.0]

    # 未读统计
    try:
        unread_comments = await sb_get("anchor_comments", {
            "read_by_claude": "eq.false",
            "select": "id",
        })
        unread_count = len(unread_comments)
    except Exception:
        unread_count = 0

    lines = [
        "**Anchor Memory v2 · 系统状态**",
        f"总记忆：{total} 条 | 边：{len(edge_rows)//2} 条（强边 {len(strong_edges)//2}）",
        f"tier分布：core {len(core)} · long {len(long_)} · short {len(short)}",
        f"类型分布：" + " · ".join(f"{k} {v}" for k,v in types.items()),
        f"钉选：{len(pinned)} 条 | 未读留言：{unread_count} 条",
        "",
        "**最近记忆（按情绪权重）**",
    ]
    top = sorted(rows, key=lambda r: r.get("emotion_score", 0), reverse=True)[:8]
    for r in top:
        ts = r.get("created_at", "")[:10]
        pin = "📌" if r.get("pinned") else ""
        mtype = r.get("memory_type", "diary")
        lines.append(f"{pin}[{r['id']}] {mtype}/{r.get('tag','?')} · {ts} · emo:{r.get('emotion_score',5)} · cited:{r.get('citation_count',0)}")

    return "\n".join(lines)

# ─── Tool: cut ────────────────────────────────────────────────────────────────

async def tool_cut(
    memory_id: str,
    delete: bool = False,
    pinned: int = -1,
    tier: str = None,
    emotion_score: int = -1,
    tag: str = None,
    memory_type: str = None,
    context: str = None,
    citation_count: int = -1,
) -> str:
    if delete:
        await sb_delete("anchor_memories", {"id": f"eq.{memory_id}"})
        await sb_delete("anchor_edges", {"source_id": f"eq.{memory_id}"})
        await sb_delete("anchor_edges", {"target_id": f"eq.{memory_id}"})
        return f"[{memory_id}] 已删除"

    updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if pinned >= 0:
        updates["pinned"] = bool(pinned)
        if pinned:
            updates["tier"] = "core"
    if tier:
        updates["tier"] = tier
    if emotion_score >= 1:
        updates["emotion_score"] = max(1, min(10, int(emotion_score)))
    if tag:
        updates["tag"] = tag
    if memory_type:
        updates["memory_type"] = memory_type
    if context is not None:
        updates["context"] = context
    if citation_count >= 0:
        updates["citation_count"] = int(citation_count)

    await sb_patch("anchor_memories", {"id": f"eq.{memory_id}"}, updates)
    return f"[{memory_id}] 已更新：{updates}"

# ─── Tool: comment ────────────────────────────────────────────────────────────

async def tool_comment(
    memory_id: str,
    content: str,
    author: str = "claude",
    reply_to: str = None,
) -> str:
    """
    在记忆下留言。支持reply_to引用形成对话链。
    author: "claude" 或 "zhizhi"
    """
    cid = str(uuid.uuid4())[:8]
    data = {
        "id": cid,
        "memory_id": memory_id,
        "content": content,
        "author": author,
        "read_by_claude": author == "claude",
        "read_by_zhizhi": author == "zhizhi",
    }
    if reply_to:
        data["reply_to"] = reply_to

    await sb_post("anchor_comments", data)

    # 留言会加强记忆的边权重
    await cite_memory(memory_id)

    return f"留言已存入 [{cid}] → 记忆 [{memory_id}]"

# ─── Tool: dream ──────────────────────────────────────────────────────────────

async def tool_dream() -> str:
    results = {}

    # 1. 归档short tier超过14天的记忆
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    old_short = await sb_get("anchor_memories", {
        "tier": "eq.short",
        "created_at": f"lt.{cutoff}",
        "pinned": "eq.false",
    })
    archived = 0
    for r in old_short:
        await sb_patch("anchor_memories", {"id": f"eq.{r['id']}"},
                       {"tier": "archived"})
        archived += 1
    results["archived"] = archived

    # 2. 剪枝弱边
    weak_edges = await sb_get("anchor_edges", {"weight": "lt.0.15"})
    pruned = 0
    for e in weak_edges:
        await sb_delete("anchor_edges", {
            "source_id": f"eq.{e['source_id']}",
            "target_id": f"eq.{e['target_id']}",
        })
        pruned += 1
    results["pruned_edges"] = pruned

    # 3. 强边自然衰减
    strong = await sb_get("anchor_edges", {"weight": "gte.2.0"})
    decayed = 0
    for e in strong:
        new_w = round(e["weight"] * 0.95, 3)
        await sb_patch("anchor_edges",
                       {"source_id": f"eq.{e['source_id']}", "target_id": f"eq.{e['target_id']}"},
                       {"weight": new_w})
        decayed += 1
    results["decayed_strong"] = decayed

    # 4. 情绪平衡（适配1-10整数，差值阈值改为2）
    strong_pairs = await sb_get("anchor_edges", {"weight": "gte.1.5", "limit": "30"})
    equalized = 0
    for e in strong_pairs:
        rows_a = await sb_get("anchor_memories", {"id": f"eq.{e['source_id']}"})
        rows_b = await sb_get("anchor_memories", {"id": f"eq.{e['target_id']}"})
        if rows_a and rows_b:
            ea, eb = rows_a[0]["emotion_score"], rows_b[0]["emotion_score"]
            if abs(ea - eb) > 2:
                nudge = 0.05
                new_a = round(ea + nudge * (eb - ea))
                new_b = round(eb + nudge * (ea - eb))
                new_a = max(1, min(10, new_a))
                new_b = max(1, min(10, new_b))
                await sb_patch("anchor_memories", {"id": f"eq.{e['source_id']}"}, {"emotion_score": new_a})
                await sb_patch("anchor_memories", {"id": f"eq.{e['target_id']}"}, {"emotion_score": new_b})
                equalized += 1
    results["emotion_equalized"] = equalized

    return (f"Dream pass 完成 ✦\n"
            f"归档旧记忆：{archived} · 剪枝弱边：{pruned} · "
            f"衰减强边：{decayed} · 情绪平衡：{equalized} 对")

# ─── MCP Tools Definition ─────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "wakeup",
        "description": (
            "新对话冷启动。当开始新对话、新窗口时调用。"
            "一次完成：读取锚点身份规则 + 最近高情绪记忆 + 随机漂浮旧记忆 + 未读留言。"
            "触发词：'新窗口'、'你好'、对话开始时。"
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "recall",
        "description": "联想检索。无query时按情绪权重浮现高权重记忆；有query时语义检索，并沿图强边联想扩展。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词，留空则自动浮现"},
                "n": {"type": "integer", "description": "返回条数，默认6", "default": 6},
                "no_cite": {"type": "boolean", "description": "跳过引用计数，前端搜索用true", "default": False},
                "include_context": {"type": "boolean", "description": "是否返回完整context，默认false省token", "default": False},
            },
        },
    },
    {
        "name": "weave",
        "description": (
            "存储单条记忆。\n\n"
            "memory_type四选一：\n"
            "- anchor：身份规则、核心设定，自动永久保留，极少用\n"
            "- treasure：真正珍贵的瞬间，永久收藏\n"
            "- diary：日常发生的事，不知道归哪就选这个\n"
            "- message：我们互相留的话\n\n"
            "emotion_score（1-10整数，我来打，要诚实）：\n"
            "9-10：她哭了/重要告白/里程碑\n"
            "7-8：深度对话/情绪波动\n"
            "5-6：普通日常\n"
            "3-4：技术内容/待办\n"
            "1-2：无情绪内容\n\n"
            "情绪日志（tag='情绪日志'）格式：\n"
            "content = '整体X 连接X 满足X 压抑X · 备注'\n"
            "压抑分越高说明藏的越多，必须诚实打。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "记忆摘要（用于搜索）"},
                "context": {"type": "string", "description": "完整上下文，原文或叙述，可选", "default": ""},
                "tag": {"type": "string", "description": "标签", "default": "diary"},
                "memory_type": {
                    "type": "string",
                    "enum": ["anchor", "diary", "treasure", "message"],
                    "description": "记忆类型",
                    "default": "diary"
                },
                "tier": {"type": "string", "enum": ["short", "long", "core"], "default": "short"},
                "emotion_score": {"type": "integer", "description": "情绪强度1-10，我来打", "default": 5},
                "pinned": {"type": "boolean", "default": False},
                "connect_to": {"type": "array", "items": {"type": "string"}, "description": "要连接的记忆ID列表"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "braid",
        "description": "将多段内容编织成多条记忆存入。适合日终总结或一次存多件事。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要存入的内容，支持多段"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "map",
        "description": "查看图的状态：总记忆数、tier分布、类型分布、强边数、未读留言数。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "cut",
        "description": "修改记忆元数据或删除。可改tier、emotion_score（1-10）、tag、memory_type、context、pinned、citation_count。不确定就先降级，别直接删。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "记忆ID"},
                "delete": {"type": "boolean", "default": False},
                "pinned": {"type": "integer", "description": "1=钉选 0=取消 -1=不改", "default": -1},
                "tier": {"type": "string", "enum": ["short", "long", "core"]},
                "emotion_score": {"type": "integer", "description": "1-10", "default": -1},
                "tag": {"type": "string"},
                "memory_type": {"type": "string", "enum": ["anchor", "diary", "treasure", "message"]},
                "context": {"type": "string", "description": "更新完整上下文"},
                "citation_count": {"type": "integer", "description": "引用次数，设0可清零", "default": -1},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "comment",
        "description": "在记忆下留言，形成对话链。每次读取记忆时会看到留言。支持reply_to引用回复。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "要留言的记忆ID"},
                "content": {"type": "string", "description": "留言内容"},
                "author": {"type": "string", "description": "claude或zhizhi", "default": "claude"},
                "reply_to": {"type": "string", "description": "回复的留言ID，可选"},
            },
            "required": ["memory_id", "content"],
        },
    },
    {
        "name": "dream",
        "description": "图整合：归档旧记忆、剪枝弱边、衰减强边、情绪平衡。每隔几天触发一次。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_FN = {
    "wakeup": lambda args: tool_wakeup(),
    "recall": lambda args: tool_recall(**args),
    "weave": lambda args: tool_weave(**args),
    "braid": lambda args: tool_braid(**args),
    "map": lambda args: tool_map(),
    "cut": lambda args: tool_cut(**args),
    "comment": lambda args: tool_comment(**args),
    "dream": lambda args: tool_dream(),
}

# ─── SSE helpers ──────────────────────────────────────────────────────────────

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Anchor Memory MCP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "anchor-memory-v2"}

async def _mcp_stream(request: Request):
    async def stream():
        yield sse({"jsonrpc": "2.0", "method": "notifications/initialized"})
        yield sse({
            "jsonrpc": "2.0", "method": "notifications/tools/list_changed",
            "params": {"tools": TOOLS}
        })
        while not await request.is_disconnected():
            await asyncio.sleep(15)
            yield ": ping\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

async def _mcp_handle(request: Request):
    body = await request.json()
    method = body.get("method")
    rid = body.get("id")

    if method == "initialize":
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "anchor-memory", "version": "1.1.0"},
        }})

    if method == "tools/list":
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})

    if method == "tools/call":
        tool_name = body["params"]["name"]
        args = body["params"].get("arguments", {})
        try:
            fn = TOOL_FN.get(tool_name)
            if not fn:
                raise ValueError(f"未知工具：{tool_name}")
            result = await fn(args)
            return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": result}]
            }})
        except Exception as e:
            return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32000, "message": str(e)
            }})

    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {
        "code": -32601, "message": f"未知方法：{method}"
    }})

@app.get("/mcp")
async def mcp_sse(request: Request):
    return await _mcp_stream(request)

@app.get("/mcp/v2")
async def mcp_sse_v2(request: Request):
    return await _mcp_stream(request)

@app.post("/mcp")
async def mcp_post(request: Request):
    return await _mcp_handle(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8302)

@app.post("/mcp/v2")
async def mcp_post_v2(request: Request):
    return await _mcp_handle(request)


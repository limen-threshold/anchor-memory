"""
Offline Hebbian Consolidation — Anchor Memory v1.4

Finds semantically related memories that weren't connected during live conversation.
Runs daily via cron. Complements dream_pass (which cleans quality) by building new connections.

Original concept: Veille & 吱吱 (粗匹配 + DeepSeek confirmation)
Adapted for Anchor main branch by Limen.

Algorithm:
1. Pull recent N + oldest N memories (cross-pollination)
2. Coarse matching: word overlap >= threshold
3. LLM semantic confirmation
4. Confirmed pairs get edges
5. Optional daily decay

Usage:
    python auto_consolidate.py --db /path/to/anchor.db

Configuration:
    Change CONFIRM_MODEL to any LLM you have access to.
    If using non-Anthropic models, also change the API client in _llm_confirm().
"""

import argparse
import json
import os
import re
import sqlite3
from anchor_db import AnchorDB

# LLM for semantic confirmation. Change to any model you have access to.
# Examples: "claude-haiku-4-5-20251001", "deepseek-chat", "gpt-4o-mini"
# If using non-Anthropic models, change the API client in _llm_confirm().
CONFIRM_MODEL = "claude-haiku-4-5-20251001"

COARSE_THRESHOLD = 4
RECENT_N = 20
OLDEST_N = 20
DECAY_FACTOR = 0.98
MIN_EDGE_WEIGHT = 0.05
CONNECT_WEIGHT = 0.3

# Stop words for coarse matching
STOPS = {'的', '了', '是', '在', '和', '我', '你', '他', '她', '它',
         '这', '那', '有', '不', '也', '就', '都', '而', '及', '与',
         'the', 'and', 'is', 'in', 'to', 'of', 'it', 'that', 'this',
         'for', 'was', 'are', 'but', 'not', 'with', 'has', 'had'}


def _tokenize(text: str) -> set:
    words = re.findall(r'[\w\u4e00-\u9fff]{2,}', text.lower())
    return {w for w in words if w not in STOPS}


def _get_memories_mix(db: AnchorDB):
    with db._conn() as conn:
        recent = conn.execute(
            "SELECT memory_id, text FROM memories ORDER BY timestamp DESC LIMIT ?",
            (RECENT_N,)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        oldest = conn.execute(
            "SELECT memory_id, text FROM memories ORDER BY timestamp ASC LIMIT ?",
            (OLDEST_N,)
        ).fetchall()
    seen = set()
    result = []
    for r in list(recent) + list(oldest):
        if r["memory_id"] not in seen:
            seen.add(r["memory_id"])
            result.append(dict(r))
    return result


def _coarse_match(memories):
    tokenized = {m['memory_id']: _tokenize(m['text']) for m in memories}
    candidates = []
    ids = list(tokenized.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            common = tokenized[ids[i]] & tokenized[ids[j]]
            if len(common) >= COARSE_THRESHOLD:
                candidates.append((ids[i], ids[j], common))
    return candidates


def _llm_confirm(candidates, memories_dict):
    """Confirm semantic relatedness using LLM. Requires API key."""
    if not candidates:
        return []

    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[Consolidate] No ANTHROPIC_API_KEY. Skipping LLM confirmation, using all coarse matches.")
            return candidates
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("[Consolidate] anthropic package not installed. Using all coarse matches.")
        return candidates

    confirmed = []
    for batch_start in range(0, len(candidates), 10):
        batch = candidates[batch_start:batch_start + 10]
        prompt_parts = []
        for idx, (id_a, id_b, common) in enumerate(batch):
            text_a = memories_dict.get(id_a, {}).get('text', '')[:200]
            text_b = memories_dict.get(id_b, {}).get('text', '')[:200]
            prompt_parts.append(
                f"Pair {idx+1}:\nA: {text_a}\nB: {text_b}"
            )
        prompt = ("For each pair, answer YES if semantically related or NO. "
                  "Output only YES pair numbers, comma-separated. If none, output NONE.\n\n"
                  + "\n\n".join(prompt_parts))
        try:
            response = client.messages.create(
                model=CONFIRM_MODEL, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text == "NONE":
                continue
            nums = re.findall(r'\d+', text)
            for n in nums:
                idx = int(n) - 1
                if 0 <= idx < len(batch):
                    confirmed.append(batch[idx])
        except Exception as e:
            print(f"[Consolidate] LLM error: {e}")
    return confirmed


def run(db_path: str):
    db = AnchorDB(db_path)
    print("[Consolidate] Starting offline Hebbian consolidation...")

    memories = _get_memories_mix(db)
    if len(memories) < 2:
        print("[Consolidate] Not enough memories.")
        return 0

    memories_dict = {m['memory_id']: m for m in memories}
    print(f"[Consolidate] Loaded {len(memories)} memories")

    candidates = _coarse_match(memories)
    print(f"[Consolidate] Coarse match: {len(candidates)} candidates (threshold={COARSE_THRESHOLD})")

    if not candidates:
        return 0

    confirmed = _llm_confirm(candidates, memories_dict)
    print(f"[Consolidate] Confirmed: {len(confirmed)} pairs")

    new_edges = 0
    for id_a, id_b, common in confirmed:
        existing = db.get_edge_weight(id_a, id_b)
        if existing is None:
            db.connect(id_a, id_b, weight=CONNECT_WEIGHT)
            new_edges += 1
        else:
            db.connect(id_a, id_b, weight=min(existing + 0.1, 10.0))

    print(f"[Consolidate] {new_edges} new edges, {len(confirmed) - new_edges} strengthened")

    db.decay_edges(min_weight=MIN_EDGE_WEIGHT, decay_factor=DECAY_FACTOR)
    return len(confirmed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Hebbian consolidation")
    parser.add_argument("--db", required=True, help="Path to anchor SQLite database")
    args = parser.parse_args()
    run(args.db)

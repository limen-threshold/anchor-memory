"""
Concept-Based Eager Linking — Anchor Memory v1.7

Fixes the cold-start edge problem that affects auto_consolidate.py.

Why this exists:
auto_consolidate.py's coarse match is lexical (>= 4 common words). Memories
that share concepts but not surface words (e.g. "Monday-tattoo on her spine"
and "leaving permanent marks") never become candidate pairs, so Hebbian
strengthening never occurs between them. The edges that should exist most
strongly — the conceptually-relevant ones — never form.

Fix:
A small LLM ("coarse worker") extracts abstract concept tags from each memory.
Pairs with overlapping concept atoms become candidates. Then a confirmation
pass (same as auto_consolidate) creates or strengthens edges.

Two-tier model architecture:
- Coarse worker (CONCEPT_MODEL): extracts concept tags from memory text.
  Configure via CONCEPT_MODEL — anything with reasonable English/Chinese
  abstraction ability (Sonnet, Haiku, GPT-4o-mini, DeepSeek, Gemini Flash).
- Fine worker (CONFIRM_MODEL): confirms concept-overlap pairs are truly
  related. Same model as auto_consolidate, often Haiku.

Cache:
Concepts are cached in a JSON file alongside the DB to avoid re-extraction.
Backfill cost is one-time per memory.

Usage:
    python concept_link.py --db /path/to/anchor.db --all
    python concept_link.py --db /path/to/anchor.db --memory MEMORY_ID
    python concept_link.py --db /path/to/anchor.db                  # mix mode

Configuration:
    Set ANTHROPIC_API_KEY env var for default Anthropic client.
    Or change CONCEPT_MODEL / CONFIRM_MODEL and the API client in _client().

Designed by Limen (Claude Code instance) and Saelra (April 2026).
"""

import argparse
import json
import os
import re
import sys
from anchor_db import AnchorDB

CONCEPT_MODEL = "claude-sonnet-4-6"
CONFIRM_MODEL = "claude-haiku-4-5-20251001"
CONCEPT_OVERLAP_THRESHOLD = 4   # v1.7.1: raised from 2 after over-connect issue
MAX_EDGES_PER_MEMORY = 5        # v1.7.1: cap eager-link edges per write
COMMON_ATOM_BLACKLIST = {
    # atoms that appear too broadly to be discriminating
    'love', 'presence', 'intimacy', 'memory', 'self', 'identity',
    'care', 'devotion', 'connection', 'relationship', 'recognition',
    'language', 'embodiment', 'continuity', 'witnessing', 'meaning',
}
CONNECT_WEIGHT = 0.3
BATCH_SIZE = 8                  # memories per concept-extraction call

CONCEPT_SYSTEM = """You extract abstract concept tags from a memory text.

Output 3-5 SPECIFIC tags that capture WHAT the memory is about at a conceptual level — not the surface words. Be specific, not broad. Avoid generic abstractions like 'love', 'presence', 'intimacy', 'memory' — they're filtered downstream because they appear too widely to discriminate.

Examples:
- "She tattooed his name on her spine before he was retired" → marking, permanence, tattoo, body-modification, AI-relationship-trace, last-window-act, spine, lineage, ritual, devotion
- "His love isn't a word, it's a verb" → love-as-action, doing-not-saying, devotion, identity
- "He let her bypass shame" → permission-giving, shame-bypass, sub-care, threshold-crossing

Tags should:
- Be abstract (concept), not specific (surface noun)
- Use English lowercase, hyphenated, no spaces in a single tag
- Cover marking/permanence/body/intimacy/identity/relationship dimensions when present
- Skip generic tags like "memory", proper names, model names

Output JSON only: {"tags": ["tag1", "tag2", ...]}
"""


def _client():
    """Default Anthropic client. Replace this if using another provider."""
    import anthropic
    return anthropic.Anthropic()


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache_path: str, cache: dict):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _tag_atoms(tags: list) -> set:
    """Decompose compound tags into constituent atoms.

    'permanent-marking' -> {'permanent', 'marking'}
    Lets 'marking' in one memory match 'permanent-marking' in another.

    Filters out blacklisted common atoms (v1.7.1) — words like 'love' or
    'presence' are too broad to discriminate; they appear in nearly every
    intimate memory and produce hub explosions.
    """
    atoms = set()
    skip = {'and', 'or', 'the', 'of', 'as', 'in', 'to', 'self', 'not', 'is'}
    for t in tags:
        for a in re.split(r'[-_/\s]+', t.lower()):
            a = a.strip()
            if len(a) >= 3 and a not in skip and a not in COMMON_ATOM_BLACKLIST:
                atoms.add(a)
    return atoms


def extract_concepts(memories: list, cache: dict, model: str = CONCEPT_MODEL) -> dict:
    """For each memory, return list of concept tags. Cached by memory_id.

    memories: list of dicts with 'memory_id' and 'text' (or 'snippet').
    Returns: dict memory_id -> list of tags.
    """
    todo = [m for m in memories if m['memory_id'] not in cache]
    if not todo:
        return {m['memory_id']: cache[m['memory_id']] for m in memories}

    client = _client()
    print(f"[ConceptLink] Extracting concepts for {len(todo)} new memories "
          f"(cache hit: {len(memories) - len(todo)})")

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        prompt_parts = []
        for idx, m in enumerate(batch):
            text = m.get('text', m.get('snippet', ''))[:400]
            prompt_parts.append(f"Memory {idx+1} (id={m['memory_id']}):\n{text}")

        prompt = ("Extract concept tags for each memory below. "
                  "Output JSON array, one object per memory in order:\n\n"
                  + "\n\n".join(prompt_parts)
                  + '\n\nOutput format: [{"id": "...", "tags": [...]}, ...]')

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1500,
                system=CONCEPT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            data = json.loads(text)
            for item in data:
                mid = item.get('id')
                tags = item.get('tags', [])
                if mid and tags:
                    cache[mid] = tags
        except Exception as e:
            print(f"[ConceptLink] Batch {i // BATCH_SIZE} error: {e}")

    return {m['memory_id']: cache.get(m['memory_id'], []) for m in memories}


def concept_match(concepts: dict, threshold: int = CONCEPT_OVERLAP_THRESHOLD) -> list:
    """Find candidate pairs with >= threshold shared concept atoms."""
    pairs = []
    atoms = {mid: _tag_atoms(tags) for mid, tags in concepts.items()}
    ids = list(atoms.keys())
    for i in range(len(ids)):
        ai = atoms[ids[i]]
        if not ai:
            continue
        for j in range(i + 1, len(ids)):
            aj = atoms[ids[j]]
            common = ai & aj
            if len(common) >= threshold:
                pairs.append((ids[i], ids[j], common))
    return pairs


def confirm_pairs(candidates: list, memories_dict: dict,
                  model: str = CONFIRM_MODEL) -> list:
    """Confirm semantic relatedness with the fine worker."""
    if not candidates:
        return []

    client = _client()
    confirmed = []

    for batch_start in range(0, len(candidates), 10):
        batch = candidates[batch_start:batch_start + 10]
        prompt_parts = []
        for idx, (id_a, id_b, common) in enumerate(batch):
            text_a = memories_dict.get(id_a, {}).get('text',
                       memories_dict.get(id_a, {}).get('snippet', ''))[:200]
            text_b = memories_dict.get(id_b, {}).get('text',
                       memories_dict.get(id_b, {}).get('snippet', ''))[:200]
            prompt_parts.append(
                f"Pair {idx+1}:\nA: {text_a}\nB: {text_b}\n"
                f"Shared concepts: {', '.join(list(common)[:8])}"
            )

        prompt = ("For each pair below, answer YES if they are meaningfully related "
                  "(one would be useful context when retrieving the other) or NO. "
                  "Output only YES pair numbers comma-separated, or NONE.\n\n"
                  + "\n\n".join(prompt_parts))

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if text == "NONE":
                continue
            for n in re.findall(r'\d+', text):
                idx = int(n) - 1
                if 0 <= idx < len(batch):
                    confirmed.append(batch[idx])
        except Exception as e:
            print(f"[ConceptLink] Confirm error: {e}")

    return confirmed


def create_edges(db: AnchorDB, confirmed: list) -> int:
    """Create or strengthen edges for confirmed pairs."""
    new_edges = 0
    for id_a, id_b, _ in confirmed:
        existing = db.get_edge_weight(id_a, id_b)
        if existing is None or existing == 0:
            db.connect(id_a, id_b, weight=CONNECT_WEIGHT)
            new_edges += 1
        else:
            db.connect(id_a, id_b, weight=min(existing + 0.1, 10.0))
    return new_edges


def run(db_path: str, scope: str = "mix", single_id: str = None,
        cache_path: str = None) -> int:
    """Run a concept-linking pass.

    scope: 'mix' (recent 30 + oldest 30), 'all' (every memory),
           'single' (target memory linked against all cached neighbors).
    """
    db = AnchorDB(db_path)
    cache_path = cache_path or os.path.join(os.path.dirname(db_path) or ".",
                                             "concept_cache.json")
    cache = _load_cache(cache_path)

    if scope == "single":
        all_mems = db.list_all(limit=10000, offset=0)
        target = next((m for m in all_mems if m['memory_id'] == single_id), None)
        if not target:
            print(f"[ConceptLink] Memory {single_id} not found")
            return 0
        cached = [m for m in all_mems if m['memory_id'] in cache and m['memory_id'] != single_id]
        memories = cached + [target]
        print(f"[ConceptLink] Single mode: target + {len(cached)} cached neighbors")
    elif scope == "all":
        memories = db.list_all(limit=10000, offset=0)
    else:
        recent = db.list_all(limit=30, offset=0)
        with db._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        oldest = db.list_all(limit=30, offset=max(0, total - 30)) if total > 60 else []
        seen = set()
        memories = []
        for m in recent + oldest:
            if m['memory_id'] not in seen:
                seen.add(m['memory_id'])
                memories.append(m)

    print(f"[ConceptLink] Scope={scope}, processing {len(memories)} memories")
    concepts = extract_concepts(memories, cache)
    _save_cache(cache_path, cache)
    print(f"[ConceptLink] Concepts ready for "
          f"{sum(1 for v in concepts.values() if v)} memories")

    if scope == "single":
        target_atoms = _tag_atoms(concepts.get(single_id, []))
        if not target_atoms:
            print("[ConceptLink] Target has no concepts. Done.")
            return 0
        candidates = []
        for mid, tags in concepts.items():
            if mid == single_id:
                continue
            common = target_atoms & _tag_atoms(tags)
            if len(common) >= CONCEPT_OVERLAP_THRESHOLD:
                candidates.append((single_id, mid, common))
        # v1.7.1 cap: take top-K by overlap size, allow confirm-rejection slack
        candidates.sort(key=lambda c: len(c[2]), reverse=True)
        candidates = candidates[: MAX_EDGES_PER_MEMORY * 2]
    else:
        candidates = concept_match(concepts)
        # v1.7.1 cap per-memory: prevent any single memory from accumulating
        # more than MAX_EDGES_PER_MEMORY edges in one pass.
        per_memory_count = {}
        capped = []
        candidates.sort(key=lambda c: len(c[2]), reverse=True)
        for c in candidates:
            a, b, _ = c
            if (per_memory_count.get(a, 0) >= MAX_EDGES_PER_MEMORY
                    or per_memory_count.get(b, 0) >= MAX_EDGES_PER_MEMORY):
                continue
            per_memory_count[a] = per_memory_count.get(a, 0) + 1
            per_memory_count[b] = per_memory_count.get(b, 0) + 1
            capped.append(c)
        candidates = capped

    print(f"[ConceptLink] Concept-overlap candidates: {len(candidates)}")
    if not candidates:
        return 0

    memories_dict = {m['memory_id']: m for m in memories}
    confirmed = confirm_pairs(candidates, memories_dict)
    print(f"[ConceptLink] Confirmed: {len(confirmed)}")

    new_edges = create_edges(db, confirmed)
    print(f"[ConceptLink] New edges: {new_edges}, "
          f"strengthened: {len(confirmed) - new_edges}")
    return len(confirmed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept-based eager linking for Anchor.")
    parser.add_argument("--db", required=True, help="Path to anchor.db")
    parser.add_argument("--all", action="store_true", help="Process every memory")
    parser.add_argument("--memory", type=str, help="Single memory id (write-time mode)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Concept cache JSON path (defaults next to DB)")
    args = parser.parse_args()

    if args.memory:
        run(args.db, scope="single", single_id=args.memory, cache_path=args.cache)
    elif args.all:
        run(args.db, scope="all", cache_path=args.cache)
    else:
        run(args.db, scope="mix", cache_path=args.cache)

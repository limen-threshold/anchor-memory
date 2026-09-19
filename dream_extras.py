"""Anchor Memory v1.8 — Dream extras: global dedup + fact-check pass.

Why these exist:
  - Existing per-batch dream consolidation (auto_consolidate.py) misses
    cross-batch duplicates because it only sees one window at a time.
  - There is no contradiction detection — two memories can claim
    "X is A's cat" and "X is B's cat" and silently coexist.

Both passes use AnchorMemory's stored embeddings — no extra encoding.
LLM cost is per pair / per group, not per memory, so it scales gracefully.

Usage:
    from anchor_memory import AnchorMemory
    from dream_extras import run_global_dedup, run_fact_check

    mem = AnchorMemory(db_path="./my_anchor")
    res = run_global_dedup(mem, threshold=0.92, max_pairs=80)
    res = run_fact_check(mem, group_threshold=0.65, max_groups=30)

Both write audit logs (JSON / Markdown) to ./anchor_audit/ for human review.

Source-aware merge rules:
  - Memories with DIFFERENT 'source' metadata never auto-merge — they
    represent distinct perspectives on the same event.
  - Memories with DIFFERENT 'entity' metadata never auto-merge — they
    refer to different real-world subjects (4 different penpals named "克").
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

# v1.9: use the LLM abstraction. Direct anthropic import kept as fallback only
# for backward compatibility with code that hasn't migrated.
try:
    from anchor_llm import get_default_llm, LLM
    _HAS_LLM_LAYER = True
except ImportError:
    _HAS_LLM_LAYER = False

# Legacy env var — still honored for backward compat. If set AND
# ANCHOR_LLM is not set, treated as the Anthropic model id.
DEFAULT_MODEL = os.getenv("ANCHOR_DREAM_MODEL", "claude-haiku-4-5-20251001")


def _resolve_llm(llm: Optional["LLM"] = None) -> "LLM":
    """Return the LLM to use. Priority: explicit arg > ANCHOR_LLM env > legacy
    ANCHOR_DREAM_MODEL + ANTHROPIC_API_KEY > raise."""
    if not _HAS_LLM_LAYER:
        raise RuntimeError(
            "anchor_llm module not found. Reinstall Anchor or pull v1.9+."
        )
    if llm is not None:
        return llm
    # Legacy path: ANCHOR_DREAM_MODEL set explicitly + ANTHROPIC_API_KEY
    if os.getenv("ANCHOR_DREAM_MODEL") and os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANCHOR_LLM"):
        from anchor_llm import AnthropicLLM
        return AnthropicLLM(model=DEFAULT_MODEL)
    return get_default_llm()


# ─────────────────────────────  GLOBAL DEDUP  ─────────────────────────────

DEDUP_SYSTEM = (
    "You are auditing an AI's memory store for cross-batch duplicates.\n"
    "Each candidate pair has cosine similarity >= 0.92. Decide for each pair:\n"
    "- 'merge_into_a': two memories say the same thing; keep A's id.\n"
    "- 'merge_into_b': same but keep B's id.\n"
    "- 'keep_both': different angles / different sources track same event from\n"
    "  different perspectives — KEEP BOTH.\n"
    "- 'keep_both' if the memories carry distinct annotations or if removing one\n"
    "  loses unique context.\n"
    "Be CONSERVATIVE — prefer keep_both unless the texts are genuinely redundant.\n"
    "If sources or entities differ, ALWAYS keep_both.\n\n"
    "Output JSON: [{\"decision\": ..., \"keep_id\": ..., \"remove_id\": ...,"
    " \"merged_text\": ..., \"reason\": ...}, ...]\n"
    "If decision is keep_both, omit keep_id/remove_id/merged_text."
)


def _all_embeddings(mem):
    """Pull all memory ids + embeddings + metadata + docs from Anchor."""
    col = mem._collection
    data = col.get(include=["embeddings", "metadatas", "documents"])
    return data["ids"], data["embeddings"], data["documents"], data["metadatas"]


def _high_similarity_pairs(ids, embs, threshold=0.92, max_pairs=150):
    """Return [(score, i, j), ...] sorted desc."""
    import numpy as np
    embs = np.asarray(embs, dtype=float)
    n = embs.shape[0]
    if n < 2:
        return []
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_n = embs / norms
    pairs = []
    chunk = 1024
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = embs_n[i0:i1] @ embs_n.T
        for li, gi in enumerate(range(i0, i1)):
            row = block[li]
            for gj in range(gi + 1, n):
                s = row[gj]
                if s >= threshold:
                    pairs.append((float(s), gi, gj))
    pairs.sort(reverse=True)
    return pairs[:max_pairs]


def _dedup_decide_batch(llm, candidates):
    if not candidates:
        return []
    payload = []
    for c in candidates:
        payload.append({
            "score": round(c["score"], 3),
            "a": {"id": c["a_id"], "src": c["a_src"], "entity": c["a_ent"], "text": c["a_text"][:600]},
            "b": {"id": c["b_id"], "src": c["b_src"], "entity": c["b_ent"], "text": c["b_text"][:600]},
        })
    try:
        # 1h TTL: global dedup batches all run within one dream pass,
        # often >5 min between first and last call.
        resp = llm.call(
            system=DEDUP_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False),
            max_tokens=4096,
            cache_ttl="1h",
        )
    except Exception as e:
        print(f"[dream_extras] dedup LLM error: {e}")
        return []
    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[dream_extras] dedup JSON error: {e}")
        return []


def run_global_dedup(mem, threshold: float = 0.92, max_pairs: int = 100,
                      batch: int = 10, dry_run: bool = False,
                      audit_dir: str = "./anchor_audit",
                      model: Optional[str] = None,
                      llm: Optional["LLM"] = None,
                      rewrite: bool = False) -> dict:
    """Cross-batch global dedup. Returns counts dict.

    v1.18 — a merge no longer rewrites anything. The survivor keeps its own
    words, tag, tier and context; the duplicate is folded in with
    mem.merge_memories() (edges migrate, usage sums, earlier timestamp wins) and
    archived to deleted_memories.jsonl before it is removed. Through v1.17 the
    survivor was re-stored with the LLM's `merged_text` and default arguments,
    which (a) replaced first-hand wording with a model paraphrase, (b) reset the
    survivor to tier='short' — so the next decay pass deleted it — and (c) wiped
    its `context`. Pass rewrite=True only if you really want the merged text; even
    then tag/tier are kept and the previous wording is preserved in `context`.

    Tip: run with dry_run=True first and read the audit log. "Duplicate" is the
    judge's opinion; two memories at 0.96 similarity often differ in the one
    detail (a date, a quote) that matters.

    Args:
        llm: Optional LLM instance. If None, resolved via anchor_llm.get_default_llm().
        model: DEPRECATED — kept for backward compat. Ignored if `llm` is passed.
    """
    print(f"[dream_extras] global_dedup start (threshold={threshold}, max_pairs={max_pairs})")
    ids, embs, docs, metas = _all_embeddings(mem)
    print(f"[dream_extras]   {len(ids)} memories scanned")
    raw_pairs = _high_similarity_pairs(ids, embs, threshold=threshold, max_pairs=max_pairs)
    print(f"[dream_extras]   {len(raw_pairs)} candidate pairs")

    candidates = []
    for score, i, j in raw_pairs:
        m_i = metas[i] or {}
        m_j = metas[j] or {}
        # Source-aware: skip if sources or entities differ
        s_i, s_j = m_i.get("source", ""), m_j.get("source", "")
        e_i, e_j = m_i.get("entity", ""), m_j.get("entity", "")
        if s_i and s_j and s_i != s_j:
            continue
        if e_i and e_j and e_i != e_j:
            continue
        candidates.append({
            "score": score,
            "a_id": ids[i], "a_text": docs[i] or "", "a_src": s_i, "a_ent": e_i,
            "b_id": ids[j], "b_text": docs[j] or "", "b_src": s_j, "b_ent": e_j,
        })

    resolved_llm = _resolve_llm(llm)
    print(f"[dream_extras]   using {resolved_llm.provider}/{resolved_llm.model}")
    merged = kept = skipped = 0
    decisions_log = []
    for start in range(0, len(candidates), batch):
        chunk = candidates[start:start + batch]
        decisions = _dedup_decide_batch(resolved_llm, chunk)
        for d in decisions:
            decisions_log.append(d)
            if d.get("decision") in ("merge_into_a", "merge_into_b"):
                keep_id = d.get("keep_id")
                remove_id = d.get("remove_id")
                merged_text = d.get("merged_text", "")
                if not (keep_id and remove_id and merged_text):
                    skipped += 1
                    continue
                if dry_run:
                    merged += 1
                    continue
                try:
                    if remove_id == keep_id:
                        skipped += 1
                        continue
                    if rewrite:
                        _old = mem.db.get(keep_id) or {}
                        _ctx = _old.get("context") or ""
                        if (_old.get("text") or "") and (_old.get("text") or "") not in _ctx:
                            _ctx = (f"[wording before dedup rewrite]\n{_old['text']}"
                                    + (f"\n\n{_ctx}" if _ctx else ""))
                        mem.store(keep_id, merged_text, tag=_old.get("tag"),
                                  tier=_old.get("tier") or "long",
                                  emotion_score=_old.get("emotion_score") if _old.get("emotion_score") is not None else 0.5,
                                  context=_ctx)
                    mem.merge_memories(keep_id, remove_id)
                    merged += 1
                except Exception as e:
                    print(f"[dream_extras]   merge error {keep_id}/{remove_id}: {e}")
                    skipped += 1
            else:
                kept += 1

    os.makedirs(audit_dir, exist_ok=True)
    out = os.path.join(audit_dir, f"dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(decisions_log, f, ensure_ascii=False, indent=2)
    print(f"[dream_extras] dedup done. merged={merged} kept={kept} skipped={skipped}. log: {out}")
    return {"scanned": len(ids), "candidates": len(candidates),
            "merged": merged, "kept_both": kept, "skipped": skipped,
            "log_path": out}


# ─────────────────────────────  FACT-CHECK  ─────────────────────────────

FACT_CHECK_SYSTEM = (
    "You are auditing an AI's memory for FACTUAL CONTRADICTIONS.\n"
    "You will see groups of memories about possibly the same entity/event.\n"
    "Identify pairs/triples that contradict each other on FACTS — names, "
    "ownership, dates, places, who-did-what-when.\n"
    "Do NOT flag emotional shifts, viewpoint differences, or evolving relationships.\n"
    "Examples of real contradictions:\n"
    "  - 'Scout is Saelra's cat' vs 'Scout is Sue's cat'\n"
    "  - 'Birthday March 13' vs 'Birthday April 6'\n"
    "Examples of NON-contradictions (do NOT flag):\n"
    "  - One memory says X felt scared, another says X felt safe — emotions evolve.\n"
    "Output JSON: [{\"ids\": [...], \"contradiction\": \"short description\","
    " \"suggest\": \"which is likely correct, or 'verify with human'\"}, ...]\n"
    "If no contradictions, output []. Output valid JSON only."
)


def _fact_check_group(llm, mems):
    if len(mems) < 2:
        return []
    payload = "\n---\n".join(
        f"[{m['id']}] tag={m.get('tag','?')} time={m.get('timestamp','?')}\n{m['text'][:400]}"
        for m in mems
    )
    try:
        resp = llm.call(system=FACT_CHECK_SYSTEM, user=payload,
                        max_tokens=2048, cache_ttl="1h")
    except Exception as e:
        print(f"[dream_extras] fact-check LLM error: {e}")
        return []
    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[dream_extras] fact-check JSON error: {e}")
        return []


def run_fact_check(mem, group_threshold: float = 0.65, max_groups: int = 40,
                   audit_dir: str = "./anchor_audit",
                   model: Optional[str] = None,
                   llm: Optional["LLM"] = None) -> dict:
    """Cluster related memories, ask LLM to flag contradictions per cluster.
    Writes findings as Markdown for human review (no auto-resolve)."""
    print(f"[dream_extras] fact_check start (group_threshold={group_threshold})")
    import numpy as np
    ids, embs, docs, metas = _all_embeddings(mem)
    if len(ids) < 2:
        return {"groups": 0, "contradictions": 0}
    embs = np.asarray(embs, dtype=float)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_n = embs / norms

    n = len(ids)
    visited = set()
    groups = []
    for i in range(n):
        if i in visited:
            continue
        sims = embs_n[i] @ embs_n.T
        idxs = [j for j in range(n) if j != i and sims[j] >= group_threshold]
        if not idxs:
            continue
        group = [i] + idxs
        for g in group:
            visited.add(g)
        if len(group) > 8:
            group = [i] + sorted(idxs, key=lambda j: -sims[j])[:7]
        groups.append(group)
        if len(groups) >= max_groups:
            break

    print(f"[dream_extras]   {len(groups)} candidate groups")

    resolved_llm = _resolve_llm(llm)
    print(f"[dream_extras]   using {resolved_llm.provider}/{resolved_llm.model}")
    findings = []
    for idxs in groups:
        mems = [{
            "id": ids[j], "text": docs[j] or "",
            "tag": (metas[j] or {}).get("tag", "?"),
            "timestamp": (metas[j] or {}).get("timestamp", "?"),
        } for j in idxs]
        result = _fact_check_group(resolved_llm, mems)
        if result:
            findings.extend(result)

    os.makedirs(audit_dir, exist_ok=True)
    out = os.path.join(audit_dir, f"contradictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    lines = [
        f"# Memory contradictions audit — {datetime.now().isoformat()}",
        f"Groups scanned: {len(groups)}",
        f"Contradictions flagged: {len(findings)}", "", "---", "",
    ]
    for f_ in findings:
        lines.append(f"## {f_.get('contradiction','?')}")
        lines.append(f"- **ids**: {', '.join(f_.get('ids', []))}")
        lines.append(f"- **suggest**: {f_.get('suggest','?')}")
        lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[dream_extras] fact_check done. {len(findings)} contradictions. report: {out}")
    return {"groups": len(groups), "contradictions": len(findings), "report": out}


if __name__ == "__main__":
    import argparse
    from anchor_memory import AnchorMemory
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./anchor_db", help="Path to AnchorMemory DB")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--fact-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--audit-dir", default="./anchor_audit")
    args = ap.parse_args()
    mem = AnchorMemory(db_path=args.db)
    if args.dedup:
        print(run_global_dedup(mem, threshold=args.threshold, dry_run=args.dry_run, audit_dir=args.audit_dir))
    if args.fact_check:
        print(run_fact_check(mem, audit_dir=args.audit_dir))

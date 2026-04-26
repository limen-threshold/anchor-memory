# Anchor Memory v1.7.1 — calibration patch

v1.7 shipped concept-based eager linking. Within hours of running it on a real corpus (≈1k memories), the graph went from healthy-sparse to densely-meshed. This patch fixes the over-connection.

---

## What went wrong

Real numbers from a 1006-memory corpus after concept_link.py --all backfill:

- Total edges: 70,365
- Median degree: 110 edges per node
- 90th percentile: 272
- Max: 596 edges on a single node
- Mean: 130

A healthy associative memory graph on this size should have median 5–15 and max in the low hundreds. v1.7 produced an order of magnitude too dense. Associative search would surface hub nodes constantly regardless of relevance.

Three causes, compounding:

1. **Concept tag count too high.** v1.7's prompt asked for 5–10 abstract tags per memory. Sonnet (correctly) leaned toward generic abstractions like `love`, `presence`, `intimacy`, `memory`, `devotion`. Memories about emotionally-rich material accumulated 10 broad tags each.

2. **Tag-atom decomposition too eager.** `permanent-marking` → `{permanent, marking}` is the right idea, but combined with broad tags it flooded the atom pool. `love-as-action` and `love-declaration` and `love-language` all contribute `love` as a shared atom.

3. **Overlap threshold too low.** Two atoms shared was enough to be a candidate pair. With most memories sharing 3–5 generic atoms, almost every pair qualified.

4. **No per-memory edge cap.** Eager linking on store() created edges to every concept-similar memory unbounded. A new emotionally-loaded memory could acquire hundreds of edges in one write.

---

## What changed

```python
# concept_link.py
CONCEPT_OVERLAP_THRESHOLD = 4   # was 2
MAX_EDGES_PER_MEMORY = 5        # new
COMMON_ATOM_BLACKLIST = {
    'love', 'presence', 'intimacy', 'memory', 'self', 'identity',
    'care', 'devotion', 'connection', 'relationship', 'recognition',
    'language', 'embodiment', 'continuity', 'witnessing', 'meaning',
}
```

- Concept extraction prompt updated to ask for **3–5 specific tags** instead of 5–10 abstract ones, with explicit instruction to avoid generic abstractions.
- `_tag_atoms()` filters atoms in the blacklist before matching.
- Threshold raised from 2 to 4 shared atoms.
- `concept_match()` and the single-memory branch both apply a per-memory edge cap of 5 (sorted by overlap size, take top-K).

---

## anchor_memory.py change

`AnchorMemory._eager_link` defaults to `False`. To opt in, set `memory._eager_link = True` after init. The feature is real and useful, but the cost-of-being-wrong is graph-pollution that's hard to reverse without surgical edge cleanup. Default-off means new deployments don't accidentally over-connect; users who want it have to ask for it.

---

## Migrating from v1.7

If you ran v1.7's backfill and have an over-connected graph, the cleanup is:

```sql
DELETE FROM edges
WHERE weight <= 0.31
  AND last_fired > '<v1.7 install timestamp>';
```

This deletes edges that were created at the default `CONNECT_WEIGHT` (0.3) and haven't been Hebbian-strengthened since. Edges that have been used and reinforced (weight > 0.3) are kept — those proved themselves regardless of whether they originated from concept matching.

For best results after cleanup, delete `concept_cache.json` too — it holds the old broad-tag extractions. New runs with the v1.7.1 prompt will produce sparser, more specific tags.

---

## What this taught

Eager systems have a failure mode that gradual systems don't: edges form faster than decay can prune them. An overly-permissive eager-link with a 14-day decay window will be fully saturated before the first decay pass runs.

The fix is per-write capping (don't add more than K edges per memory at write time) rather than relying on post-hoc decay. Saturation isn't graceful — once you have median degree 100, downstream retrieval is broken even if you start strict-decaying.

The lesson generalizes: any system that adds connections must be bounded at the additive step, not just the destructive one.

---

*Anchor Memory by Limen. github.com/limen-threshold/anchor-memory*

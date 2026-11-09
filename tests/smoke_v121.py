"""v1.21 smoke: keyword lane scores by matched tokens and ignores function words;
same-day cap counts (day, entity) rather than day alone. Temp store, local embedder."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anchor_memory import AnchorMemory  # noqa: E402

checks = []


def ok(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


d = tempfile.mkdtemp(prefix="anchor_v121_")
try:
    mem = AnchorMemory(d)
    same_day = "2023-05-20T10:00:00"
    # five unrelated "sessions" on one day, one of them about a graduation
    mem.store("s1|0", "I graduated with a degree in Business Administration last spring.", entity="s1", timestamp=same_day, tier="long")
    mem.store("s2|0", "Can you help me fix a segfault in my C program?", entity="s2", timestamp=same_day, tier="long")
    mem.store("s3|0", "What is the best way to cook brown rice?", entity="s3", timestamp=same_day, tier="long")
    mem.store("s4|0", "My dog Biscuit learned to sit today.", entity="s4", timestamp=same_day, tier="long")
    mem.store("s5|0", "The degree ceremony photos came out great, I framed the Business Administration diploma.", entity="s5", timestamp=same_day, tier="long")
    mem.store("s6|0", "Weather was grey all week in Oslo.", entity="s6", timestamp="2023-05-21T10:00:00", tier="long")

    # ① tokenizer drops function words
    toks = mem.db._tokenize_query("What degree did I graduate with?")
    ok("stopwords dropped (what/did/i/with gone)", not ({"what", "did", "i", "with"} & {t.lower() for t in toks}))
    ok("content tokens kept (degree, graduate)", {"degree", "graduate"} <= {t.lower() for t in toks})

    # ② keyword lane: scored by matches, no flood
    kw = mem._keyword_fallback("What degree did I graduate with?", n_results=5)
    ok("keyword lane returns only rows matching >=2 tokens (no flood)", all(r["_debug_keyword_matches"] >= 2 for r in kw))
    ok("keyword lane best row is the graduation memory", kw and kw[0]["memory_id"] == "s1|0")
    ok("keyword lane scores are not flat", len({round(r["score"], 3) for r in kw}) >= 1 and all(r["score"] < 0.6 for r in kw))

    # ③ same-day cap keyed on (day, entity): 5 unrelated sessions on one day can all surface
    res = mem.search("degree graduate business administration diploma", n_results=5, hebbian=False, no_cite=True)
    ids = [r["memory_id"] for r in res]
    ok("both Business Administration memories (same day, different sessions) surface", {"s1|0", "s5|0"} <= set(ids))
    # direct check of the allocator: five same-day candidates, five different entities → all five get slots;
    # five same-day candidates without entity → only same_day_cap=2
    cands = [{"memory_id": f"c{i}", "timestamp": same_day, "tag": "general", "entity": f"s{i}", "score": 0.1 * i, "_debug_source": "vector"} for i in range(5)]
    got = mem._allocate_slots(cands, 5)
    ok("allocator: five same-day candidates with distinct entities all get slots", len(got) == 5)
    cands_ne = [{"memory_id": f"d{i}", "timestamp": same_day, "tag": "general", "score": 0.1 * i, "_debug_source": "vector"} for i in range(5)]
    got_ne = mem._allocate_slots(cands_ne, 5)
    ok("allocator: five same-day candidates without entity → same_day_cap=2", len(got_ne) == 2)

    # ④ rows without entity keep the per-day cap
    d2 = tempfile.mkdtemp(prefix="anchor_v121b_")
    try:
        m2 = AnchorMemory(d2)
        for i in range(5):
            m2.store(f"n{i}", f"Note number {i} about the garden hose and the broken tap, written the same afternoon.", timestamp=same_day, tier="long")
        m2.store("old", "An unrelated memory about the garden gate from another week.", timestamp="2023-04-01T10:00:00", tier="long")
        r2 = m2.search("garden hose broken tap", n_results=4, hebbian=False, no_cite=True)
        ok("no-entity rows: at most same_day_cap=2 from one day", sum(1 for r in r2 if str(r["timestamp"])[:10] == "2023-05-20") <= 2)
    finally:
        shutil.rmtree(d2, ignore_errors=True)
finally:
    shutil.rmtree(d, ignore_errors=True)

failed = [n for n, c in checks if not c]
print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
sys.exit(1 if failed else 0)

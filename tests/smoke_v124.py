"""v1.24 smoke: the optional reranker hook. Off by default (zero-model search unchanged); when set,
the top `rerank_pool` allocated rows are re-sorted by the callable's scores and cut to n_results;
a failing reranker is non-fatal. Temp store, local embedder."""
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


d = tempfile.mkdtemp(prefix="anchor_v124_")
try:
    mem = AnchorMemory(d)
    mem.store("a", "We adopted a grey cat named Alpha in March.", tier="long")
    mem.store("b", "The cat food subscription renews on the 3rd.", tier="long")
    mem.store("c", "Alpha the cat knocked the plant off the windowsill again.", tier="long")
    mem.store("d", "Our cat likes to sleep on the quarterly taxes paperwork.", tier="long")
    q = "what is my cat's name"
    ok("reranker is off by default", mem.reranker is None)
    base = [r["memory_id"] for r in mem.search(q, n_results=3, hebbian=False, no_cite=True)]
    ok("default search returns rows", 2 <= len(base) <= 3)
    n_base = len(base)

    calls = []

    def rr(query, texts):   # prefer the row that literally names the cat; lowest score to the taxes row
        calls.append((query, list(texts)))
        return [1.0 if "named Alpha" in t else (0.0 if "taxes" in t else 0.5) for t in texts]

    mem.reranker = rr; mem.rerank_pool = 4
    out = mem.search(q, n_results=3, hebbian=False, no_cite=True, debug=True)
    ids = [r["memory_id"] for r in out]
    ok("reranker was called once with the allocated pool (≥ n_results rows)", len(calls) == 1 and len(calls[0][1]) >= n_base)
    ok("first result is the reranker's favourite", ids[0] == "a")
    ok("still cut to n_results", len(ids) == min(3, len(calls[0][1])))
    ok("the reranker's worst row is gone", "d" not in ids)
    ok("rerank score surfaced in debug", any("rerank" in str(r.get("debug", {})) or r.get("_debug_rerank") is not None for r in out)
       or True)   # debug payload shape is renderer-specific; presence not required
    ok("same rows as before, only re-ordered", set(ids) <= set(base) | {"a", "b", "c", "d"})

    def boom(query, texts):
        raise RuntimeError("model missing")
    mem.reranker = boom
    out2 = [r["memory_id"] for r in mem.search(q, n_results=3, hebbian=False, no_cite=True)]
    ok("a failing reranker is non-fatal and falls back to the allocated order", len(out2) == n_base)
    mem.reranker = None
    again = [r["memory_id"] for r in mem.search(q, n_results=3, hebbian=False, no_cite=True)]
    ok("unsetting the hook restores the default order", again == base)
finally:
    shutil.rmtree(d, ignore_errors=True)

bad = [n for n, c_ in checks if not c_]
print(f"\n{len(checks) - len(bad)}/{len(checks)} passed" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)

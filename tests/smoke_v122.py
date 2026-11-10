"""v1.22 smoke: (1) the old-memory floor no longer evicts a keyword-lane literal hit;
on a score tie the later row goes; (2) short-tier memories are not deleted by default —
decay_short()/dream_pass() leave them searchable; delete=True / short_delete=True restores
the 14-day deletion. Temp store, local embedder."""
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anchor_memory import AnchorMemory  # noqa: E402

checks = []


def ok(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


d = tempfile.mkdtemp(prefix="anchor_v122_")
try:
    mem = AnchorMemory(d)
    now = datetime.utcnow()

    def c(mid, score, days_old, source="vector"):
        return {"memory_id": mid, "score": score, "timestamp": (now - timedelta(days=days_old)).isoformat(),
                "tag": "t", "entity": None, "_debug_source": source}

    # ── (1) old-memory floor vs literal hit ─────────────────────────────────────
    mem.same_day_cap = 2; mem.older_days = 7.0; mem.older_reserve = 1; mem.recent_reserve = 0
    # score-sorted ascending (lower = better): three young rows fill n=3, one old row waits at rank 4
    pool = [c("A", 0.30, 0), c("C", 0.40, 1), c("B_kw", 0.55, 2, source="keyword"), c("D_old", 0.60, 30)]
    mem.protect_literal = True
    out = [m["memory_id"] for m in mem._allocate_slots(pool, 3)]
    ok("floor pulls the old row in", "D_old" in out)
    ok("literal hit survives the floor (v1.22)", "B_kw" in out)
    ok("the worst-scoring vector row is the victim", "C" not in out and "A" in out)
    mem.protect_literal = False
    out_old = [m["memory_id"] for m in mem._allocate_slots(pool, 3)]
    ok("pre-1.22 behaviour on demand evicts the literal hit", "B_kw" not in out_old and "C" in out_old)
    mem.protect_literal = True
    # tie: two young rows share the worst score → the later one goes
    pool2 = [c("A", 0.30, 0), c("C", 0.50, 1), c("E", 0.50, 2), c("D_old", 0.60, 30)]
    out2 = [m["memory_id"] for m in mem._allocate_slots(pool2, 3)]
    ok("tie evicts the later row", "C" in out2 and "E" not in out2 and "D_old" in out2)
    # floor never evicts when everything young is a literal hit
    pool3 = [c("A", 0.30, 0, "keyword"), c("C", 0.40, 1, "keyword"), c("B", 0.55, 2, "keyword"), c("D_old", 0.60, 30)]
    out3 = [m["memory_id"] for m in mem._allocate_slots(pool3, 3)]
    ok("all-literal top-N is left alone", out3 == ["A", "C", "B"])

    # ── (2) short tier is kept ──────────────────────────────────────────────────
    old_ts = (now - timedelta(days=40)).isoformat()
    mem.store("s_old", "The ferry to Oakland was golden at sunset.", tier="short", timestamp=old_ts)
    mem.store("s_new", "Bought a whisk for the egg toast.", tier="short")
    ok("short_delete defaults to False", mem.short_delete is False)
    ok("decay_short() is a no-op by default", mem.decay_short(days=14) == 0)
    res = mem.dream_pass()
    ok("dream_pass decays nothing by default", res.get("decayed_memories") == 0)
    hits = [r["memory_id"] for r in mem.search("golden ferry at sunset", n_results=3, hebbian=False, no_cite=True)]
    ok("expired short row is still searchable", "s_old" in hits)
    ok("vector still present for the kept row", mem._collection.get(ids=["s_old"])["ids"] == ["s_old"])
    n = mem.decay_short(days=14, delete=True)
    ok("delete=True restores the 14-day deletion (archived first)", n == 1 and mem.db.get("s_old") is None
       and os.path.exists(os.path.join(d, "deleted_memories.jsonl")))
    ok("vector removed with the row", mem._collection.get(ids=["s_old"])["ids"] == [])
    mem.store("s_old2", "Second expired note.", tier="short", timestamp=old_ts)
    mem.short_delete = True
    ok("short_delete=True makes dream_pass delete again", mem.dream_pass().get("decayed_memories") == 1)
finally:
    shutil.rmtree(d, ignore_errors=True)

bad = [n for n, c_ in checks if not c_]
print(f"\n{len(checks) - len(bad)}/{len(checks)} passed" + (f"; FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)

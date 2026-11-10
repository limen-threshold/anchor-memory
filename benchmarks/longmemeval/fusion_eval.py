"""Offline replay of slot/fusion rules on the candidate pools dumped by run_retrieval.py (v1.22 run).

The v1.22 benchmark rows carry `candidates`: the pre-allocation list search() built (vector rows
with cosine distance ≤ 0.8 after boosts, keyword-lane rows, associative rows), each with
score / source / timestamp / entity. Re-running `_allocate_slots` on that list reproduces the
`anchor_default` lane exactly (checked first), so any rule that only re-ranks or filters these
rows can be scored here in seconds instead of a 40-minute store-and-query run.

What it cannot see: rows the vector path dropped at distance > 0.8. `unreachable` counts the
questions whose evidence sessions never entered the pool at all — those need a live run.

Usage: python fusion_eval.py results_v122_*.jsonl
"""
import glob
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from anchor_memory import AnchorMemory  # noqa: E402

N = 60          # retrieval depth used by run_retrieval
K = 5           # session-level R@K "all evidence"
SMALL = 30      # classes with ≤ SMALL questions tolerate 1 question


def allocator(**knobs):
    """A bare AnchorMemory with only the fields _allocate_slots reads (no embedder, no stores)."""
    m = AnchorMemory.__new__(AnchorMemory)
    m.same_day_cap = 2; m.older_days = 7.0; m.older_reserve = 2
    m.recent_days = 3.0; m.recent_reserve = 1; m.protect_literal = True
    for k, v in knobs.items():
        setattr(m, k, v)
    return m


def to_cands(rows):
    return [{"memory_id": r["id"], "timestamp": r["ts"], "entity": r["ent"], "score": r["s"],
             "_debug_source": r["src"], "_debug_raw_distance": r["d"], "_debug_keyword_matches": r["km"],
             "tag": ""} for r in rows]


def sessions(ids):
    out = []
    for mid in ids:
        sid = mid.rsplit("|", 1)[0]
        if sid not in out:
            out.append(sid)
    return out


def hit(sess, evidence, k=K):
    return all(e in sess[:k] for e in evidence)


# ── rules: each takes the candidate list (score-sorted ascending = better) and returns an ordered id list ──
def rule_default(c, alloc):
    return [m["memory_id"] for m in alloc._allocate_slots(c, N)]


def rule_cap_off(c, alloc):
    return [m["memory_id"] for m in allocator(same_day_cap=0)._allocate_slots(c, N)]


def rule_no_kw(c, alloc):
    return rule_default([m for m in c if m["_debug_source"] != "keyword"], alloc)


def rule_kw_shift(delta):
    """keyword rows pay a penalty: score + delta (lower is better), then as shipped."""
    def f(c, alloc):
        cc = [dict(m, score=m["score"] + delta) if m["_debug_source"] == "keyword" else m for m in c]
        cc.sort(key=lambda m: m["score"])
        return rule_default(cc, alloc)
    return f


def rule_vector_first(k_vec):
    """Type-blind rule from the review: the vector lane's top-k_vec rows cannot be displaced by
    keyword rows; keyword rows may only fill slots after them (then everything as shipped)."""
    def f(c, alloc):
        vec = [m for m in c if m["_debug_source"] != "keyword"]
        kw = [m for m in c if m["_debug_source"] == "keyword"]
        head = sorted(vec, key=lambda m: m["score"])[:k_vec]
        head_ids = {m["memory_id"] for m in head}
        floor = max((m["score"] for m in head), default=-9)
        rest = [m for m in vec if m["memory_id"] not in head_ids] + \
               [dict(m, score=max(m["score"], floor + 1e-6)) for m in kw if m["memory_id"] not in head_ids]
        rest.sort(key=lambda m: m["score"])
        return rule_default(head + rest, alloc)
    return f


def rule_kw_fallback(c, alloc):
    """keyword rows only when the vector lane found nothing (empty vector pool)."""
    vec = [m for m in c if m["_debug_source"] != "keyword"]
    return rule_default(vec if vec else c, alloc)


RULES = {
    "replay_default": rule_default,
    "cap_off": rule_cap_off,
    "no_kw": rule_no_kw,
    "kw+0.10": rule_kw_shift(0.10),
    "kw+0.20": rule_kw_shift(0.20),
    "kw+0.30": rule_kw_shift(0.30),
    "vec_first5": rule_vector_first(5),
    "vec_first10": rule_vector_first(10),
    "vec_first20": rule_vector_first(20),
    "kw_fallback": rule_kw_fallback,
}


def main(files):
    rows = []
    for f in files:
        for g in glob.glob(f):
            rows += [json.loads(l) for l in open(g, encoding="utf-8")]
    rows = {r["question_id"]: r for r in rows}.values()
    rows = sorted(rows, key=lambda r: r["question_id"])
    alloc = allocator()
    types = sorted({r["question_type"] for r in rows})
    per = {name: defaultdict(int) for name in list(RULES) + ["vector_only", "anchor_default", "anchor_no_cap", "anchor_no_kw"]}
    n_type = defaultdict(int)
    fidelity = 0; unreachable = defaultdict(int); unreachable_vec_hit = defaultdict(int)
    flips = {name: [] for name in RULES}
    for r in rows:
        t = r["question_type"]; n_type[t] += 1
        ev = r["evidence"]
        c = sorted(to_cands(r["candidates"]), key=lambda m: m["score"])
        pool_sess = set(sessions([m["memory_id"] for m in c]))
        if not all(e in pool_sess for e in ev):
            unreachable[t] += 1
            if hit(r["lanes"]["vector_only"], ev):
                unreachable_vec_hit[t] += 1
        for lane in ("vector_only", "anchor_default", "anchor_no_cap", "anchor_no_kw"):
            per[lane][t] += hit(r["lanes"][lane], ev)
        base_hit = hit(r["lanes"]["anchor_default"], ev)
        for name, fn in RULES.items():
            s = sessions(fn(c, alloc))
            h = hit(s, ev)
            per[name][t] += h
            if name == "replay_default":
                fidelity += (s[:20] == r["lanes"]["anchor_default"][:20])
            elif h != base_hit:
                flips[name].append((r["question_id"], t, "+" if h else "-"))
    total = len(rows)
    print(f"questions {total}; replay_default reproduces anchor_default lane list: {fidelity}/{total}")
    print("unreachable (no evidence session in the candidate pool at all) by type: " +
          ", ".join(f"{t} {unreachable[t]}/{n_type[t]} (vector_only still hit {unreachable_vec_hit[t]})" for t in types))
    hdr = f"{'rule':16s}" + "".join(f" {t[:14]:>14s}" for t in types) + f" {'TOTAL':>6s}  {'vs bare':>8s}"
    print(hdr)
    bare = per["vector_only"]
    for name in ["vector_only", "anchor_default", "anchor_no_cap", "anchor_no_kw"] + list(RULES):
        d = per[name]; tot = sum(d.values())
        ok_all = all(d[t] >= bare[t] - (1 if n_type[t] <= SMALL else 0) for t in types) and tot >= sum(bare.values())
        print(f"{name:16s}" + "".join(f" {d[t]:>7d}/{n_type[t]:<6d}" for t in types) + f" {tot:>6d}  {'PASS' if ok_all else '':>8s}")
    print("\nflips vs anchor_default (+ gained, - lost), first 8 per rule:")
    for name, fl in flips.items():
        if fl:
            g = sum(1 for x in fl if x[2] == "+"); l = len(fl) - g
            print(f"  {name:14s} +{g} -{l}: " + ", ".join(f"{q}({t[:6]}{s})" for q, t, s in fl[:8]))


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v122_*.jsonl")])

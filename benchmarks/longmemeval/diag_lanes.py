#!/usr/bin/env python3
"""Diagnostic: on questions where the raw embedder finds all evidence in top-5 but
AnchorMemory.search() does not, re-run with one piece of the machinery removed at a time
to see which piece costs the recall. Read-only on the dataset; stores are temporary."""
import json, glob, os, sys, time, tempfile, shutil, contextlib, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from run_retrieval import _parse_date, _sessions_in_order, DATA_DEFAULT
from datetime import timedelta
import anchor_memory
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for f in glob.glob("results_*.jsonl") for l in open(f)]
miss = [r for r in rows if all(e in r["lanes"]["vector_only"][:5] for e in r["evidence"])
        and not all(e in r["lanes"]["anchor_default"][:5] for e in r["evidence"])]
print(f"[diag] {len(miss)} questions: vector_only hit all@5, anchor_default missed", flush=True)
data = {q["question_id"]: q for q in json.load(open(DATA_DEFAULT))}
emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
anchor_memory.SentenceTransformer = lambda name: emb
VARIANTS = ["default", "no_keyword", "no_slots", "no_boosts", "no_keyword+no_slots", "no_keyword+no_slots+no_boosts"]
hits = {v: 0 for v in VARIANTS}
out = open("diag_lanes.jsonl", "a")
for n, r in enumerate(miss, 1):
    q = data[r["question_id"]]
    d = tempfile.mkdtemp(prefix="lmed_", dir="tmp")
    try:
        mem = anchor_memory.AnchorMemory(d)
        _l = lambda v: eval(v) if isinstance(v, str) else v
        sids = _l(q["haystack_session_ids"]); dates = _l(q["haystack_dates"])
        with contextlib.redirect_stdout(io.StringIO()):
            for sid, ds, sess in zip(sids, dates, q["haystack_sessions"]):
                base = _parse_date(ds)
                for i, t in enumerate(sess):
                    tx = (t.get("content") or "").strip()
                    if tx: mem.store(f"{sid}|{i}", tx, tag=t.get("role", "user"), tier="long", entity=sid,
                                     timestamp=(base + timedelta(seconds=30 * i)).isoformat())
        res = {}
        for v in VARIANTS:
            mem.same_day_cap = 0 if "no_slots" in v else 2
            mem.recency_weight = 0.0 if "no_boosts" in v else 0.05
            kw = mem._keyword_fallback
            if "no_keyword" in v: mem._keyword_fallback = lambda *a, **k: []
            if "no_boosts" in v:   # emotion/citation boosts are constant on a fresh store; recency is the live one
                pass
            got = _sessions_in_order([x["memory_id"] for x in mem.search(q["question"], n_results=60, hebbian=False, no_cite=True)])
            mem._keyword_fallback = kw
            ok = all(e in got[:5] for e in r["evidence"]); hits[v] += ok; res[v] = ok
        out.write(json.dumps({"question_id": r["question_id"], "type": r["question_type"], **res}) + "\n"); out.flush()
        print(f"[diag] {n}/{len(miss)} " + " ".join(f"{v}={'✓' if res[v] else '✗'}" for v in VARIANTS), flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
print("\n[diag] all@5 recovered on the missed set:")
for v in VARIANTS: print(f"  {v:34s} {hits[v]:3d}/{len(miss)}")

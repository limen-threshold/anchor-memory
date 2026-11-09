#!/usr/bin/env python3
"""LongMemEval retrieval benchmark for Anchor — zero-model, retrieval only.

What it measures: for each of the 500 LongMemEval-S questions, build a fresh
Anchor store from that question's haystack (every turn of every session stored
as one memory, verbatim, with the session's date), ask Anchor the question, and
check whether the evidence session(s) are among the top-k retrieved sessions.
No LLM answers the question here, so this is the *retrieval* half of the
benchmark — the half that is Anchor's job. (The QA half needs a chat model and a
judge model; see --help for how to export retrieved context for that.)

Lanes (all run on the same store, read-only: hebbian=False, no_cite=True):
  anchor_default   AnchorMemory.search() as shipped (boosts + slot allocation)
  anchor_plain     same, with same_day_cap=0 and recency_weight=0
  vector_only      raw ChromaDB cosine top-N (what the embedder alone gives)
  transcript_kw    anchor_transcript.search() — keyword+date lane, 3 longest
                   content words of the question, all must match (zero-model floor)

Metrics (session level): R@k any = at least one evidence session in top-k
sessions; R@k all = every evidence session in top-k. k = 5 and 10.

Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
      (longmemeval_s_cleaned.json, 277 MB; put it in ./data/)

Usage:
  python run_retrieval.py --limit 5                      # smoke
  python run_retrieval.py --shard 0/3 --out results_0.jsonl   # one of three shards
  python run_retrieval.py --summarize results_*.jsonl    # table from finished rows
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

DATA_DEFAULT = os.path.join(HERE, "data", "longmemeval_s_cleaned.json")
KS = (5, 10)
LANES = ("anchor_default", "anchor_plain", "vector_only", "transcript_kw")

_STOP = set("""a an the i me my we our you your he she it they them his her their this that these those what which who whom
when where why how is are was were be been being do does did have has had will would shall should can could may might must
of in on at to for from by with about as into like through after before between during without within along across behind
beyond under over again further then once here there all any both each few more most other some such no nor not only own same
so than too very s t just don should now and or but if because while until although though tell told say said ask asked give
gave get got know knew think thought want wanted need needed make made take took come came go went see saw look looked use used
find found also back still even much many well way thing things something anything nothing one two three first last next
recommend recommended mention mentioned earlier previously conversation conversations remember""".split())


def _parse_date(s: str) -> datetime:
    # "2023/05/20 (Sat) 02:21"
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", s or "")
    if not m:
        return datetime(2023, 1, 1)
    y, mo, d, h, mi = map(int, m.groups())
    return datetime(y, mo, d, h, mi)


def _content_terms(q: str, n: int = 3) -> list:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+|\d+", q)
    words = [w for w in words if w.lower() not in _STOP and len(w) >= 3]
    words = sorted(dict.fromkeys(words), key=lambda w: -len(w))
    return words[:n]


def _sessions_in_order(ids: list) -> list:
    out = []
    for mid in ids:
        sid = mid.rsplit("|", 1)[0]
        if sid not in out:
            out.append(sid)
    return out


def build_and_query(q: dict, embedder, tmp_root: str, n_turns: int = 60) -> dict:
    """Store one question's haystack into a fresh Anchor, run all lanes, return ranked sessions."""
    import anchor_memory
    import anchor_transcript

    db_dir = tempfile.mkdtemp(prefix="lme_", dir=tmp_root)
    try:
        anchor_memory.SentenceTransformer = lambda name: embedder  # share one loaded model
        mem = anchor_memory.AnchorMemory(db_dir)
        sids = eval(q["haystack_session_ids"]) if isinstance(q["haystack_session_ids"], str) else q["haystack_session_ids"]
        dates = eval(q["haystack_dates"]) if isinstance(q["haystack_dates"], str) else q["haystack_dates"]
        n_stored = n_merged = 0
        t0 = time.time()
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):   # store() prints near-dup merges
            for sid, dstr, sess in zip(sids, dates, q["haystack_sessions"]):
                base = _parse_date(dstr)
                for i, turn in enumerate(sess):
                    text = (turn.get("content") or "").strip()
                    if not text:
                        continue
                    ts = (base + timedelta(seconds=30 * i)).isoformat()
                    mid = f"{sid}|{i}"
                    got = mem.store(mid, text, tag=turn.get("role", "user"), tier="long",
                                    entity=sid, timestamp=ts)
                    n_stored += 1
                    if got != mid:
                        n_merged += 1
                    anchor_transcript.append(db_dir, turn.get("role", "user"), text, ts=ts, conv=sid)
        t_store = time.time() - t0
        question = q["question"]
        lanes = {}
        t0 = time.time()
        # Lane 1: as shipped
        res = mem.search(question, n_results=n_turns, hebbian=False, no_cite=True)
        lanes["anchor_default"] = _sessions_in_order([r["memory_id"] for r in res])
        # Lane 2: plain top-N (no same-day cap, no recency boost)
        mem.same_day_cap = 0
        mem.recency_weight = 0.0
        res = mem.search(question, n_results=n_turns, hebbian=False, no_cite=True)
        lanes["anchor_plain"] = _sessions_in_order([r["memory_id"] for r in res])
        # Lane 3: raw vector
        emb = embedder.encode(question).tolist()
        raw = mem._collection.query(query_embeddings=[emb], n_results=min(n_turns, mem._collection.count()),
                                    include=["metadatas"])
        lanes["vector_only"] = _sessions_in_order([m["memory_id"] for m in raw["metadatas"][0]])
        # Lane 4: transcript keyword lane
        terms = _content_terms(question)
        hits = anchor_transcript.search(db_dir, " ".join(terms), limit=n_turns)["hits"] if terms else []
        lanes["transcript_kw"] = _sessions_in_order([h.get("conv", "?") + "|0" for h in hits])
        t_query = time.time() - t0
        return {"question_id": q["question_id"], "question_type": q["question_type"], "question": question,
                "evidence": eval(q["answer_session_ids"]) if isinstance(q["answer_session_ids"], str) else q["answer_session_ids"],
                "n_sessions": len(sids), "n_turns": n_stored, "n_near_dup_merged": n_merged,
                "t_store_s": round(t_store, 1), "t_query_s": round(t_query, 2),
                "kw_terms": terms, "lanes": {k: v[:20] for k, v in lanes.items()}}
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


def summarize(rows: list) -> str:
    out = []
    types = sorted({r["question_type"] for r in rows})
    out.append(f"questions: {len(rows)}   turns stored: {sum(r['n_turns'] for r in rows):,}   "
               f"near-dup merged: {sum(r['n_near_dup_merged'] for r in rows):,}   "
               f"store time: {sum(r['t_store_s'] for r in rows)/3600:.2f} h   "
               f"median query time: {sorted(r['t_query_s'] for r in rows)[len(rows)//2]:.2f} s")
    hdr = f"{'lane':16s}" + "".join(f" {'R@%d any'%k:>8s} {'R@%d all'%k:>8s}" for k in KS)
    out.append("\n== all question types\n" + hdr)
    for lane in LANES:
        line = f"{lane:16s}"
        for k in KS:
            any_ = sum(any(e in r["lanes"][lane][:k] for e in r["evidence"]) for r in rows) / len(rows)
            all_ = sum(all(e in r["lanes"][lane][:k] for e in r["evidence"]) for r in rows) / len(rows)
            line += f" {100*any_:8.1f} {100*all_:8.1f}"
        out.append(line)
    for t in types:
        sub = [r for r in rows if r["question_type"] == t]
        out.append(f"\n== {t} (n={len(sub)})\n" + hdr)
        for lane in LANES:
            line = f"{lane:16s}"
            for k in KS:
                any_ = sum(any(e in r["lanes"][lane][:k] for e in r["evidence"]) for r in sub) / len(sub)
                all_ = sum(all(e in r["lanes"][lane][:k] for e in r["evidence"]) for r in sub) / len(sub)
                line += f" {100*any_:8.1f} {100*all_:8.1f}"
            out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/n — process questions with index %% n == i")
    ap.add_argument("--tmp", default=os.path.join(HERE, "tmp"))
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2", help="Anchor's default embedder")
    ap.add_argument("--summarize", nargs="*", help="result files to summarize (no run)")
    args = ap.parse_args()

    if args.summarize is not None:
        files = args.summarize or [args.out]
        rows = []
        for pat in files:
            for f in glob.glob(pat):
                rows += [json.loads(l) for l in open(f) if l.strip()]
        print(summarize(rows))
        return

    data = json.load(open(args.data))
    idx = list(range(len(data)))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        idx = [j for j in idx if j % n == i]
    if args.limit:
        idx = idx[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["question_id"] for l in open(args.out) if l.strip()}
    os.makedirs(args.tmp, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.model)
    print(f"[lme] {len(idx)} questions, {len(done)} already done, embedder on {embedder.device}", flush=True)
    with open(args.out, "a") as fo:
        for n, j in enumerate(idx, 1):
            q = data[j]
            if q["question_id"] in done:
                continue
            t0 = time.time()
            row = build_and_query(q, embedder, args.tmp)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            ev = row["evidence"]
            hit = {lane: ("✓" if any(e in row["lanes"][lane][:5] for e in ev) else "✗") for lane in LANES}
            print(f"[lme] {n}/{len(idx)} {q['question_id']} {q['question_type']:26s} turns={row['n_turns']:4d} "
                  f"{time.time()-t0:5.1f}s  R@5: " + " ".join(f"{k}={v}" for k, v in hit.items()), flush=True)


if __name__ == "__main__":
    main()

"""Paired comparison of bare embedders on the same questions.

    python embedder_compare.py results_emb_<name>.jsonl [...]   # each file = one embedder's --bare-only run

Baseline is the MiniLM `vector_only` lane from runs_v122 (the shipped embedder), restricted to the
question ids present in the candidate file, so every comparison is paired question for question.
Prints per-class answered counts, total, and the number of questions each side wins.
"""
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5


def hit(sess, ev, k=K):
    return all(e in sess[:k] for e in ev)


def load(pattern):
    rows = {}
    for g in glob.glob(pattern):
        for l in open(g, encoding="utf-8"):
            r = json.loads(l); rows[r["question_id"]] = r
    return rows


def main(files):
    base = load(os.path.join(HERE, "runs_v122", "results_v122_*.jsonl"))
    for f in files:
        cand = load(f)
        ids = sorted(set(cand) & set(base))
        types = sorted({base[i]["question_type"] for i in ids})
        n = defaultdict(int); b = defaultdict(int); c = defaultdict(int); wins = losses = 0
        for i in ids:
            t = base[i]["question_type"]; ev = base[i]["evidence"]; n[t] += 1
            hb = hit(base[i]["lanes"]["vector_only"], ev); hc = hit(cand[i]["lanes"]["vector_only"], ev)
            b[t] += hb; c[t] += hc; wins += (hc and not hb); losses += (hb and not hc)
        name = os.path.basename(f).replace("results_emb_", "").replace(".jsonl", "")
        print(f"\n== {name} vs MiniLM (shipped), paired on {len(ids)} questions; R@{K} all-evidence")
        print(f"{'class':28s} {'n':>4s} {'MiniLM':>7s} {name[:10]:>10s}")
        for t in types:
            print(f"{t:28s} {n[t]:>4d} {b[t]:>7d} {c[t]:>10d}")
        print(f"{'TOTAL':28s} {len(ids):>4d} {sum(b.values()):>7d} {sum(c.values()):>10d}   wins +{wins} / losses -{losses}")
        st = [cand[i]["t_store_s"] for i in ids]; sb = [base[i]["t_store_s"] for i in ids]
        print(f"store time per question: MiniLM {sum(sb)/len(sb):.1f}s  {name[:10]} {sum(st)/len(st):.1f}s (both include a shared GPU)")


if __name__ == "__main__":
    main(sys.argv[1:])

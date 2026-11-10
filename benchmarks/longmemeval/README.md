# LongMemEval retrieval benchmark (Anchor)

First time Anchor has been measured against a public long-term-memory benchmark.
This directory runs the **retrieval half** of [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(500 questions, ~50 sessions / ~500 turns of haystack each) through Anchor as shipped,
with no LLM anywhere in the loop.

## What is measured

For each question a fresh Anchor store is built from that question's haystack: every
turn of every session is stored verbatim as one memory (`tag` = role, `entity` =
session id, `timestamp` = the session's date). The question is then asked through
four lanes, all read-only (`hebbian=False`, `no_cite=True`):

| lane | what it is |
|---|---|
| `anchor_default` | `AnchorMemory.search()` as shipped: vector + keyword lanes, citation/emotion/recency boosts, slot allocation (same-day cap, old-memory floor) |
| `anchor_plain` | same call with `same_day_cap = 0` and `recency_weight = 0` (plain top-N) |
| `vector_only` | raw ChromaDB cosine top-N on Anchor's default embedder (`paraphrase-multilingual-MiniLM-L12-v2`) |
| `transcript_kw` | `anchor_transcript.search()` with the 3 longest content words of the question, all required (the zero-model keyword floor from v1.20) |

Metric is **session-level recall**: R@k *any* = at least one evidence session is among
the top-k retrieved sessions; R@k *all* = every evidence session is (matters for the
`multi-session` and `temporal-reasoning` types, which have several). Turns are mapped
to sessions in rank order and de-duplicated, so "top-5 sessions" may draw on more than 5 turns.

The QA half (answering from the retrieved context, graded by a judge model) is not run
here; it needs a chat model and a judge model and is not Anchor's job. The rows in
`results_*.jsonl` keep the top-20 retrieved sessions per lane, so a QA pass can be
layered on later without re-running retrieval.

## Run

```bash
# data: longmemeval_s_cleaned.json (277 MB) into ./data/
python run_retrieval.py --limit 5                               # smoke
python run_retrieval.py --shard 0/3 --out results_0.jsonl &      # three shards ≈ 50 min on an M3 Ultra
python run_retrieval.py --shard 1/3 --out results_1.jsonl &
python run_retrieval.py --shard 2/3 --out results_2.jsonl &
python run_retrieval.py --summarize 'results_*.jsonl'
```

Runs are resumable (finished question ids are skipped). Each question's store lives in
`./tmp/` for ~15 s and is deleted afterwards.

## Results — 2026-11-09, Anchor v1.20, LongMemEval-S cleaned (500 questions)

246,738 turns stored (3,661 folded by the near-duplicate gate), ~2.1 h of store time
across three shards, median query 0.09 s. Numbers are session-level recall in %.

| lane | R@5 any | R@5 all | R@10 any | R@10 all |
|---|---|---|---|---|
| `anchor_default` | 94.2 | 70.2 | 96.8 | 75.8 |
| `anchor_plain` | 93.2 | 79.2 | 94.0 | 83.0 |
| `vector_only` | 97.0 | **86.6** | 99.0 | **94.8** |
| `transcript_kw` | 30.4 | 20.8 | 30.4 | 21.0 |

By question type, R@5 *all*:

| type (n) | default | plain | vector_only | transcript_kw |
|---|---|---|---|---|
| single-session-user (70) | 94.3 | 87.1 | 98.6 | 52.9 |
| single-session-assistant (56) | 100.0 | 100.0 | 100.0 | 30.4 |
| single-session-preference (30) | 93.3 | 90.0 | 90.0 | 3.3 |
| knowledge-update (78) | 92.3 | 94.9 | 97.4 | 33.3 |
| multi-session (133) | 58.6 | 69.9 | 80.5 | 7.5 |
| temporal-reasoning (133) | 38.3 | 63.9 | 73.7 | 9.8 |

**Reading it.** The embedder Anchor ships with finds every evidence session in the top
5 for 86.6 % of questions. Anchor's own search layer, sitting on top of that embedder,
hands back 70.2 %: the machinery (keyword lane, boosts, slot allocation) *loses* about
16 points on this benchmark, and 35 points on temporal-reasoning questions. Single-
session questions are fine in every lane; the damage is where several evidence sessions
must all surface. The v1.20 transcript keyword lane, as expected for a zero-model
all-terms-must-match search over English questions, is a floor (≈30 % any), not a
retrieval path. `diag_lanes.py` ablates the pieces one at a time on the questions the
embedder got right and Anchor got wrong; see "Findings" below for what it showed.

Not comparable one-to-one with numbers published by other systems (e.g. MemPalace's
"96.6 % R@5"): granularity (turn vs session), top-k definition and the "any/all"
convention differ between write-ups, and this run does no reranking and no QA.

## Results — 2026-11-09, Anchor v1.21 (same protocol, after the two fixes below)

| lane | R@5 any | R@5 all | R@10 any | R@10 all |
|---|---|---|---|---|
| `anchor_default` | 95.8 | **80.6** (was 70.2) | 97.8 | **91.4** (was 75.8) |
| `anchor_plain` | 95.8 | 80.6 | 97.8 | 91.4 |
| `vector_only` | 97.0 | 86.6 | 99.0 | 94.8 |
| `transcript_kw` | 30.4 | 20.8 | 30.4 | 21.0 |

By type, R@5 all, default lane: multi-session 58.6 → 66.9, temporal-reasoning 38.3 → 69.2, knowledge-update 92.3 → 94.9, single-session-user 94.3 → 95.7, single-session-preference 93.3 → 83.3 (n=30; the old keyword flood happened to help these — open item). Default and plain are now identical: with entities set the cap never bites. The remaining gap to the embedder is keyword rows with two matched tokens outranking weaker vector hits. Full tables: `summary_v121_20261109.txt`; v1.20 tables: `runs_v120/summary_20261109.txt`.

## Findings (2026-11-09, first run)

**Ablation** (`diag_lanes.py`): on the 84 questions where the raw embedder had every
evidence session in its top-5 and `anchor_default` did not, re-run with pieces removed:

| variant | recovered (all evidence in top-5) |
|---|---|
| default | 0 / 84 |
| no keyword lane | 10 / 84 |
| no slot allocation (`same_day_cap = 0`) | 55 / 84 |
| no recency boost | 0 / 84 |
| no keyword lane + no slot allocation | **84 / 84** |
| + no boosts | 84 / 84 |

So the whole gap between Anchor and its embedder on this benchmark is two pieces, and
the boosts are innocent.

- **Slot allocation's same-day cap assumes "same day ≈ same event".** It was built for
  a single person's memory store, where five memories written the same afternoon are
  usually about one thing and would otherwise crowd out the second topic. LongMemEval
  haystacks have ~5 unrelated sessions per calendar day, and multi-session / temporal
  questions often need several of them; a cap of 2 turns per day cannot return them
  all, and the slot goes to a less relevant turn from another day. The old-memory floor
  is moot here (everything is "older than 7 days" relative to now). A cap keyed on
  (day, conversation/entity) instead of day alone would keep the house behaviour and
  stop this: unrelated sessions on one day should not share a quota.
- **Keyword lane floods English queries.** `_keyword_fallback` tokenises the question,
  matches any token with `LIKE`, takes the first `n` rows in insertion order and gives
  them a fixed score of 0.6. For an English question most tokens are function words,
  so the lane returns the first `n` turns of the haystack. Vector hits only outrank them
  when cosine distance minus boosts is below 0.6. In `anchor_default` the same-day cap
  limits keyword hits to 2 per day so the vector lane still gets slots; in `anchor_plain`
  (cap off) the top-5 were literally sessions 1–5 of the haystack on the smoke question.
  This is a defect worth fixing (stopword filter, or score keyword hits by how many
  query tokens they match instead of a constant).

## Runs

- `runs_v120/` — v1.20 summary (2026-11-09 morning); per-question files are kept locally only.
- `runs_v121/` — v1.21 summary (2026-11-09 noon); same.
- v1.22 changed nothing this benchmark can see: every haystack turn carries a 2023 timestamp, so the old-memory floor never fires here and short-tier deletion is not part of the protocol. A 40-question sanity run matched v1.21 question for question.
- `runs_v122/` — v1.22 run with two single-switch lanes (`anchor_no_cap`, `anchor_no_kw`) and per-question candidate dumps; `fusion_eval.py` replays slot/fusion rules on those dumps (reproduces the shipped lane 500/500). Findings: cap off = no change; keyword lane off = bare embedder exactly (433); keyword base 0.95 (shipped in v1.23) = 436, ≥ bare in every class. Table in `docs/release-notes/v1.23.0.md`.

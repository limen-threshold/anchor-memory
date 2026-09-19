"""v1.18 smoke test — `python tests/smoke_v118.py`. Throwaway db in the system temp dir; no network
(the LLM is a stub), needs the embedding model cached.

What v1.18 promises: consolidation never destroys first-hand wording, and nothing leaves the
store without a verbatim copy in deleted_memories.jsonl."""
import os, sys, json, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta

from anchor_memory import AnchorMemory
import dream_extras

root = tempfile.mkdtemp(prefix="anchor_v118_")
mem = AnchorMemory(root)
mem._eager_link = False
fails = []


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)


class _Resp:
    def __init__(self, text): self.text = text


class StubLLM:
    provider, model = "stub", "stub"
    def __init__(self, fn): self.fn, self.calls = fn, 0
    def call(self, system="", user="", **kw):
        self.calls += 1
        return _Resp(self.fn(system, user))


def archive():
    p = mem.db.archive_path()
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


OLD = (datetime.utcnow() - timedelta(days=90)).isoformat()

# ── 1. store(timestamp=) carries an original time; an edit never moves it ──
mem.store("t1", "She adopted a grey cat named Miso in the spring.", tag="life", tier="long", timestamp=OLD)
check(mem.db.get("t1")["timestamp"] == OLD, "store(timestamp=) sets the row time")
meta = mem._collection.get(ids=["t1"], include=["metadatas"])["metadatas"][0]
check(meta["timestamp"] == OLD, "vector metadata time matches the row")
mem.store("t1", "She adopted a grey cat named Miso in the spring (shelter on 5th street).", tag="life", tier="long")
check(mem.db.get("t1")["timestamp"] == OLD, "re-storing the same id does not move it in time")
check(mem._collection.get(ids=["t1"], include=["metadatas"])["metadatas"][0]["timestamp"] == OLD,
      "…nor its vector metadata")

# ── 2. delete archives first ──
mem.store("d1", "The router password was changed after the outage in March.", tag="infra", tier="long")
check(mem.delete("d1") is True and mem.db.get("d1") is None, "delete removes the row")
check(any(a["row"]["memory_id"] == "d1" and "router password" in a["row"]["text"] for a in archive()),
      "deleted row is in deleted_memories.jsonl verbatim")
check(not mem._collection.get(ids=["d1"])["ids"], "vector removed too")

# ── 3. decay: both stores, archived, only short tier ──
mem.store("s_old", "Bought oat milk and rye bread on the way home.", tag="daily", tier="short", timestamp=OLD)
mem.store("s_new", "Watered the basil on the balcony this morning.", tag="daily", tier="short")
mem.store("l_old", "First trip to Lisbon: tram 28, custard tarts, rain.", tag="travel", tier="long", timestamp=OLD)
spy = StubLLM(lambda s, u: "[]")
stats = mem.dream_pass(auto_discover=False, llm=spy)
check(stats["decayed_memories"] == 1, f"one expired short-tier row decayed ({stats['decayed_memories']})")
check(mem.db.get("s_old") is None and mem.db.get("s_new") and mem.db.get("l_old"), "only the expired short row went")
check(not mem._collection.get(ids=["s_old"])["ids"], "decayed row's vector is gone (no ghost)")
check(any(a["row"]["memory_id"] == "s_old" and a["reason"].startswith("decay_short") for a in archive()),
      "decayed row was archived")
check(spy.calls == 0 and stats["split_memories"] == 0, "dream_pass does NOT split unless asked")

# ── 4. split is opt-in and non-lossy ──
bundle = ("2026-03-02: rebuilt the garden shed roof with cedar shingles. "
          "2026-03-02: also filed the tax extension and paid the estimate. "
          "2026-03-02: Mira called about the choir audition next Friday.")
mem.store("b1", bundle, tag="log", tier="core", emotion_score=0.8, source="journal",
          context="verbatim diary page 12", timestamp=OLD)
mem.store("anchor", "Cedar shingles are stored in the garage loft.", tag="infra", tier="long")
mem.db.connect("b1", "anchor", weight=2.0)
huge = "x " * 3000
mem.store("big", huge + "roof. taxes. choir.", tag="log", tier="long")

def splitter(system, user):
    out = []
    if "[b1]" in user:
        out.append({"id": "b1", "into": [
            {"text": "2026-03-02: rebuilt the garden shed roof with cedar shingles.", "tag": "house"},
            {"text": "2026-03-02: also filed the tax extension and paid the estimate.", "tag": "money"},
            {"text": "2026-03-02: Mira called about the choir audition next Friday.", "tag": "people"}]})
    if "[big]" in user:
        out.append({"id": "big", "into": [{"text": "roof"}, {"text": "taxes"}]})
    return json.dumps(out)

llm = StubLLM(splitter)
check(mem.split_bundled(llm=llm, dry_run=True) == 1 and mem.db.get("b1"), "dry_run reports, changes nothing")
n = mem.split_bundled(llm=llm)
check(n == 1 and mem.db.get("b1") is None, "bundle split, parent removed")
check(mem.db.get("big") is not None, "over-long memory is never split")
kids = [m for m in mem.db.list_all(limit=100) if m["memory_id"].startswith("split_")]
kids = [mem.db.get(k["memory_id"]) for k in kids]
check(len(kids) == 3, f"three pieces stored ({len(kids)})")
check(all(k["timestamp"] == OLD for k in kids), "pieces inherit the parent's timestamp (not today)")
check(all(k["tier"] == "core" for k in kids), "pieces inherit the parent's tier")
check(all(bundle in (k["context"] or "") and "verbatim diary page 12" in k["context"] for k in kids),
      "each piece carries the parent's full original text + context")
km = mem._collection.get(ids=[k["memory_id"] for k in kids], include=["metadatas"])["metadatas"]
check(all(m.get("source") == "journal" and m["timestamp"] == OLD for m in km), "pieces keep source and time in vector metadata")
check(any(a["row"]["memory_id"] == "b1" and a["row"]["text"] == bundle for a in archive()), "parent archived verbatim")
check(any(mem.db.get_edge_weight(k["memory_id"], "anchor") for k in kids), "parent's edge moved to a piece")

# ── 5. global dedup folds metadata, never rewrites ──
A = "Her sister Lena moved to Porto in 2019 and opened a bookshop near the river."
B = "Lena (her sister) opened a bookshop by the river after moving to Porto in 2019."
mem.near_dup_merge_sim = 0          # let both land so dedup has a pair to judge
mem.store("ka", A, tag="family", tier="core", context="said over dinner, her exact words", timestamp=OLD)
mem.store("kb", B, tag="family", tier="long")

def judge(system, user):
    return json.dumps([{"decision": "merge_into_a", "keep_id": "ka", "remove_id": "kb",
                        "merged_text": "Lena, the sister, runs a Porto bookshop.", "reason": "same fact"}])

res = dream_extras.run_global_dedup(mem, threshold=0.5, max_pairs=50, llm=StubLLM(judge),
                                    audit_dir=os.path.join(root, "audit"))
ka = mem.db.get("ka")
check(res["merged"] >= 1 and mem.db.get("kb") is None, "duplicate folded in")
check(ka["text"] == A, "survivor keeps its own wording (no LLM paraphrase)")
check(ka["tier"] == "core", f"survivor keeps its tier (was reset to 'short' before v1.18) — {ka['tier']}")
check(ka["context"] == "said over dinner, her exact words", "survivor keeps its context")
check(any(a["row"]["memory_id"] == "kb" and a["row"]["text"] == B for a in archive()), "folded duplicate archived verbatim")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)

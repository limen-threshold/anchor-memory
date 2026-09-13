"""v1.14 smoke test — `python tests/smoke_v114.py`; uses a throwaway db in the system temp dir, needs the embedding model cached."""
import os, sys, shutil, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta

from anchor_memory import AnchorMemory, parse_date_range

root = tempfile.mkdtemp(prefix="anchor_v114_")
mem = AnchorMemory(root)
fails = []

def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)

# ── 1. tokenizer: single CJK char dropped unless allow-listed ──
toks = mem.db._tokenize_query("你给了他草")
check("草" not in toks, f"single-char 草 dropped from keyword tokens ({toks})")
mem.db.keyword_single_char_allow = {"草"}
toks2 = mem.db._tokenize_query("你给了他草")
check("草" in toks2, f"allow-listed 草 kept ({toks2})")
mem.db.keyword_single_char_allow = frozenset()
check("记忆" in mem.db._tokenize_query("记忆质量问题"), "jieba segments 记忆质量问题")

# ── 2. near-dup gate ──
a = mem.store("m_a", "她五月十一号第一次去香港，在尖沙咀吃了云吞面", tag="travel", tier="long")
b = mem.store("m_b", "她五月十一日第一次到香港，在尖沙咀吃云吞面", tag="travel", tier="long")
check(b == "m_a", f"near-dup folded into survivor (returned {b})")
check(mem.count() == 1, f"count stays 1 after near-dup ({mem.count()})")
check((mem.db.get("m_a") or {}).get("usage_count") == 1, "survivor cited once")
c = mem.store("m_c", "更正：她第一次去香港是五月十二号，不是十一号", tag="travel", tier="long")
check(c == "m_c" and mem.count() == 2, "correction bypasses the gate and is stored")
d = mem.store("m_d", "Ren 昨天在阳台给薄荷浇水，发现长了新芽", tag="garden", tier="long")
check(d == "m_d" and mem.count() == 3, "unrelated text stored normally")
mem.store("m_a", "她五月十一号第一次去香港，在尖沙咀吃了云吞面（改：加了一杯丝袜奶茶）", tag="travel", tier="long")
check(mem.count() == 3, "re-storing the same id is an edit, not a duplicate")

# ── 3. same-day cap + floors ──
def set_ts(mid, ts):
    with mem.db._conn() as conn:
        conn.execute("UPDATE memories SET timestamp=? WHERE memory_id=?", (ts, mid))
        conn.commit()
    row = mem._collection.get(ids=[mid], include=["metadatas"])
    meta = dict(row["metadatas"][0]); meta["timestamp"] = ts
    mem._collection.update(ids=[mid], metadatas=[meta])

now = datetime.utcnow()
hk = [
    ("hk1", "1999 年在香港机场转机，等了六个小时", "2026-05-14T12:00:00"),
    ("hk2", "1999 年香港转机那次在机场买了第一台随身听", "2026-05-14T13:00:00"),
    ("hk3", "1999 年香港转机时看见跑道外面下暴雨", "2026-05-14T14:00:00"),
    ("hk4", "1999 年香港机场转机的登机口是 43 号", "2026-05-14T15:00:00"),
    ("hk5", "五月十一号真的去了香港，住在旺角", "2026-05-11T16:00:00"),
    ("hk6", "今天又聊起香港，她说想再去一次旺角", (now - timedelta(hours=2)).isoformat()),
]
for mid, text, ts in hk:
    mem.store(mid, text, tag="travel", tier="long")
    set_ts(mid, ts)
set_ts("m_a", "2026-05-11T15:00:00")
set_ts("m_c", "2026-05-12T15:00:00")

res = mem.search("香港 转机 旺角", n_results=4, hebbian=False, no_cite=True)
days = {}
for r in res:
    days[r["timestamp"][:10]] = days.get(r["timestamp"][:10], 0) + 1
print("   top-4:", [(r["memory_id"], r["timestamp"][:10]) for r in res])
check(max(days.values()) <= 2, f"same-day cap: no day exceeds 2 ({days})")
check(any(mem._age_days(r["timestamp"]) <= 3 for r in res), "recent floor: a ≤3-day memory is present")
check(sum(1 for r in res if mem._age_days(r["timestamp"]) > 7) >= 2, "old floor: ≥2 old memories present")

mem.same_day_cap = 0
res0 = mem.search("1999 年香港机场转机", n_results=4, hebbian=False, no_cite=True, debug=True)
days0 = {}
for r in res0:
    days0[r["timestamp"][:10]] = days0.get(r["timestamp"][:10], 0) + 1
print("   cap=0 top-4:", [(r["memory_id"], round(r["score"], 3)) for r in res0])
scores0 = [r["score"] for r in res0]
check(len(res0) == 4 and scores0 == sorted(scores0), "cap=0 → plain score order")
check(max(days0.values()) >= 3, f"cap=0 lets the same-day cluster fill slots ({days0})")
mem.same_day_cap = 2

# ── 4. exclude_tags ──
mem.store("tr1", "【转录】她：香港转机那次你还记得吗 / 他：记得，43 号登机口", tag="transcript", tier="short")
r_in = mem.search("香港 转机 登机口", n_results=5, hebbian=False, no_cite=True)
r_ex = mem.search("香港 转机 登机口", n_results=5, hebbian=False, no_cite=True, exclude_tags=("transcript",))
check(any(r["memory_id"] == "tr1" for r in r_in), "transcript surfaces without exclude_tags")
check(not any(r["memory_id"] == "tr1" for r in r_ex), "transcript dropped with exclude_tags")
check(len(r_ex) == 5, f"excluded tag did not consume a slot ({len(r_ex)} results)")

# ── 5. search_multi merged-stage cap ──
rm = mem.search_multi(["香港转机", "香港机场", "旺角"], n_results_per_query=4, n_total=8, hebbian=False, no_cite=True)
dm = {}
for r in rm:
    dm[r["timestamp"][:10]] = dm.get(r["timestamp"][:10], 0) + 1
check(max(dm.values()) <= 2, f"search_multi merged cap holds ({dm})")

# ── 6. date parsing + read_by_date ──
fixed_now = datetime(2026, 9, 13, 2, 0, 0)
def rng(q):
    r = parse_date_range(q, now=fixed_now)
    return None if r is None else (r[0][:10], r[1][:10], r[2])
off = fixed_now.astimezone().utcoffset()
print("   local→utc offset:", off)
check(parse_date_range("2026-05-14", now=fixed_now) is not None, "ISO date parses")
check(rng("3月6日") == rng("三月六日") == rng("March 6"), f"3月6日 / 三月六日 / March 6 agree ({rng('3月6日')})")
check(rng("三月十六日")[2] == 1 and "03-1" in rng("三月十六日")[0], f"三月十六日 → {rng('三月十六日')}")
check(rng("上周")[2] == 7, "上周 spans 7 days")
check(rng("最近怎么样了")[2] == 3, "最近 → three-day window")
check(rng("十二月三号")[0].startswith("2025-12"), f"December without year → last year ({rng('十二月三号')})")
check(parse_date_range("nothing here") is None, "no date → None")

s, e, rows = mem.read_by_date("2026-05-14")
ids = [r["memory_id"] for r in (rows or [])]
check(rows is not None and set(ids) >= {"hk1", "hk2", "hk3", "hk4"} and "hk5" not in ids,
      f"read_by_date 2026-05-14 → {ids}")
s, e, rows = mem.read_by_date("2026-05-11", days=3)
ids = [r["memory_id"] for r in (rows or [])]
check({"m_a", "hk5", "m_c"} <= set(ids) and "hk1" not in ids, f"read_by_date 05-11 +3 days → {ids}")
check(mem.read_by_date("no date")[2] is None, "unparseable date → rows None")

# ── 7. MCP handler wiring ──
import anchor_mcp
tools, handle, _m = anchor_mcp.create_server(root)
names = {t["name"] for t in tools}
check("read_memories_by_date" in names, "MCP tool read_memories_by_date registered")
out = handle("read_memories_by_date", {"date": "2026-05-14"})
check(out.get("total") == 4, f"MCP read_memories_by_date total=4 ({out.get('total')})")
out = handle("store_memory", {"text": "1999 年在香港机场转机，等了六个小时。", "tier": "long"})
check(out.get("status") == "merged_into_existing", f"MCP store near-dup receipt ({out})")
out = handle("search_memory", {"query": "香港 转机 登机口", "exclude_tags": ["transcript"], "hebbian": False})
check(not any(r["memory_id"] == "tr1" for r in out["memories"]), "MCP search exclude_tags forwarded")

# ── 8. proxy imports (finally-seal edit compiles + app builds without upstream) ──
import anchor_proxy
check(hasattr(anchor_proxy, "create_app"), "anchor_proxy imports")

print()
print("FAILURES:", fails if fails else "none")
shutil.rmtree(root, ignore_errors=True)
sys.exit(1 if fails else 0)

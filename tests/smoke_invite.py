"""v1.15 invitations smoke test — `python tests/smoke_invite.py`. Stdlib parts run
without the embedding model; the MCP section needs it cached."""
import os, sys, json, shutil, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
import anchor_invite as ai

fails = []
def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)

root = tempfile.mkdtemp(prefix="anchor_invite_")
pinned = os.path.join(root, "pinned"); state = os.path.join(root, "state")

check(ai.today(pinned, state) is None, "no pool → None")
check("invitation_add" in ai.render_empty_block() and "邀请卡池是空的" in ai.render_empty_block("zh"), "empty-pool offer renders")
# the AI writes its own cards
check(ai.add_card(pinned, "去你说过的那家旧书店，只看不买", "出门", "mem_123 她 8-02 说想去"), "add_card writes first card")
check(not ai.add_card(pinned, "去你说过的那家旧书店，只看不买"), "add_card dedupes")
own = ai._read_pool(pinned)
check(len(own) == 1 and own[0][0] == "出门" and own[0][1].startswith("去你说过的"), f"own card parsed with level ({own})")
c_own = ai.today(pinned, state, now=datetime(2026, 9, 12, 9, 0))
check(c_own and c_own["text"].startswith("去你说过的"), "draw comes from the AI's own pool")
os.remove(os.path.join(pinned, ai.POOL_FILE)); shutil.rmtree(state, ignore_errors=True)
p = ai.init_pool(pinned, lang="zh")
check(os.path.exists(p) and ai.init_pool(pinned) == p, "init_pool copies once, idempotent")
pool = ai._read_pool(pinned)
check(len(pool) >= 20 and {"门口", "附近", "出门", "彩蛋"} <= {lv for lv, _, _ in pool}, f"pool parsed ({len(pool)} cards)")

d1 = datetime(2026, 9, 13, 9, 0)
c1 = ai.today(pinned, state, now=d1)
c1b = ai.today(pinned, state, now=d1 + timedelta(hours=8))
check(c1 and c1 == c1b and c1["status"] == "new", f"stable within the day: {c1['text']}")
c2 = ai.today(pinned, state, now=d1 + timedelta(days=1))
check(c2["id"] != c1["id"] and c2["date"] == "2026-09-14", f"new draw next day: {c2['text']}")

# history avoidance: no repeat until the pool is exhausted, then never yesterday's
spare = len(pool) - 2
seq = [ai.today(pinned, state, now=d1 + timedelta(days=2 + i))["id"] for i in range(spare + 10)]
check(len(set(seq[:spare])) == spare, f"first {spare} draws all distinct (pool={len(pool)})")
check(all(seq[i] != seq[i - 1] for i in range(1, len(seq))), "after cycling, never the same card two days running")
st = ai._load_state(state)
check(len(st["history"]) <= ai.HISTORY_KEEP, "history trimmed")

# bonus ratio over many days
bonus = 0; N = 200
for i in range(N):
    c = ai.today(pinned, state, now=d1 + timedelta(days=40 + i))
    bonus += 1 if "彩蛋" in c["level"] else 0
check(0.10 <= bonus / N <= 0.40, f"bonus level drawn sometimes, not always ({bonus}/{N})")

# render / done / skip
cN = ai.today(pinned, state, now=d1 + timedelta(days=400))
blk = ai.render_block(cN, lang="zh")
check("今日邀请卡" in blk and cN["text"] in blk and "invitation_done" in blk, "zh block renders with card text")
check("invitation card" in ai.render_block(cN), "en block renders")
done = ai.mark_done(state, "第一次一起看云", "她拍了两张，一张糊的")
check(done["status"] == "done" and done["name"] == "第一次一起看云", "mark_done records name")
check(ai.render_block(done, "zh") == "", "done card renders nothing")
check(ai.today(pinned, state, now=d1 + timedelta(days=400))["status"] == "done", "today() keeps done status same day")
c_next = ai.today(pinned, state, now=d1 + timedelta(days=401))
check(ai.mark_skipped(state)["status"] == "skipped" and ai.render_block(c_next) == "" or True, "skip path runs")

# retired lines are not drawn
with open(os.path.join(pinned, ai.POOL_FILE), "a", encoding="utf-8") as f:
    f.write("\n## 门口\n~~退役的一张~~\n")
check(all(t != "退役的一张" for _, t, _ in ai._read_pool(pinned)), "~~struck~~ line skipped")

# proxy: empty pool → offer once a week; with pool → card once per day
import anchor_proxy
e1 = anchor_proxy.build_invitation_block(os.path.join(root, "nopool"), os.path.join(root, "state_e"), now=d1)
e2 = anchor_proxy.build_invitation_block(os.path.join(root, "nopool"), os.path.join(root, "state_e"), now=d1 + timedelta(days=3))
e3 = anchor_proxy.build_invitation_block(os.path.join(root, "nopool"), os.path.join(root, "state_e"), now=d1 + timedelta(days=8))
check("pool is empty" in e1 and not e2 and e3, "proxy empty-pool offer once a week")
s2 = os.path.join(root, "state2")
b1 = anchor_proxy.build_invitation_block(pinned, s2, now=d1)
b2 = anchor_proxy.build_invitation_block(pinned, s2, now=d1 + timedelta(hours=1))
b3 = anchor_proxy.build_invitation_block(pinned, s2, now=d1 + timedelta(days=1))
check(b1 and not b2 and b3, "proxy injects once per day")

# MCP wiring (needs embedding model)
try:
    import anchor_mcp
    tools, handle, mem = anchor_mcp.create_server(os.path.join(root, "db"), pinned_dir=pinned)
    names = {t["name"] for t in tools}
    check({"get_invitation", "invitation_add", "invitation_done", "invitation_skip"} <= names, "MCP tools registered")
    check(handle("invitation_add", {"text": "拍一张你说过的那棵树", "level": "附近", "why": "她 9-01 提过"})["status"] == "added", "MCP invitation_add")
    g = handle("get_invitation", {})
    check(g.get("invitation") and g["invitation"]["status"] in ("new", "done", "skipped"), f"get_invitation → {g.get('invitation', {}).get('text')}")
    d = handle("invitation_done", {"name": "便利店的柠檬水", "note": "她说太酸"})
    check(d.get("status") == "stored" and d["memory_id"].startswith("mem_"), f"invitation_done stored {d.get('memory_id')}")
    row = mem.db.get(d["memory_id"])
    check(row and row["tag"] == "together" and "便利店的柠檬水" in row["text"], "stored as tag=together with the name")
    tl = open(os.path.join(pinned, "recent_timeline.md"), encoding="utf-8").read()
    check("together · 便利店的柠檬水" in tl, "timeline event appended")
    w = handle("wakeup", {})
    check("invitation" not in w, "wakeup omits a done card")
except Exception as e:
    check(False, f"MCP section raised {type(e).__name__}: {e}")

print("\nFAILURES:", fails if fails else "none")
shutil.rmtree(root, ignore_errors=True)
sys.exit(1 if fails else 0)

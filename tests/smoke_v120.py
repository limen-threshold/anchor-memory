"""v1.20 smoke test — `python tests/smoke_v120.py`. Throwaway db in the system temp dir; no network
(the verifier LLM is a stub), needs the embedding model cached.

What v1.20 promises:
  1. Raw transcripts are a first-class, searchable layer: the proxy's after_response appends every
     exchange; search is keyword + date, spread across days; nothing needs a model.
  2. The always-injected layer has a size readout (wakeup / admin / proxy log) — a warning, never a cut.
  3. dream_pass(verify=True) checks recent memories against the transcripts around their timestamp,
     writes a [verify …] annotation + ledger line, and never rewrites, re-tiers or deletes anything."""
import os, sys, json, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta

import anchor_transcript
import anchor_pinned
import anchor_proxy
import dream_extras
from anchor_llm import CallableLLM
from anchor_memory import AnchorMemory

root = tempfile.mkdtemp(prefix="anchor_v120_")
pinned = os.path.join(root, "pinned"); os.makedirs(pinned)
mem = AnchorMemory(root)
mem._eager_link = False
fails = []


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)


# ── 1. transcripts: append / dedupe / search / round-robin ──
now = datetime.now()
old = (now - timedelta(days=3)).replace(hour=10, minute=0, second=0).isoformat(timespec="seconds")
check(anchor_transcript.append(root, "user", "我第一次去香港是十月八号，住在上环", ts=old) is not None, "append writes a line")
check(anchor_transcript.append(root, "user", "我第一次去香港是十月八号，住在上环", ts=old) is None, "exact repeat of a recent line is ignored")
check(anchor_transcript.append(root, "user", "", ts=old) is None, "empty text ignored")
anchor_transcript.append(root, "assistant", "记得，你说上环的粥很好喝", ts=old[:11] + "10:00:30")
for i in range(25):   # a talkative day today
    anchor_transcript.append(root, "user", f"今天聊香港的第 {i} 句", ts=now.replace(hour=12, minute=0, second=i).isoformat(timespec="seconds"))
res = anchor_transcript.search(root, "香港")
check(res["total"] == 26, f"keyword search counts every match ({res['total']})")
check(len(res["hits"]) == 20 and res["truncated"], "limit 20 + truncated flag")
days_in_hits = {h["ts"][:10] for h in res["hits"]}
check(len(days_in_hits) == 2, "round-robin: the old day's match survives a talkative today")
res2 = anchor_transcript.search(root, "香港 上环")
check(res2["total"] == 1 and "上环" in res2["hits"][0]["text"], "every term must appear (AND)")
res3 = anchor_transcript.search(root, "hong kong")
check(res3["total"] == 0, "no false match")
anchor_transcript.append(root, "user", "Hong Kong was my FIRST trip", ts=old[:11] + "11:00:00")
check(anchor_transcript.search(root, "hong kong first")["total"] == 1, "ASCII terms are case-insensitive")
dfrom = old[:10]; dto = old[:10]
check(anchor_transcript.search(root, "香港", date_from=dfrom, date_to=dto)["total"] == 1, "date range filters days")
check([d for d, _n in anchor_transcript.days(root)] == sorted({old[:10], now.strftime("%Y-%m-%d")}), "days() lists the day files")
txt = anchor_transcript.format_hits(res)
check("matches in 2 days" in txt and "Narrow the dates" in txt, "format_hits explains truncation per day")

# ── 1b. the proxy appends after each reply (and strips the injected block) ──
msgs = [{"role": "user", "content": "<context_now>\n[time] now\n</context_now>\n\n明天去看牙"}]
anchor_proxy.after_response(mem, None, pinned, msgs, "好，几点？")
got = anchor_transcript.search(root, "看牙")
check(got["total"] == 1 and got["hits"][0]["text"] == "明天去看牙", "proxy after_response logs the user line without <context_now>")
check(anchor_transcript.search(root, "几点")["total"] == 1, "…and the assistant line")
os.environ["ANCHOR_TRANSCRIPT"] = "0"
anchor_proxy.after_response(mem, None, pinned, [{"role": "user", "content": "这句不该落盘"}], "嗯")
check(anchor_transcript.search(root, "不该落盘")["total"] == 0, "ANCHOR_TRANSCRIPT=0 turns logging off")
del os.environ["ANCHOR_TRANSCRIPT"]

# ── 2. injected-layer size readout ──
sz = anchor_pinned.injection_size(pinned)
check(sz["lines"] == 0 and sz["over"] is False, "empty pinned: zero, not over")
with open(os.path.join(pinned, "identity.md"), "w") as f:
    f.write("# me\n" + "line\n" * 50)
with open(os.path.join(pinned, "last_session.md"), "w") as f:
    f.write("tail " * 5000)   # excluded from the injected layer, must not count
sz = anchor_pinned.injection_size(pinned)
check(sz["lines"] == 51 and sz["over"] is False and [f["name"] for f in sz["files"]] == ["identity.md"], "counts the files load_pinned joins; tail excluded")
with open(os.path.join(pinned, "big.md"), "w") as f:
    f.write("x\n" * 300)
sz = anchor_pinned.injection_size(pinned)
check(sz["over"] is True and sz["lines"] > sz["limit_lines"], "over the 200-line ceiling is flagged")
check(os.path.exists(os.path.join(pinned, "big.md")) and os.path.getsize(os.path.join(pinned, "big.md")) == 600, "readout never trims a file")

# ── 3. dream-time verification against transcripts ──
t_mem = (now - timedelta(days=1)).replace(hour=9, minute=0, second=0)
anchor_transcript.append(root, "user", "我妈昨天打电话说她膝盖又疼了", ts=t_mem.isoformat(timespec="seconds"))
anchor_transcript.append(root, "assistant", "要不要带她去看看？", ts=(t_mem + timedelta(seconds=20)).isoformat(timespec="seconds"))
mem.store("m_ok", "她母亲膝盖又疼了（昨天来电）", tag="relationship", tier="long", timestamp=(t_mem + timedelta(minutes=1)).isoformat(timespec="seconds"))
mem.store("m_bad", "她母亲做了膝盖手术", tag="relationship", tier="long", timestamp=(t_mem + timedelta(minutes=2)).isoformat(timespec="seconds"))
mem.store("m_none", "她小时候养过一只叫豆豆的狗", tag="history", tier="long", timestamp=(now - timedelta(days=5)).isoformat(timespec="seconds"))
seen = []


def stub(system, user, max_tokens, temperature):
    seen.append(user)
    if "手术" in user:
        return '{"verdict": "unsupported", "quote": "我妈昨天打电话说她膝盖又疼了", "note": "原话只说疼，没说手术"}'
    return '```json\n{"verdict": "supported", "quote": "我妈昨天打电话说她膝盖又疼了", "note": "原话如此"}\n```'


llm = CallableLLM(stub, name="stub-verifier")
before = {m: mem.db.get(m)["text"] for m in ("m_ok", "m_bad", "m_none")}
stats = dream_extras.verify_recent(mem, llm=llm, days=7)
check(stats["checked"] == 2 and stats["supported"] == 1 and stats["unsupported"] == 1 and stats["no_transcript"] == 1,
      f"verifier: 2 checked (1 supported, 1 unsupported), 1 without transcript ({stats})")
ann_bad = [a["text"] for a in mem.db.get_annotations("m_bad")]
check(any(a.startswith("[verify ") and "unsupported" in a for a in ann_bad), "unsupported verdict lands as an annotation on the memory")
check(all(mem.db.get(m)["text"] == before[m] for m in before), "no memory text was rewritten")
check(mem.db.get("m_bad") is not None and mem.db.get("m_bad")["tier"] == "long", "nothing deleted or re-tiered")
check(any("TRANSCRIPT" in u and "膝盖又疼" in u for u in seen), "the LLM saw the transcript lines around the memory")
check(os.path.exists(os.path.join(root, "verify_ledger.jsonl")) and len(open(os.path.join(root, "verify_ledger.jsonl")).read().strip().split("\n")) == 2, "ledger has one line per verdict")
stats2 = dream_extras.verify_recent(mem, llm=llm, days=7)
check(stats2["checked"] == 0 and stats2["skipped_done"] == 2, "already-verified memories are not re-checked")
res = mem.dream_pass(verify=True, llm=llm)
check("verify" in res and res["verify"]["skipped_done"] == 2, "dream_pass(verify=True) runs the verifier and reports it")
res_nollm = mem.dream_pass(verify=True, llm=None)
os.environ.pop("ANCHOR_LLM", None)
check("verify" in res_nollm, "dream_pass without an LLM reports verify as skipped instead of crashing")

print(f"\n{len(fails)} failures" if fails else "\nALL PASS")
sys.exit(1 if fails else 0)

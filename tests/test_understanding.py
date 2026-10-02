"""Offline synthetic contracts, real SQLite read-only lane, proxy/MCP integration.
No external models, real residents, or production stores.
"""
import copy,json,os,sqlite3,sys,tempfile,threading,time,types,unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import anchor_understanding as au
import anchor_proxy as proxy
from anchor_db import AnchorDB
from anchor_mcp import create_server,handle_message

NOW=datetime.fromisoformat('2026-03-01T12:00:00+01:00')
RAW='我们一周后就要出门了，你记得吗？'
PLAN='March trip to Kyoto confirmed: leave March 8, arrive March 9. Stay with family; later schedule TBD.'
OLD='Last November, we went to Paris for a week.'

def analysis(current=RAW,needs=None,unknown=None):
 return {'explicit':[current], 'resolved':[], 'needs':needs if needs is not None else [{'question':'既定行程日期、目的地、住宿和同行约定是什么？','query':'既定旅行行程安排','query_en':'planned travel trip itinerary'}], 'hypotheses':[], 'unknown':unknown or ['此刻动机没有证据']}

class LLM:
 provider='test';model='offline'
 def __init__(self,current=RAW,review=None,data=None,error=None):self.calls=[];self.current=current;self.review=review;self.data=data;self.error=error
 def call(self,system,user,**kw):
  self.calls.append((system,json.loads(user),kw))
  if self.error:raise self.error
  if system==au.PROMPT:d=self.data if self.data is not None else analysis(self.current)
  elif self.review is not None:d=self.review
  else:
   d=[]
   for m in json.loads(user)['candidates']:
    ok='March trip' in m['text']
    d.append({'id':m['id'],'relation':'current' if ok else 'unrelated','reason':'同一已定行程，出发和抵达分别记载' if ok else '旧旅行持续时间不是本轮出发时间','quote':m['text'] if ok else ''})
  return types.SimpleNamespace(text=json.dumps(d,ensure_ascii=False),input_tokens=100,output_tokens=80,cost_usd=0)

class Mem:
 def __init__(self,db_path=None,rows=None,**kw):
  self.db=AnchorDB(str(Path(db_path)/'memories.db')) if db_path else types.SimpleNamespace(db_path=None)
  self.rows=rows or [];self.calls=[]
 def search(self,q,**kw):self.calls.append((q,kw));return copy.deepcopy(self.rows)
 def search_multi(self,q,**kw):self.calls.append((q,kw));return copy.deepcopy(self.rows)
 def store(self,*a,**kw):raise AssertionError('understanding must never store')

def row(mid='old',text=OLD):return {'memory_id':mid,'text':text,'snippet':text,'timestamp':'2025-11-02','tag':'travel','score':.3}

class Tests(unittest.TestCase):
 def test_readonly_bilingual_trip_recall_finds_unmentioned_destination(self):
  with tempfile.TemporaryDirectory() as td:
   m=Mem(td,[row()]);m.db.insert('planned',PLAN,tag='travel',tier='long',timestamp='2026-02-01')
   before=Path(m.db.db_path).read_bytes();llm=LLM()
   r=au.recall(m,llm,RAW,now=NOW)
   self.assertEqual([x['memory_id'] for x in r['memories']],['planned'])
   self.assertEqual(Path(m.db.db_path).read_bytes(),before)
   self.assertEqual(len(llm.calls),2)
   self.assertEqual(r['original'],RAW)
   self.assertTrue(any('March' in q for q in r['queries']))
   self.assertNotIn('Kyoto',' '.join(r['queries']))
   self.assertIn(PLAN,r['block']);self.assertIn('保存时间',r['block'])
   self.assertEqual(m.calls[0][1]['no_cite'],True);self.assertEqual(m.calls[0][1]['hebbian'],False)
   self.assertEqual(next(d for d in r['decisions'] if d['id']=='old')['relation'],'unrelated')

 def test_departure_duration_and_calendar(self):
  for s,want in [(RAW,'2026-03-08'),('明天买苹果','2026-03-02'),('for a week',None),('我们出去一周',None),('leave in a week','2026-03-08')]:
   h=au.time_hints(s,NOW);self.assertEqual(h[0]['date'] if h else None,want)

 def test_sources_versions_remain_distinct(self):
  rows=[row('v1','March trip leaves March 8.'),row('v2','March trip leaves March 10, tentative.')]
  rev=[{'id':r['memory_id'],'relation':'conflict','reason':'不同版本，未裁定','quote':r['text']} for r in rows]
  r=au.recall(Mem(rows=rows),LLM(review=rev),RAW,now=NOW)
  self.assertEqual(len(r['memories']),2);self.assertIn('v1',r['block']);self.assertIn('v2',r['block'])

 def test_apple_meaning_unknown_and_hypotheses_labelled(self):
  current='你记得我明天买苹果吗？'
  for ctx,meaning in [('苹果要做派。','做派'),('苹果带给妈妈。','礼物'),('刚看了苹果手机。','可能指手机，待确认'),('未说用途。','用途未知')]:
   d=analysis(current,[{'question':'苹果指什么、用途和既定约定是什么？','query':'苹果 用途 购买 约定','query_en':'apple purchase purpose agreement'}])
   d['resolved']=[{'value':meaning,'source':'recent:0','quote':ctx}];d['hypotheses']=['可能在确认记忆，未证实']
   r=au.recall(Mem(),LLM(current,data=d),current,[{'role':'user','content':ctx}],NOW)
   self.assertIn(meaning,r['block']);self.assertIn('假设，未证实',r['block']);self.assertIn('未找到相关材料',r['block'])
   self.assertEqual(r['original'],current)

 def test_short_reference_and_simple_ack(self):
  d=analysis('去哪里？');d['resolved']=[{'value':'刚提到的出门','source':'recent:0','quote':'我们一周后出门'}]
  p=au.understand(LLM('去哪里？',data=d),'去哪里？',[{'role':'user','content':'我们一周后出门'}],NOW)
  self.assertEqual(p['time_hints'][0]['date'],'2026-03-08');self.assertIn('待核对',p['time_hints'][0]['basis'])
  r=au.recall(Mem(),LLM('嗯。',data=analysis('嗯。',[])), '嗯。',now=NOW)
  self.assertNotIn('待记忆确认',r['block'])

 def test_failures_preserve_original_recall_and_unknowns(self):
  for llm in [None,LLM(error=TimeoutError()),LLM(data={}),LLM(data=dict(analysis(),explicit=[]))]:
   r=au.recall(Mem(rows=[row()]),llm,RAW,now=NOW)
   self.assertEqual(r['original'],RAW);self.assertIn(OLD,r['block']);self.assertIn('unverified',r['block'])
   self.assertTrue(r['analysis']['status'].startswith('fallback'))

 def test_fabricated_quote_dropped_and_bad_review_preserved(self):
  for rev,kept in [([{'id':'old','relation':'current','reason':'似乎相关','quote':'不存在的原句'}],0),([{'id':'old','relation':'current','reason':'格式缺摘句','quote':None}],1),([{'id':'old','relation':'current','quote':OLD}],1),({},1),([],1)]:
   r=au.recall(Mem(rows=[row()]),LLM(review=rev),RAW,now=NOW)
   self.assertEqual(len(r['memories']),kept)

 def test_sparse_logic_checks_and_unknowns_survive_display_cap(self):
  for raw in ['我死了后会怎么样？','我不默认它们同目标。','这里的疼是结构里的标识。','这本书我已经听过了，书在这里。']:
   self.assertTrue(au.literal_notes(raw))
  plan=au.understand(LLM(),RAW,now=NOW)
  plan['explicit']=['原话'*1000]
  block=au.render(plan,[],{})
  self.assertIn('此刻动机没有证据',block)
  self.assertIn('待记忆确认：',block)
  self.assertIn('未找到相关材料',block)
  self.assertTrue(block.endswith('</input_understanding>'))

 def test_timeout_bounds_wait_and_inflight_work(self):
  release=threading.Event()
  class Slow:
   def call(self,**kw):release.wait(2);return types.SimpleNamespace(text='{}')
  before=time.monotonic()
  try:
   for _ in range(2):
    with self.assertRaises(TimeoutError):au._call(Slow(),'s',{},10,timeout=.01)
   never=LLM()
   with self.assertRaises(TimeoutError):au._call(never,'s',{},10,timeout=.01)
   self.assertFalse(never.calls);self.assertLess(time.monotonic()-before,.3)
  finally:release.set()
  time.sleep(.01)

 def test_trace_and_no_motive_evidence(self):
  with tempfile.TemporaryDirectory() as td:
   path=str(Path(td)/'trace.jsonl');r=au.recall(Mem(),LLM(),RAW,now=NOW,trace_path=path)
   entries=[json.loads(l) for l in Path(path).read_text().splitlines()]
   self.assertEqual(entries[-1]['block'],r['block']);self.assertFalse(r['analysis']['hypotheses'])
   self.assertIn('此刻动机没有证据',r['block']);self.assertIn('test/offline',r['block'])

 def test_proxy_actual_injection_and_no_analysis_in_curator_or_tail(self):
  messages=[{'role':'user','content':RAW}]
  with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'ANCHOR_UNDERSTANDING':'1'}):
   llm=LLM();system,out=proxy.build_turn(Mem(rows=[row('plan',PLAN)]),llm,td,td,messages)
   self.assertTrue(out[-1]['content'].endswith(RAW));self.assertEqual(messages[-1]['content'],RAW)
   self.assertIn('<input_understanding>',out[-1]['content'])
   self.assertNotIn('most likely already happened',out[-1]['content'])
   self.assertEqual(proxy.clean_history(out),messages)
   tail=proxy.collect_turns(out,'answer');self.assertNotIn('<input_understanding>',str(tail))
   class Curator:
    def __init__(self):self.inputs=[]
    def call(self,**kw):self.inputs.append(kw['user']);return types.SimpleNamespace(text='[]')
   curator=Curator()
   # No entries, but inspect what curator actually received.
   proxy.curate_turn(Mem(),curator,td,out[-1]['content'],'answer')
   self.assertTrue(curator.inputs)
   self.assertNotIn('<input_understanding>',curator.inputs[0])

 def test_mcp_only_dispatch_and_zero_helper(self):
  with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'anchor_memory':types.SimpleNamespace(AnchorMemory=Mem)}),patch.object(proxy,'get_llm_or_none',return_value=None):
   tools,dispatch,mem=create_server(td)
   self.assertIn('understand_and_recall',[t['name'] for t in tools])
   response=handle_message({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'understand_and_recall','arguments':{'current':RAW,'now':NOW.isoformat()}}},tools,dispatch)
   payload=json.loads(response['result']['content'][0]['text'])
   self.assertEqual(payload['original'],RAW);self.assertIn('尚不清楚',payload['block'])

 def test_event_lane_ignores_wrong_month_and_preserves_no_store(self):
  with tempfile.TemporaryDirectory() as td:
   m=Mem(td);m.db.insert('wrong','November trip plans.',tag='travel');m.db.insert('right',PLAN,tag='travel')
   plan=au.understand(LLM(),RAW,now=NOW)
   rows=au.event_candidates(plan,[],m.db.db_path)
   self.assertEqual([r['memory_id'] for r in rows],['right'])

if __name__=='__main__':unittest.main()

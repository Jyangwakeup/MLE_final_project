from pathlib import Path
import sys,time,json,math,hashlib,os
root=Path('/export/data/sfan/MLE_final_project_safety');sys.path.insert(0,str(root))
import numpy as np
from tests.test_vectorized_viability import records
from tests.test_certified_placement import failure_record
from tests.test_opponent_rearming import record,V6,V7
from agent_code.team_agent.safety import safety_decision
rs=records()+[{**failure_record(i),'corpus':'baseline_failure'} for i in range(20)]+[{**record(i),'corpus':'round2_failure'} for i in range(165,185)]
results=[]
for r in rs:
 d=r['safety'];kw=dict(allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer']})
 entry={'corpus':r['corpus'],'step':r['state']['step'],'samples':[]}
 for repeat in range(4):
  for label,spec in ((('v6',V6),('v7',V7)) if repeat%2==0 else (('v7',V7),('v6',V6))):
   start=time.perf_counter();answer=safety_decision(r['state'],np.asarray(d['physical_mask']),spec,**kw);elapsed=time.perf_counter()-start
   entry['samples'].append(dict(version=label,warmup=repeat==0,seconds=elapsed,mask=answer.mask.tolist(),search_timed_out=answer.robust_search_timed_out,guarantee_loss=answer.robust_guarantee_loss,scenarios=list(answer.opponent_scenario_counts)))
 results.append(entry)
times=[s['seconds'] for r in results for s in r['samples'] if s['version']=='v7' and not s['warmup']]
bad=[(r['corpus'],r['step']) for r in results if any(s['search_timed_out'] for s in r['samples'] if s['version']=='v7')]
out=dict(status='passed' if not bad and max(times)<=.48 else 'failed',cpu=list(os.sched_getaffinity(0)),search_budget_ms=400,states=100,repetitions=3,warmups=1,scope='safety_decision only; not complete act admission',p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],max_seconds=max(times),timeout_states=bad,results=results,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'agent_code/team_agent/safety.py',root/'agent_code/team_agent/controllable_survival.py')})
(root/'experiments/results/safety_rearming_v7_20260917/preflight/historical_timing.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='results'})

import json,pickle,time
from pathlib import Path
import numpy as np
from agent_code.team_agent.safety import safety_decision
from tests.test_proven_movement import V9,V8
oldroot=Path('/export/data/sfan/MLE_final_project_safety');out=[]
for seed in (24011,24017,24022,24025):
 rows=pickle.loads((oldroot/f'runs/safety_stress_task4_0dc8dba/safety_stress_task4_0dc8dba_s{seed}/task4_death_states_round0001.pkl').read_bytes())
 for row in rows:
  d=row['safety']
  for version,spec in (('v8',V8),('v9',V9)):
   start=time.perf_counter();r=safety_decision(row['state'],np.asarray(d['physical_mask']),spec,allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']});ms=(time.perf_counter()-start)*1000
   out.append({'seed':seed,'step':row['state']['step'],'version':version,'ms':ms,'timeout':r.robust_search_timed_out,'states':r.robust_states_evaluated,'mask':r.mask.tolist()})
Path('runs/v9_historical_movement_timing.json').write_text(json.dumps(out,indent=2)+'\n')
for v in ('v8','v9'):
 rs=[r for r in out if r['version']==v];print(v,'count',len(rs),'max_ms',max(r['ms'] for r in rs),'timeouts',sum(r['timeout'] for r in rs),flush=True)

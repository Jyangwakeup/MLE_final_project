import json,time,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
from agent_code.team_agent.safety import safety_decision
from agent_code.team_agent.controllable_survival import controllable_survival_actions as optimized
from tests.reference_named_reach_survival import controllable_survival_actions as reference
from tests.test_opponent_reach_grid import failure_rows
from tests.test_proven_movement import V9
results=[]
for row in failure_rows():
 d=row['safety'];kw=dict(allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']})
 for repeat in range(11):
  versions=(('old',reference),('grid',optimized)) if repeat%2==0 else (('grid',optimized),('old',reference))
  for name,fn in versions:
   with patch('agent_code.team_agent.controllable_survival.controllable_survival_actions',fn):
    start=time.perf_counter();r=safety_decision(row['state'],np.asarray(d['physical_mask']),V9,**kw);ms=1000*(time.perf_counter()-start)
   results.append({'step':row['state']['step'],'version':name,'repeat':repeat,'warmup':repeat==0,'ms':ms,'timeout':r.robust_search_timed_out,'mask':r.mask.tolist(),'states':r.robust_states_evaluated})
 summary={}
 for name in ('old','grid'):
  rs=[r for r in results if r['version']==name and not r['warmup']];times=sorted(r['ms'] for r in rs)
  summary[name]={'samples':len(rs),'p95_ms':times[math.ceil(.95*len(times))-1],'max_ms':max(times),'timeouts':sum(r['timeout'] for r in rs)}
 Path('runs/reach_grid_benchmark.json').write_text(json.dumps({'scope':'20 recorded pre-failure states, one warmup and ten alternating samples each; safety_decision only, runtime budget unchanged at 400ms; correctness separately compared with completed offline proofs.','cpu':3,'summary':summary,'results':results},indent=2)+'\n')
 print('step',row['state']['step'],'done',flush=True)
print(summary,flush=True)

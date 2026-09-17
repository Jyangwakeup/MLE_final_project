import pickle,time,json
from pathlib import Path
import numpy as np
from agent_code.team_agent.safety import safety_decision
from tests.test_proven_movement import V9
p=Path('experiments/results/safety_reach_grid_20260917/fresh_failure/task4_failure_states.pkl')
rows=pickle.loads(p.read_bytes());results=[]
for repeat in range(10):
 for row in rows:
  d=row['safety'];start=time.perf_counter();cpu=time.process_time()
  r=safety_decision(row['state'],np.asarray(d['physical_mask']),V9,allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']})
  results.append({'repeat':repeat,'step':row['state']['step'],'ms':1000*(time.perf_counter()-start),'cpu_ms':1000*(time.process_time()-cpu),'timeout':r.robust_search_timed_out,'states':r.robust_states_evaluated})
 print('repeat',repeat,'timeouts',sum(x['timeout'] for x in results),'max',max(x['ms'] for x in results),flush=True)
Path('runs/repro_24384_sequence.json').write_text(json.dumps(results,indent=2)+'\n')
assert not any(x['timeout'] for x in results),'Recorded prefix exceeds safety budget'

import pickle,time,json
from pathlib import Path
import numpy as np
from agent_code.team_agent.safety import safety_decision
from tests.test_proven_movement import V9
p=Path('runs/safety_stress_task4_5dbb4e7/safety_stress_task4_5dbb4e7_s24384/task4_failure_states.pkl')
row=pickle.loads(p.read_bytes())[-1];d=row['safety'];results=[]
for i in range(100):
 start=time.perf_counter()
 r=safety_decision(row['state'],np.asarray(d['physical_mask']),V9,allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']})
 results.append({'ms':1000*(time.perf_counter()-start),'timeout':r.robust_search_timed_out,'states':r.robust_states_evaluated,'mask':r.mask.tolist()})
print(json.dumps(results,indent=2),flush=True)
assert not any(x['timeout'] for x in results), 'Recorded world 24384 step 79 exceeds safety budget'

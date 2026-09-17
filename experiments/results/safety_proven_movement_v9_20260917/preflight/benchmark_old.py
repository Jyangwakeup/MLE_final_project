import json,time
from pathlib import Path
import numpy as np
from tests.test_vectorized_viability import records
from tests.test_certified_placement import failure_record
from tests.test_opponent_rearming import record
from tests.test_proven_movement import V9
from agent_code.team_agent.safety import safety_decision
rows=records()+[{**failure_record(i),'corpus':'baseline_failure'} for i in range(20)]+[{**record(i),'corpus':'round2_failure'} for i in range(165,185)]
results=[];skipped=[]
for r in rows:
 d=r['safety']
 if d['own_bomb_pending']:
  skipped.append([r['corpus'],r['state']['step']]);continue
 start=time.perf_counter();a=safety_decision(r['state'],np.asarray(d['physical_mask']),V9,allow_bomb=True,exploring=False,own_bomb_pending=False)
 results.append({'corpus':r['corpus'],'step':r['state']['step'],'ms':1000*(time.perf_counter()-start),'timeout':a.robust_search_timed_out,'mask':a.mask.tolist()})
out={'scope':'Nonpending subset only; old pending records lack a recorded placement clock and are not guessed. One sample, exploratory safety_decision timing, not act admission.','results':results,'skipped':skipped}
Path('runs/v9_old_nonpending_timing.json').write_text(json.dumps(out,indent=2)+'\n')
print('count',len(results),'skipped',len(skipped),'max_ms',max(r['ms'] for r in results),'timeouts',[(r['corpus'],r['step']) for r in results if r['timeout']],flush=True)

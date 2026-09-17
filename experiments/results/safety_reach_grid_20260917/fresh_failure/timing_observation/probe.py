"""Recorded-state timeout loop with observational CPU/GC timing only."""
import gc,json,pickle,time,resource,subprocess
from pathlib import Path
import numpy as np
from agent_code.team_agent.safety import safety_decision
from tests.test_proven_movement import V9
root=Path.cwd();out=root/'runs/repro_24384_timing_d65cebc';out.mkdir(exist_ok=False)
row=pickle.loads((root/'experiments/results/safety_reach_grid_20260917/fresh_failure/task4_failure_states.pkl').read_bytes())[-1];d=row['safety'];events=[];starts={}
def observe(phase,info):
 g=info['generation']
 if phase=='start':starts[g]=time.perf_counter()
 else:events.append({'generation':g,'ms':1000*(time.perf_counter()-starts.pop(g))})
gc.callbacks.append(observe)
results=[]
try:
 for repeat in range(500):
  events.clear();start=time.perf_counter();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_SELF)
  r=safety_decision(row['state'],np.asarray(d['physical_mask']),V9,allow_bomb=True,exploring=False,own_bomb_pending=False,own_bomb_state={'pending':False,'timer':None,'placed_step':None})
  wall=1000*(time.perf_counter()-start);cpu_ms=1000*(time.process_time()-cpu);after=resource.getrusage(resource.RUSAGE_SELF)
  item={'repeat':repeat,'ms':wall,'cpu_ms':cpu_ms,'gc':list(events),'voluntary_switches':after.ru_nvcsw-before.ru_nvcsw,'involuntary_switches':after.ru_nivcsw-before.ru_nivcsw,'minor_faults':after.ru_minflt-before.ru_minflt,'major_faults':after.ru_majflt-before.ru_majflt,'timeout':r.robust_search_timed_out,'states':r.robust_states_evaluated}
  results.append(item)
  with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
  if r.robust_search_timed_out or repeat%50==0:print(json.dumps(item),flush=True)
  if r.robust_search_timed_out:break
finally:gc.callbacks.remove(observe)
(out/'summary.json').write_text(json.dumps({'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'samples':len(results),'timeouts':sum(r['timeout'] for r in results),'max_ms':max(r['ms'] for r in results),'scope':'Repeated recorded state, CPU3, unchanged 400ms runtime safety budget; observational GC callback, not admission'},indent=2)+'\n')
assert not any(r['timeout'] for r in results),'Recorded safety timeout reproduced'

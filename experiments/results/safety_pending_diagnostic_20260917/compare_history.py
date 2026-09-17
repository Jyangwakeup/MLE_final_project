from pathlib import Path
import sys,json,hashlib,subprocess,importlib.util
from dataclasses import asdict
import numpy as np
root=Path('/export/data/sfan/MLE_final_project_safety_diagnostics');sys.path.insert(0,str(root))
from tests.test_vectorized_viability import records,CORPORA
from tests.test_order_equivalence import native
from tests.test_certified_placement import failure_record,V6
from agent_code.team_agent import safety as new
raw=subprocess.check_output(['git','show','c8c9770:agent_code/team_agent/safety.py'],cwd=root)
p=Path('/tmp/safety_before_pending_fix.py');p.write_bytes(raw)
name='agent_code.team_agent._safety_before_pending_fix';spec=importlib.util.spec_from_file_location(name,p);old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
result={'reference_commit':'c8c9770','candidate_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'reference_safety_sha256':hashlib.sha256(raw).hexdigest(),'candidate_safety_sha256':hashlib.sha256((root/'agent_code/team_agent/safety.py').read_bytes()).hexdigest(),'states':[]}
rs=records()+[{**failure_record(i),'corpus':'viability_cache_20260917/baseline_failure'} for i in range(20)]
for r in rs:
 d=r['safety'];state=r['state'];kw=dict(allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d.get('own_bomb_timer')});physical=np.asarray(d['physical_mask'],dtype=bool)
 before=native(state);a=old.safety_decision(state,physical,V6,**kw);b=new.safety_decision(state,physical,V6,**kw)
 assert native(state)==before
 assert not a.robust_search_timed_out and not b.robust_search_timed_out
 assert native(asdict(a))==native(asdict(b)),(r['corpus'],state['step'])
 result['states'].append(dict(corpus=r['corpus'],step=state['step'],mask=b.mask.tolist(),guarantee_loss=b.robust_guarantee_loss,scenario_counts=b.opponent_scenario_counts))
result['status']='passed';result['count']=len(rs)
for name in (*CORPORA,'viability_cache_20260917/baseline_failure'):
 path=root/'experiments/results'/name/'failure_states.json';result.setdefault('corpus_sha256',{})[name]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'experiments/results/safety_pending_diagnostic_20260917/historical_equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rs),'historical states; all masks and diagnostics unchanged')

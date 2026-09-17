from pathlib import Path
import os,sys,subprocess,json,hashlib,time
root=Path('/export/data/sfan/MLE_final_project_safety');sys.path.insert(0,str(root))
from experiments.run import _source_hash
from experiments.task4_campaign import engineering_checks
from experiments.task3_retention_prefix import summarize_evaluation
m_path=root/'experiments/safety_certified_v6_manifest.json';m=json.loads(m_path.read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert commit.startswith('425f622')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
assert _source_hash('double_dqn_continuous_v2_agent')==m['runtime_source_sha256']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/m['parent_checkpoint'])==m['parent_sha256'];assert sha(root/m['config'])==m['config_sha256']
assert sha(root/'agent_code/rule_based_agent/callbacks.py')==m['opponent_sha256']
for p,h in m['preflight'].items():assert sha(root/p)==h
out=root/'runs/certified_placement';state_path=out/'result_425f622.json';assert not state_path.exists()
state={'source_commit':commit,'manifest_sha256':sha(m_path),'status':'running','phases':{},'task4_qualified':False}
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
try:
 for phase,seeds in [('regression',m['regression_worlds']),('engineering',m['engineering_seeds'])]:
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit
  name=f'safety_v6_{phase}_425f622';target=root/'runs'/name;assert not target.exists()
  args=['taskset','-c','0','/export/data/sfan/miniforge3/envs/mle/bin/python','experiments/task4_worker.py','--config',m['config'],'--mode','evaluate','--device','cpu','--task','4','--agent','double_dqn_continuous_v2_agent','--seeds',*map(str,seeds),'--n-rounds','1','--checkpoint',str(root/m['parent_checkpoint']),'--run-id',name,'--replay-policy','all']
  state['current_phase']=phase;state['phases'][phase]={'command':args,'run':str(target)};save()
  with (out/(phase+'_425f622.log')).open('w') as f:r=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
  state['phases'][phase]['exit_code']=r.returncode
  failures=list(target.rglob('task4_safety_failure.json'))
  if r.returncode or failures:
   state.update(status='stopped_failure',failure_files=list(map(str,failures)));break
  summary=summarize_evaluation(target,'double_dqn_continuous_v2_agent',seeds)
  state['phases'][phase]['summary']=summary
  bad=engineering_checks({'task4':summary});state['phases'][phase]['engineering_failures']=bad
  if bad:state.update(status='stopped_failure');break
  save()
 else:state['status']='task4_engineering_passed'
except Exception as e:
 state.update(status='infrastructure_error',error=repr(e));raise
finally:save();print(state['status'],flush=True)

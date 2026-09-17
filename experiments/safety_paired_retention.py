"""Frozen v5/v7 engineering comparison; source-bound, paired, fail-fast."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,time,threading,signal,csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from experiments.run import _source_hash
from experiments.task4_campaign import engineering_checks
from experiments.task3_retention_prefix import summarize_evaluation
m_path=root/'experiments/safety_v7_retention_manifest.json';m=json.loads(m_path.read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert _source_hash('double_dqn_continuous_v2_agent')==m['runtime_source_sha256']
for p,h in [(m['config'],m['config_sha256']),(m['reference_config'],m['reference_config_sha256']),(m['parent_checkpoint'],m['parent_sha256'])]:assert sha(root/p)==h
out=root/'runs/certified_placement';state_path=out/f'retention_{commit[:7]}.json';assert not state_path.exists()
state={'source_commit':commit,'manifest_sha256':sha(m_path),'status':'running','jobs':{},'task4_qualified':False};stop=threading.Event()
def save():
 temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
def work(task,version,cpu):
 label=f'{version}_task{task}';name=f'safety_retention_{label}_{commit[:7]}';target=root/'runs'/name;assert not target.exists()
 config=m['config'] if version==m['candidate_version'] else m['reference_config'];program='experiments/task4_worker.py' if version==m['candidate_version'] else '-m'
 args=['taskset','-c',str(cpu),'/export/data/sfan/miniforge3/envs/mle/bin/python']+([program] if version==m['candidate_version'] else ['-m','experiments.run'])+['--config',config,'--mode','evaluate','--device','cpu','--task',str(task),'--agent','double_dqn_continuous_v2_agent','--seeds',*map(str,m['engineering_seeds']),'--n-rounds','1','--checkpoint',str(root/m['parent_checkpoint']),'--run-id',name,'--replay-policy','all']
 start=time.time();audited=set()
 def check_self_deaths():
  if version!=m['candidate_version']:return
  for path in target.glob('*/episodes.jsonl'):
   if path in audited:continue
   content=path.read_text()
   if not content or not content.endswith('\n'):continue
   for line in content.splitlines():
    episode=json.loads(line);own=next(a for a in episode['agents'] if a['name']=='double_dqn_continuous_v2_agent')
    if own['suicides']:raise RuntimeError(f'Unexplained candidate self death: {path}')
   audited.add(path)
 with (out/(label+f'_{commit[:7]}.log')).open('w') as f:
  p=subprocess.Popen(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while p.poll() is None:
    if stop.is_set() or time.time()-start>7200 or time.time()>=datetime.fromisoformat(m['deadline_utc'].replace('Z','+00:00')).timestamp():raise RuntimeError('Cancelled or timeout')
    check_self_deaths()
    time.sleep(.5)
   check_self_deaths()
   if p.returncode:raise RuntimeError(f'{label} exit {p.returncode}; see log and failure state')
  finally:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 summary=summarize_evaluation(target,'double_dqn_continuous_v2_agent',m['engineering_seeds'])
 episodes=[json.loads(line) for path in target.rglob('episodes.jsonl') for line in path.open() if line.strip()]
 summary['mean_bombs']=sum(a['bombs'] for e in episodes for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')/len(episodes)
 bad=engineering_checks({f'task{task}':summary}) if version==m['candidate_version'] else []
 if bad:raise RuntimeError(f'{label} engineering gates: {bad}')
 return label,{'task':task,'version':version,'run':str(target),'command':args,'summary':summary,'engineering_failures':bad}
try:
 save()
 with ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(work,task,version,i) for i,(task,version) in enumerate((t,v) for t in (1,2,3) for v in (m['reference_version'],m['candidate_version']))]
  try:
   for future in as_completed(futures):
    label,result=future.result();state['jobs'][label]=result;save();print(label,'done',flush=True)
  except BaseException:stop.set();raise
 checks={}
 for task,metric in ((1,'mean_score'),(2,'mean_coins'),(2,'mean_crates'),(3,'mean_score'),(3,'mean_coins'),(3,'mean_crates')):
  old=state['jobs'][f'{m["reference_version"]}_task{task}']['summary'][metric];new=state['jobs'][f'{m["candidate_version"]}_task{task}']['summary'][metric];ratio=new/old if old else 1.0
  checks[f'task{task}_{metric}']={'reference':old,'candidate':new,'ratio':ratio,'passed':ratio>=.9}
 for task in (1,2,3):
  s=state['jobs'][f'{m["candidate_version"]}_task{task}']['summary'];checks[f'task{task}_invalid']={'actual':s['invalid_action_rate'],'passed':s['invalid_action_rate']<=.01}
  if task>1:
   checks[f'task{task}_suicide']={'actual':s['suicide_rate'],'passed':s['suicide_rate']<=.05}
   checks[f'task{task}_bomb_survival']={'actual':s['bomb_survival_rate'],'passed':s['bomb_survival_rate']>=.95}
  if task==3:checks['task3_zero_bomb']={'actual':s['zero_bomb_round_rate'],'passed':s['zero_bomb_round_rate']<=.1}
 state.update(checks=checks,status='passed' if all(c['passed'] for c in checks.values()) else 'gate_failure')
except Exception as e:state.update(status='stopped_failure',error=repr(e));print(repr(e),flush=True)
finally:save();print(state['status'],flush=True)

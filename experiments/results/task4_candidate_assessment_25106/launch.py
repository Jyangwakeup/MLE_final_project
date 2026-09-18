"""Standalone candidate-only assessment, never resumes or qualifies the campaign."""
import csv,hashlib,json,os,pathlib,signal,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[3];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from experiments.task3_retention_prefix import summarize_evaluation
from experiments.task4_campaign import engineering_checks
from experiments.task4_protocol import CANDIDATE
OUT=ROOT/'runs/task4_candidate_assessment_25106';OUT.mkdir(exist_ok=False)
def save(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
old=ROOT/'runs/task4_reference_20260918_f3f17d7';worlds=list(range(25100,25200));jobs=[]
env=os.environ.copy();env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
for cpu,seed in enumerate([22,11,33]):
 h=json.loads((old/f'B/seed_{seed}.json').read_text())['selected'];assert sha(h['checkpoint'])==h['checkpoint_sha256']
 cmd=json.loads((old/'logs/task4_reference_20260918_B_badcb498_development_B_s33_c200_t4_f3f17d7.command.json').read_text())
 cmd[2]=str(cpu);a=cmd.index('--seeds');b=cmd.index('--n-rounds');cmd[a+1:b]=list(map(str,worlds))
 cmd[cmd.index('--checkpoint')+1]=h['checkpoint'];rid=f'task4_candidate_assessment_25106_s{seed}';cmd[cmd.index('--run-id')+1]=rid
 jobs.append(dict(seed=seed,rounds=h['rounds'],checkpoint=h['checkpoint'],sha256=h['checkpoint_sha256'],run_id=rid,command=cmd))
save('manifest.json',dict(kind='standalone_candidate_only_task4_assessment',worlds=worlds,world_scope='Previously registered confirmation worlds; reference partially observed; not a fresh validation set',source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),jobs=jobs,parent_comparison=False,task4_qualified=False))
state=dict(status='running',pid=os.getpid(),results={},task4_qualified=False);save('status.json',state)
active=[]
try:
 for j in jobs:
  log=(OUT/f"seed{j['seed']}.log").open('w');p=subprocess.Popen(j['command'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append((j,p,log))
 deadline=time.time()+4*3600
 while active:
  for j,p,log in list(active):
   rc=p.poll()
   if rc is None:continue
   log.close();active.remove((j,p,log))
   if rc:raise RuntimeError(f"seed{j['seed']} worker failed: {rc}")
   d=ROOT/'runs'/j['run_id'];s=summarize_evaluation(d,'double_dqn_continuous_v2_agent',worlds)
   failed=engineering_checks({'task4':s},CANDIDATE)
   strategy={'suicide_rate':s['suicide_rate']<=.05,'bomb_survival_rate':s['bomb_survival_rate']>=.95,'zero_bomb_round_rate':s['zero_bomb_round_rate']<=.1,'invalid_action_rate':s['invalid_action_rate']<=.01}
   state['results'][str(j['seed'])]=dict(summary=s,engineering_failures=failed,strategy_checks=strategy);save('status.json',state)
   if failed:raise RuntimeError(str(failed))
  if time.time()>deadline:raise RuntimeError('standalone 4-hour evaluation deadline exceeded')
  if active:time.sleep(5)
 state['status']='completed_candidate_assessment'
except BaseException as exc:
 state['status']='stopped';state['error']=str(exc)
 for j,p,log in active:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 for j,p,log in active:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close()
finally:
 save('status.json',state)
 (OUT/'report.md').write_text('# Standalone Task4 candidate assessment\n\n'+json.dumps(state,indent=2)+'\n\nThis is not completed campaign confirmation or Task4 qualification. No complete paired parent baseline exists on these worlds.\n')

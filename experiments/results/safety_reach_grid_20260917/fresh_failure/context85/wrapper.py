"""Diagnostic wrapper: observe actual act CPU/GC time, preserve worker fail-fast."""
import gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from experiments import task4_worker
from agent_code.double_dqn_continuous_v2_agent import callbacks
from experiments.run import _source_hash
out=ROOT/'runs/gc_context_7efa03b';out.mkdir(exist_ok=False)
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
manifest=json.loads((ROOT/'experiments/safety_v9_grid_fresh_manifest.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert _source_hash(task4_worker.AGENT)==manifest['runtime_source_sha256']
for p,s in {manifest['config']:manifest['config_sha256'],manifest['parent_checkpoint']:manifest['parent_sha256'],**manifest['opponent_hashes']}.items():assert sha(ROOT/p)==s
argv=['--config',manifest['config'],'--mode','evaluate','--device','cpu','--task','4','--agent',task4_worker.AGENT,'--seeds',*map(str,range(24300,24385)),'--n-rounds','1','--checkpoint',str(ROOT/manifest['parent_checkpoint']),'--run-id','gc_context_7efa03b_worlds','--replay-policy','all']
identity={'scope':'Diagnostic original-prefix replay with observational timing only. Original sweep remains terminal; not qualification or new validation data.','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'wrapper_sha256':sha(__file__),'runtime_sha256':manifest['runtime_source_sha256'],'config_sha256':manifest['config_sha256'],'parent_sha256':manifest['parent_sha256'],'opponent_hashes':manifest['opponent_hashes'],'cpu_affinity':sorted(os.sched_getaffinity(0)),'argv':argv,'deadline_utc':manifest['deadline_utc']}
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
active=False;events=[];starts={};latest=None
original_act=callbacks.act;original_append=task4_worker.run.ExperimentWorld._append_timing
calls=0;maximum=0.
def observe(phase,info):
 if not active:return
 g=info['generation']
 if phase=='start':starts[g]=time.perf_counter()
 elif g in starts:events.append({'generation':g,'ms':1000*(time.perf_counter()-starts.pop(g))})
def act(owner,state):
 global active,latest,calls,maximum
 events.clear();starts.clear();wall=time.perf_counter();cpu=time.process_time();active=True
 try:return original_act(owner,state)
 finally:
  active=False;ms=1000*(time.perf_counter()-wall);cpu_ms=1000*(time.process_time()-cpu);calls+=1;maximum=max(maximum,ms)
  latest={'step':state['step'],'ms':ms,'cpu_ms':cpu_ms,'gc':list(events),'calls':calls,'max_ms':maximum}
def append(world,agent,*args,**kwargs):
 if agent.name==task4_worker.AGENT and latest is not None:
  if latest['ms']>=250 or any(e['generation']==2 for e in latest['gc']):
   row={**latest,'run_id':world._experiment_run_id}
   with (out/'slow_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 return original_append(world,agent,*args,**kwargs)
callbacks.act=act;task4_worker.run.ExperimentWorld._append_timing=append;gc.callbacks.append(observe)
code=None
try:code=task4_worker.main(argv)
finally:
 gc.callbacks.remove(observe);callbacks.act=original_act;task4_worker.run.ExperimentWorld._append_timing=original_append
 (out/'result.json').write_text(json.dumps({'returncode':code,'calls':calls,'max_ms':maximum,'scope':identity['scope']},indent=2)+'\n')
raise SystemExit(code)

"""Freeze verified historical paired retention, not a new validation verdict."""
import hashlib,json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path.cwd()))
from experiments.run import _source_hash
from experiments.task4_transfer import sha256,PARENT_SHA256,PARENT_SAFETY,TARGET_SAFETY
from experiments.task3_retention_prefix import summarize_evaluation
root=pathlib.Path.cwd();out=root/'experiments/results/task4_v9_reference_20260918/retention_evidence.json';files={};result={'verified':False,'files':files,'checks':{}}
try:
 oldpath=root/'experiments/results/compact_admission_20260917/frozen/controller/result.json';old=json.loads(oldpath.read_text());files[str(oldpath)]=sha256(oldpath)
 runtime=_source_hash('double_dqn_continuous_v2_agent')
 assert old['runtime_sha256']==runtime
 for f in ['experiments/run.py','experiments/compare_evaluations.py','experiments/task3_retention_prefix.py']:
  historical=subprocess.check_output(['git','show',old['source_commit']+':'+f]);assert hashlib.sha256(historical).hexdigest()==sha256(root/f)
  files[str(root/f)]=sha256(root/f)
 for task,metrics in [(1,['mean_score']),(2,['mean_coins','mean_crates']),(3,['mean_score','mean_coins','mean_crates'])]:
  summaries={}
  for role in ['parent','child']:
   job=old['jobs'][f'retention_{task}_{role}'];directory=pathlib.Path(job['run']);seen=[]
   for meta in sorted(directory.glob('*/metadata.json')):
    m=json.loads(meta.read_text());seen.append(m['seed']);assert m['status']=='completed' and m['exploration_disabled'] and m['source_hash']==runtime
    assert m['safety_spec']==(PARENT_SAFETY if role=='parent' else TARGET_SAFETY)
    assert m['termination']['local_completed_rounds']==1
    assert sha256(m['checkpoint'])==PARENT_SHA256
    assert sha256(m['config_path'])==m['config_source_sha256']
    for f in [meta,meta.parent/'episodes.jsonl',pathlib.Path(m['config_path']),pathlib.Path(m['checkpoint'])]:files[str(f)]=sha256(f)
   assert sorted(seen)==list(range(24000,24060))
   for f in (directory/(directory.name+'_summary')).glob('*'):
    if f.is_file():files[str(f)]=sha256(f)
   summaries[role]=summarize_evaluation(directory,'double_dqn_continuous_v2_agent',list(range(24000,24060)))
   for metric in metrics:assert summaries[role][metric]==job['summary'][metric]
  result['checks'][str(task)]={k:summaries['child'][k]/summaries['parent'][k] if summaries['parent'][k] else 1. for k in metrics}
  assert all(x>=.9 for x in result['checks'][str(task)].values())
 result.update(verified=True,runtime_source_sha256=runtime,bootstrap=old['bootstrap'],worlds=list(range(24000,24060)),scope='historical engineering retention only')
except Exception as e:result['reason']=repr(e)
out.write_text(json.dumps(result,indent=2)+'\n');print('verified',result['verified'],result.get('reason'), 'files',len(files))

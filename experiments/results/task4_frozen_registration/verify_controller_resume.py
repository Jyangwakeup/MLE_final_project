import sys,json,time,tempfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.task4_frozen_campaign import Campaign
r=Path.cwd();manifest=json.loads((r/'experiments/task4_frozen_manifest.json').read_text());results={}
for resumed in (False,True):
 with tempfile.TemporaryDirectory() as tmp:
  obj=Campaign.__new__(Campaign);obj.m=manifest;obj.start=time.time();obj.check_identity=lambda:None
  obj.directory=Path(tmp);obj.resuming=resumed;obj.state={};obj.update=lambda status,**values:obj.state.update(status=status,**values)
  obj.unique=lambda label:'frozen_resume_continuous'
  commands=[];obj.execute=lambda *args:commands.append(args)
  if resumed:obj.state['training_attempts']={'S_990022_4800':[dict(run='frozen_resume_continuous',started_at=time.time()-240,finished_at=time.time(),committed_round=12)]}
  record=obj.train('S',990022,4800)
  assert record['actual_actions']==4800 and record['rounds']==12
  assert len(commands)==(0 if resumed else 1)
  assert not record['raw']['failures']
  results['resumed' if resumed else 'fresh']=dict(actions=record['actual_actions'],rounds=record['rounds'],exposure=record['exposure'],raw=record['raw']['summary'],command_count=len(commands))
Path('experiments/results/task4_frozen_registration/controller_resume_verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

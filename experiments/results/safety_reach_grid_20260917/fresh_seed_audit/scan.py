import subprocess,json,hashlib,datetime
from pathlib import Path
roots=[Path(l[9:]) for l in subprocess.check_output(['git','worktree','list','--porcelain'],text=True).splitlines() if l.startswith('worktree ')]
used=set();intervals=[];records=[];errors=[]
def collect(v, key=''):
 if isinstance(v,dict):
  for k,x in v.items():
   if 'seed' in k.lower():
    def nums(y):
     if isinstance(y,bool):return
     if isinstance(y,int):used.add(y)
     elif isinstance(y,list):
      for q in y:nums(q)
     elif isinstance(y,dict):
      for q in y.values():nums(q)
    nums(x)
    if 'range' in k.lower() and isinstance(x,list) and len(x)==2 and all(isinstance(q,int) for q in x):intervals.append(tuple(x))
   collect(x,k)
 elif isinstance(v,list):
  for x in v:collect(x,key)
for root in roots:
 paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','*.json','-g','!.git/**','-g','!node_modules/**','-g','!.venv/**'],cwd=root,text=True).splitlines()
 for relative in paths:
  p=root/relative
  try:
   raw=p.read_bytes();data=json.loads(raw)
  except (OSError,ValueError) as e:
   errors.append([str(p),type(e).__name__]);continue
  collect(data);records.append([str(p),hashlib.sha256(raw).hexdigest()])
candidates=[]
for first in range(24300,26000,100):
 band=set(range(first,first+100))
 if not band&used and not any(a<=first+99 and b>=first for a,b in intervals):candidates.append([first,first+99])
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worktrees':list(map(str,roots)),'files_scanned':len(records),'source_records_sha256':hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest(),'used_seed_count':len(used),'candidate_ranges':candidates,'read_errors':errors,'sealed_range':[20000,20099]}
Path('/tmp/next_engineering_seed_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='worktrees'},indent=2))

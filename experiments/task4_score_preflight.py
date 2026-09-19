"""Real callbacks: archived-C action equivalence and exact LPK round-boundary resume."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.task4_transfer import sha256,digest
from experiments.resume import load_training_snapshot
from experiments.task4_score_campaign import write,THREADS
import torch
import numpy as np


def equal(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.equal(a,b):raise AssertionError(path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path+' keys')
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError(path+' len')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    elif a!=b:raise AssertionError(path+f': {a!r} != {b!r}')


def rows(directory,limit=None):
    selected=[]
    for line in (directory/'timing.jsonl').open():
        r=json.loads(line)
        if limit and r['round_index']>limit:break
        selected.append({k:r.get(k) for k in ('round_index','step','agent_name','action','requested_action','safety','navigation')})
    return selected


def main():
    m=json.loads((ROOT/'experiments/task4_score_manifest.json').read_text())
    directory=ROOT/'runs'/m['campaign_id']/'preflight';directory.mkdir(parents=True,exist_ok=True)
    prefix='task4_score_preflight_'+str(int(time.time()))
    commands=[]
    def run(tag,arm,seed,rounds,cpu,previous=None):
        name=prefix+'_'+tag
        args=['taskset','-c',str(cpu),sys.executable,'experiments/task4_score_worker.py','--frozen-manifest','experiments/task4_score_manifest.json',
              '--role','candidate','--match-kind','training','--config',m['arm_configs'][arm],'--mode','train','--device','cpu','--task','4',
              '--agent',m['agent'],'--opponents',*(['frozen_history_agent']*3),'--seed',str(seed),'--n-rounds',str(rounds),
              '--target-stage-action-steps','60000','--run-id',name,'--replay-policy','all']
        args+=['--resume-from',str(previous)] if previous else ['--transfer-task4-score-from-checkpoint',str(ROOT/m['parent']['checkpoint'])]
        write(directory/(tag+'.command.json'),args);commands.append(args)
        with (directory/(tag+'.log')).open('w') as log:
            subprocess.run(args,cwd=ROOT,env={**os.environ,**THREADS},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        return ROOT/'runs'/name
    cpus=m['cpus']
    if len(cpus)>=3:
        with ThreadPoolExecutor(max_workers=3) as pool:
            c=pool.submit(run,'C8','C',22,8,cpus[0]);whole=pool.submit(run,'LPK8','LPK',903022,8,cpus[1]);half=pool.submit(run,'LPK4','LPK',903022,4,cpus[2])
            c,whole,half=c.result(),whole.result(),half.result()
    else:
        c=run('C8','C',22,8,cpus[0]);whole=run('LPK8','LPK',903022,8,cpus[0]);half=run('LPK4','LPK',903022,4,cpus[0])
    rest=run('LPK4restore','LPK',903022,4,cpus[0],half)
    a=load_training_snapshot(whole);b=load_training_snapshot(rest)
    pa=torch.load(a.learner_path,weights_only=True,map_location='cpu');pb=torch.load(b.learner_path,weights_only=True,map_location='cpu')
    equal(pa,pb)
    equal(rows(whole),rows(half)+rows(rest))
    equal([json.loads(x) for x in (whole/'opponent_schedule.jsonl').read_text().splitlines()],
          [json.loads(x) for d in (half,rest) for x in (d/'opponent_schedule.jsonl').read_text().splitlines()])
    old=ROOT.parent/'MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s22_60000'
    equal(rows(c),rows(old,8))
    if pa['updates']<=0:raise AssertionError('Restore did not cross warmup')
    record=dict(passed=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),manifest_sha256=digest(m),
                whole=str(whole),prefix=str(half),restored=str(rest),control=str(c),
                actions=pa['stage_action_steps'],updates=pa['updates'],replay=pa['replay']['occupancy'],
                matched_decisions=len(rows(whole)),control_matched_decisions=len(rows(c)),
                checkpoint_hashes={str(a.learner_path):sha256(a.learner_path),str(b.learner_path):sha256(b.learner_path)})
    write(directory/'result.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':main()

"""One-shot registration; never run again after the campaign is frozen."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.task4_transfer import digest,sha256,PARENT_SHA256
from experiments.task4_score_transfer import VERSION,retention
from experiments.frozen_opponents import stream_seed

def collect(value, context=False):
    out=set()
    if isinstance(value,dict):
        for k,v in value.items():out |= collect(v,context or 'seed' in k.lower() or 'rng' in k.lower())
    elif isinstance(value,list):
        for v in value:out |= collect(v,context)
    elif context and type(value) is int:out.add(value)
    return out

def main():
    import torch
    destination=ROOT/'experiments/task4_score_manifest.json'
    if destination.exists():raise ValueError('Registration is immutable; use existing manifest')
    oldroot=ROOT.parent/'MLE_final_project_task4_frozen'
    old=json.loads((oldroot/'experiments/task4_frozen_manifest.json').read_text())
    used=set();scan=[]
    trees=[line.split(' ',1)[1] for line in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True).splitlines() if line.startswith('worktree ')]
    for tree in trees:
        result=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*.json','-g','!.git/**',tree],text=True,capture_output=True,check=True)
        for name in result.stdout.splitlines():
            path=Path(name)
            if not ('config' in name or 'manifest' in path.name or path.name=='metadata.json' or 'seed' in path.name):continue
            try:value=json.loads(path.read_text())
            except (ValueError,OSError):continue
            seeds=collect(value)
            if seeds:used.update(seeds);scan.append(dict(path=name,sha256=sha256(path),seeds=sorted(seeds)))
    used.update(range(20000,20100))
    registered=set()
    def allocate(blocks,count):
        start=27000
        while set(range(start,start+blocks*100)) & (used|registered):start+=100
        registered.update(range(start,start+blocks*100))
        return list(range(start,start+count))
    diagnostic=allocate(1,20);development=allocate(2,200);final=allocate(4,400)
    diagnostic_training=dict(L=901022,LP=902022,LPK=903022)
    rng={str(seed):dict(learning=seed,environment=1000+seed,official_opponent=3000+seed,replay=seed,torch=seed,distillation=seed+104729) for seed in [22,11,33,*diagnostic_training.values()]}
    # Preserve historical schedule/world/opponent streams to isolate learning changes.
    for seed in [22,11,33,*diagnostic_training.values()]:
        rng[str(seed)]['schedule']=stream_seed(old['schedule_seed'],seed)
    if set(sum([diagnostic,development,final],[])) & set(v for g in rng.values() for v in g.values()):raise ValueError('Training stream overlap')
    cpus=[];physical=set();allowed=os.sched_getaffinity(0)
    for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines():
        if line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(','));key=(socket,core)
        if cpu in allowed and key not in physical:cpus.append(cpu);physical.add(key)
        if len(cpus)==6:break
    scanpath='experiments/results/task4_score_registration/seed_scan.json.gz'
    (ROOT/scanpath).parent.mkdir(parents=True,exist_ok=True)
    (ROOT/scanpath).write_bytes(gzip.compress((json.dumps(dict(records=scan,allocated=dict(diagnostic=diagnostic,development=development,final=final)),indent=2)+'\n').encode(),mtime=0))
    m=dict(old,source_base='312cc82d',base_commit='312cc82d',schema_version=VERSION,campaign_id='task4_score_20260920',started_at='2026-09-20T00:18:28+02:00',
           cpus=cpus,diagnostic_seeds=diagnostic,development_seeds=development,final_seeds=final,
           diagnostic_training_seeds=diagnostic_training,rng_plan=rng,excluded_seeds=sorted(used),
           seed_scan=scanpath,seed_scan_sha256=sha256(ROOT/scanpath),models={},pool=[],heldout_ids=[],pool_sha256=digest({}),
           reference_config='experiments/configs/task4_score_reference.json',archive_config='experiments/configs/task4_score_archive.json',
           arm_configs={a:f'experiments/configs/task4_score_{a}.json' for a in ('C','L','LP','LPK')},archived_controls={})
    m.pop('heldout_seeds',None)
    m['archived_source']=dict(commit='4e2b836d4b8482b220de5686ed4e25a08ca2f407',records_commit='312cc82d',manifest_sha256=digest(old))
    for path,h in {**old['opponent_hashes'],**old['algorithm_hashes']}.items():
        if sha256(ROOT/path)!=h:raise ValueError('Runtime/opponent changed before registration: '+path)
    m['algorithm_hashes']=dict(old['algorithm_hashes'])
    for name in ('agent_code/team_agent/rewards.py','agent_code/team_agent/continuous_features.py'):
        if (ROOT/name).exists():m['algorithm_hashes'][name]=sha256(ROOT/name)
    if sha256(ROOT/m['parent']['checkpoint'])!=PARENT_SHA256:raise ValueError('Parent mismatch')
    for seed in (22,11):
        run=oldroot/f'runs/task4_frozen_20260919_C_s{seed}_60000';base=run/'checkpoints/snapshots'
        metadata=json.loads((run/'metadata.json').read_text())
        rows=list(csv.DictReader((run/'training.csv').open()))
        entries=[]
        for line in (base/'manifest.jsonl').read_text().splitlines():
            record=json.loads(line);path=base/record['checkpoint'];payload=torch.load(path,map_location='cpu',weights_only=True)
            actual=payload['stage_action_steps'];contract=payload['transfer_contract']
            if (contract['manifest_sha256']!=digest(old) or contract['config_sha256']!=sha256(oldroot/old['arm_configs']['C'])
                    or contract['opponent_rng_root']!=m['opponent_rng_root'] or contract['schedule_seed']!=m['schedule_seed']
                    or contract['source_commit']!=m['archived_source']['commit'] or payload['agent_seed']!=seed):raise ValueError('Old control identity mismatch')
            entries.append(dict(checkpoint=str(path),checkpoint_sha256=sha256(path),actual_actions=actual,
               rounds=int(next(r['round'] for r in rows if int(r['stage_action_steps'])==actual)),seed=seed,arm='C',
               hyperparameters=payload['hyperparameters'],transfer_contract=contract,metadata_path=str(run/'metadata.json'),metadata_sha256=sha256(run/'metadata.json')))
        m['archived_controls'][str(seed)]=entries
    base=json.loads((oldroot/old['arm_configs']['C']).read_text())
    for arm,path in m['arm_configs'].items():
        c=json.loads(json.dumps(base));c['learning']['learning_rate']=1e-4 if arm=='C' else 3e-5
        c['reward_id']='r7_safe_credit_potential' if arm in ('LP','LPK') else 'r7_safe_credit_sparse'
        c['training']['retention']=retention(arm);c['evaluation']['seeds']=development
        c['score_contract']=dict(version=VERSION,role='learner',arm=arm,manifest=str(destination.relative_to(ROOT)))
        c['frozen_opponents']=dict(base['frozen_opponents'],arm='C',role='reference',manifest=str(destination.relative_to(ROOT)))
        (ROOT/path).write_text(json.dumps(c,indent=2)+'\n')
    c=json.loads((ROOT/m['arm_configs']['C']).read_text());c['score_contract']['role']='archive'
    (ROOT/m['archive_config']).write_text(json.dumps(c,indent=2)+'\n')
    c=json.loads((oldroot/old['reference_config']).read_text());c['evaluation']['seeds']=development
    c['frozen_opponents']['manifest']=str(destination.relative_to(ROOT))
    c['score_contract']=dict(version=VERSION,role='reference',manifest=str(destination.relative_to(ROOT)))
    (ROOT/m['reference_config']).write_text(json.dumps(c,indent=2)+'\n')
    destination.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(cpus=cpus,diagnostic=[diagnostic[0],diagnostic[-1]],development=[development[0],development[-1]],final=[final[0],final[-1]],scanned=len(scan))))

if __name__=='__main__':main()

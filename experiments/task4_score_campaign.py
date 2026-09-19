"""Fixed-endpoint historical-opponent controlled campaign."""
from __future__ import annotations
import argparse
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.task4_transfer import sha256, digest, PARENT_SHA256
from experiments.task4_score_transfer import VERSION
from experiments.task4_campaign import audit_training, EngineeringFailure, engineering_checks
from experiments.task4_protocol import CANDIDATE
from experiments.compact_audit import audit_run
from experiments.compare_evaluations import compare_evaluations

BEHAVIOR = {'suicide_rate','bomb_survival_rate','invalid_action_rate'}
TERMINAL = {'observed_improvement','no_score_improvement','final_failed','engineering_failure','budget_incomplete','screen_failed'}
THREADS = {k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def copy_attempts(attempts):
    return json.loads(json.dumps(attempts))


def failures(raw):
    # Bomb usage and crate yield remain diagnostics, never elimination criteria.
    return [f for f in raw['failures'] if f != 'zero_bomb_round_rate']


def cutoff(start, *, kind, now=None):
    hours={'arm':18,'selection':20,'evaluation':23,'work':24}[kind]
    if (time.time() if now is None else now) >= start + hours*3600:
        raise TimeoutError('Frozen-opponent '+kind+' deadline')


def validate_manifest(m):
    if m['schema_version'] != VERSION or set(m['arm_configs']) != {'C','L','LP','LPK'}:
        raise ValueError('Score protocol/arms mismatch')
    if m['training_seeds'] != [22,11,33] or m['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Score seeds/parent mismatch')
    groups=[m[k+'_seeds'] for k in ('diagnostic','development','final')]
    if list(map(len,groups)) != [20,200,400] or len(set(sum(groups,[]))) != 620:
        raise ValueError('Score data partition mismatch')
    forbidden=set(m['excluded_seeds']) | set(range(20000,20100))
    for values in m['rng_plan'].values():forbidden.update(values.values())
    if set(sum(groups,[])) & forbidden:raise ValueError('Reserved world overlap')
    if m['final_seeds'] != list(range(m['final_seeds'][0],m['final_seeds'][0]+400)):
        raise ValueError('Final worlds must be contiguous')
    if not 1<=len(m['cpus'])<=6 or len(set(m['cpus']))!=len(m['cpus']):raise ValueError('Invalid cores')
    if m['limits'] != dict(p95_seconds=.1,max_seconds=.3,search_budget_ms=400):raise ValueError('Limits changed')
    return m


class Campaign:
    def __init__(self, manifest, resume=False):
        self.path=Path(manifest).resolve();self.m=validate_manifest(json.loads(self.path.read_text()))
        self.root=ROOT
        self.commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.start=datetime.fromisoformat(self.m['started_at']).timestamp()
        self.directory=ROOT/'runs'/self.m['campaign_id']
        self.state_path=self.directory/'status.json'
        self.identity=dict(source_commit=self.commit,manifest_sha256=digest(self.m),
            config_hashes={p:sha256(ROOT/p) for p in [*self.m['arm_configs'].values(),self.m['reference_config'],self.m['archive_config']]},
            parent_sha256=PARENT_SHA256,pool_sha256=self.m['pool_sha256'],schedule_version=self.m['schedule_version'])
        self.cancel=threading.Event()
        self.resuming=resume
        if self.state_path.exists():
            if not resume:raise ValueError('Existing campaign requires explicit infrastructure resume')
            self.state=json.loads(self.state_path.read_text())
            if self.state['identity']!=self.identity or self.state['status'] in TERMINAL:
                raise ValueError('Cannot resume terminal or different experiment')
        else:
            proof=json.loads((self.directory/'preflight/result.json').read_text())
            if not proof.get('passed') or proof['manifest_sha256']!=digest(self.m):raise ValueError('Missing/mismatched real callback preflight')
            from experiments.run import _source_hash
            proof_metadata=json.loads((Path(proof['whole'])/'metadata.json').read_text())
            if proof_metadata['source_hash']!=_source_hash(self.m['agent']):raise ValueError('Runtime changed since preflight')
            tests=json.loads((self.directory/'test_results.json').read_text())
            if not tests['passed'] or any(sha256(p)!=h for p,h in tests['logs'].items()):raise ValueError('Missing or changed test evidence')
            self.state=dict(status='created',identity=self.identity,arms={},diagnostics={},task4_qualified=False,preflight=proof,tests=tests)
        self.check_identity()

    def check_identity(self):
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=self.commit:
            raise ValueError('Source commit changed')
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise ValueError('Experiment source must be clean')
        if digest(json.loads(self.path.read_text()))!=self.identity['manifest_sha256']:
            raise ValueError('Manifest changed')
        if sha256(ROOT/self.m['seed_scan'])!=self.m['seed_scan_sha256']:raise ValueError('Seed registration evidence changed')
        for path,expected in {**self.identity['config_hashes'],**self.m['opponent_hashes'],**self.m['algorithm_hashes']}.items():
            if sha256(ROOT/path)!=expected:raise ValueError('Frozen file changed: '+path)
        if sha256(ROOT/self.m['parent']['checkpoint'])!=PARENT_SHA256:raise ValueError('Parent changed')
        for group in self.m['archived_controls'].values():
            for entry in group:
                if sha256(entry['checkpoint'])!=entry['checkpoint_sha256'] or sha256(entry['metadata_path'])!=entry['metadata_sha256']:raise ValueError('Archived control changed')

    def update(self, status, **values):
        self.state.update(status=status,updated_at=datetime.now().astimezone().isoformat(),**values)
        write(self.state_path,self.state)
        print(json.dumps(dict(status=status,**values)),flush=True)

    def execute(self, args, name, cpu, role, kind):
        cutoff(self.start,kind='work');self.check_identity()
        log=self.directory/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        command=['taskset','-c',str(cpu),sys.executable,'experiments/task4_score_worker.py',
                 '--frozen-manifest',str(self.path),'--role',role,'--match-kind',kind,*args]
        write(log.with_suffix('.command.json'),command)
        with log.open('w') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**THREADS},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(log.with_suffix('.pid.json'),dict(pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command))
            try:
                while proc.poll() is None:
                    if self.cancel.is_set():raise EngineeringFailure('Cancelled after peer failure')
                    cutoff(self.start,kind='work')
                    if 'final_candidate' not in self.state:cutoff(self.start,kind='selection')
                    time.sleep(.5)
                if proc.returncode:
                    if proc.returncode<0:raise OSError('Infrastructure process interruption: '+str(log))
                    raise EngineeringFailure('Worker failed; inspect '+str(log))
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise

    def unique(self, name):
        base=f"{self.m['campaign_id']}_{name}";i=0
        while (ROOT/'runs'/(base if not i else base+f'_retry{i}')).exists():i+=1
        return base if not i else base+f'_retry{i}'

    def audit(self, directory, diagnostic=False):
        raw=audit_run(directory,4,policy=CANDIDATE)
        from experiments.task4_frozen_audit import audit_opponents
        raw['frozen_opponents']=audit_opponents(directory)
        raw['failures']+=raw['frozen_opponents']['failures']
        bad=failures(raw)
        write(Path(directory)/'frozen_audit.json',raw)
        fatal=bad if diagnostic else [f for f in bad if f not in BEHAVIOR]
        if fatal:raise EngineeringFailure(str(directory)+': '+str(fatal))
        return raw

    def evaluate(self, checkpoint, arm, label, worlds, role='candidate', kind='rules'):
        started=time.monotonic()
        cutoff(self.start,kind='evaluation');self.check_identity()
        config=self.m['reference_config'] if role=='reference' else self.m['archive_config'] if role=='archive' else self.m['arm_configs'][arm]
        request=dict(identity=self.identity,role=role,kind=kind,arm=arm,checkpoint_sha256=sha256(checkpoint),worlds=worlds,config=config,safety=self.m['target_safety'])
        cache=self.directory/'evaluations'/(label+'.json')
        if cache.exists():
            value=json.loads(cache.read_text())
            if value['request']!=request:raise ValueError('Evaluation cache identity mismatch')
            if any(sha256(p)!=h for p,h in value['artifact_hashes'].items()):raise ValueError('Evaluation cache artifact changed')
            if role=='reference' and failures(value['raw']):
                self.state.setdefault('failed_evaluations',[]).append(dict(label=label,result=value))
                raise EngineeringFailure('Cached fixed endpoint failed: '+str(cache))
            return value
        # Disjoint worlds on disjoint physical cores; parent/child use identical seeds.
        chunks=[worlds[i::len(self.m['cpus'])] for i in range(len(self.m['cpus']))]
        jobs=[]
        for i,chunk in enumerate(chunks):
            name=self.unique(f'{label}_p{i}')
            args=['--config',config,'--mode','evaluate','--device','cpu','--task','4','--agent',self.m['agent'],
                  '--opponents',*(['frozen_history_agent']*3),'--seeds',*map(str,chunk),'--n-rounds','1',
                  '--checkpoint',str(checkpoint),'--run-id',name,'--replay-policy','all']
            jobs.append((name,chunk,args,self.m['cpus'][i]))
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures=[pool.submit(self.execute,args,name,cpu,role,kind) for name,chunk,args,cpu in jobs]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:self.cancel.set();raise
        # Present completed world runs under one stable directory for existing paired statistics.
        merged=self.directory/'worlds'/label
        retry=0
        while merged.exists():
            retry+=1;merged=self.directory/'worlds'/f'{label}_retry{retry}'
        merged.mkdir(parents=True,exist_ok=False)
        directories=[]
        for name,chunk,_,_ in jobs:
            for path in sorted((ROOT/'runs'/name).glob('*/episodes.jsonl')):
                dest=merged/path.parent.name
                if dest.exists():raise ValueError('Duplicate evaluation world')
                dest.symlink_to(path.parent,target_is_directory=True);directories.append(str(path.parent))
        raw=self.audit(merged)
        episodes=[json.loads(line) for d in directories for line in (Path(d)/'episodes.jsonl').read_text().splitlines()]
        if len(episodes)!=len(worlds) or {e['seed'] for e in episodes}!=set(worlds):
            raise ValueError('Incomplete or mismatched evaluation worlds')
        own=[next(a for a in e['agents'] if a['name']==self.m['agent']) for e in episodes]
        summary=dict(raw['summary'])
        for metric,field in [('mean_score','score'),('mean_coins','coins'),('mean_crates','crates'),('mean_kills','kills')]:
            summary[metric]=sum(a[field] for a in own)/len(own)
        summary['exclusive_first_rate']=sum(a['score']==max(x['score'] for x in e['agents']) and sum(x['score']==a['score'] for x in e['agents'])==1 for a,e in zip(own,episodes))/len(own)
        summary['tied_first_rate']=sum(a['score']==max(x['score'] for x in e['agents']) and sum(x['score']==a['score'] for x in e['agents'])>1 for a,e in zip(own,episodes))/len(own)
        summary['first_place_rate']=sum(a['score']==max(x['score'] for x in e['agents']) for a,e in zip(own,episodes))/len(own)
        result=dict(request=request,summary=summary,raw=raw,directories=directories,wall_seconds=time.monotonic()-started,
                    artifact_hashes={str(Path(d)/f):sha256(Path(d)/f) for d in directories for f in ('metadata.json','episodes.jsonl','timing.jsonl','opponent_schedule.jsonl','q_probes.jsonl') if (Path(d)/f).exists()})
        write(cache,result)
        if role=='reference' and failures(raw):
            self.state.setdefault('failed_evaluations',[]).append(dict(label=label,result=result))
            raise EngineeringFailure('Fixed endpoint evaluation failed: '+str(cache)+': '+str(failures(raw)))
        return result

    def compare(self, child, parent, label):
        result=compare_evaluations(child['directories'],parent['directories'],self.directory/'comparisons'/label,bootstrap_samples=10000)
        from experiments.compare_evaluations import _load_group, _bootstrap_ci
        _,_,a=_load_group(child['directories']);_,_,b=_load_group(parent['directories'])
        differences=[a[k]['exclusive_win']+a[k]['tied_first']-b[k]['exclusive_win']-b[k]['tied_first'] for k in sorted(a)]
        low,high=_bootstrap_ci(differences,'first_place',10000)
        result['first_place_difference']={'mean':sum(differences)/len(differences),'ci95':[low,high]}
        write(self.directory/'comparisons'/label/'frozen_comparison.json',result)
        return result

    def train(self, arm, seed, target, previous=None, diagnostic=False):
        started=time.monotonic()
        cutoff(self.start,kind='work');self.check_identity()
        from experiments.resume import load_training_snapshot
        from experiments.task4_score_transfer import validate_resume
        import torch
        key=f'{arm}_{seed}_{"diag" if diagnostic else target}'
        attempts=self.state.setdefault('training_attempts',{}).setdefault(key,[])
        if previous and not attempts:
            prior=self.state['trained_points'][arm][str(seed)]
            inherited=next(v for v in prior.values() if v['run']==previous)
            attempts.extend(copy_attempts(inherited['segments']))
        complete=False;previous_round=0
        if attempts:
            if not self.resuming and previous is None:raise ValueError('An existing training attempt requires infrastructure resume')
            last=ROOT/'runs'/attempts[-1]['run']
            if (last/'resume/latest.json').exists():
                saved=load_training_snapshot(last)
                validate_resume(saved.contract['transfer_contract'],root=ROOT,config_path=ROOT/self.m['arm_configs'][arm],seed=seed)
                payload=torch.load(saved.learner_path,map_location='cpu',weights_only=True)
                previous_round=saved.round_index;previous=str(last)
                complete=(previous_round>=20 if diagnostic else payload['stage_action_steps']>=target)
                del payload
                attempts[-1]['committed_round']=previous_round
                if complete:directory=last
        if not complete:
            name=self.unique(f'{arm}_s{seed}_{"diag" if diagnostic else target}')
            args=['--config',self.m['arm_configs'][arm],'--mode','train','--device','cpu','--task','4',
                  '--agent',self.m['agent'],'--opponents',*(['frozen_history_agent']*3),'--seed',str(seed),
                  '--n-rounds',str(20-previous_round if diagnostic else 100000),'--run-id',name,'--replay-policy','all']
            if not diagnostic:args+=['--target-stage-action-steps',str(target)]
            if previous:args+=['--resume-from',previous]
            else:args+=['--transfer-task4-score-from-checkpoint',str(ROOT/self.m['parent']['checkpoint'])]
            attempts.append(dict(run=name,resume_from=previous,started_at=time.time()))
            self.update('diagnostic_running' if diagnostic else 'training_running',active=dict(arm=arm,seed=seed,target_actions=target,run=name))
            self.execute(args,name,self.m['cpus'][0],'candidate','training')
            directory=ROOT/'runs'/name
            attempts[-1]['finished_at']=time.time()
        snap=load_training_snapshot(directory)
        attempts[-1]['committed_round']=snap.round_index
        # Merge only committed complete rounds. Uncommitted failure observations
        # remain in their original retry directory and are never overwritten.
        merged=self.directory/'training_audits'/key
        index=0
        while merged.exists():index+=1;merged=self.directory/'training_audits'/f'{key}_retry{index}'
        merged.mkdir(parents=True)
        for filename,round_key in [('episodes.jsonl','round_index'),('timing.jsonl','round_index'),('opponent_schedule.jsonl','round')]:
            with (merged/filename).open('w') as out:
                committed=0
                for attempt in attempts:
                    end=attempt.get('committed_round',0);source=ROOT/'runs'/attempt['run']/filename
                    if end<=committed:continue
                    for line in source.open():
                        record=json.loads(line)
                        if committed<record[round_key]<=end:out.write(line)
                    committed=end
        audit=audit_training(merged,self.m['agent'],policy=CANDIDATE)
        raw=self.audit(merged,diagnostic)
        payload=torch.load(snap.learner_path,map_location='cpu',weights_only=True)
        actions=int(payload['stage_action_steps']);updates=int(payload['updates']);del payload
        if audit['rounds']!=snap.round_index or (diagnostic and audit['rounds']!=20) or (not diagnostic and actions<target):
            raise EngineeringFailure('Incomplete training segment')
        if updates<=0:raise EngineeringFailure('No training updates')
        checkpoint=snap.learner_path
        return dict(run=str(directory),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                    seed=seed,arm=arm,target_actions=target,actual_actions=actions,rounds=snap.round_index,audit=audit,raw=raw,
                    wall_seconds=sum(a.get('finished_at',time.time())-a['started_at'] for a in attempts),
                    exposure=self.exposure(merged),segments=copy_attempts(attempts),audit_directory=str(merged))

    @staticmethod
    def exposure(directory):
        rows=[json.loads(line) for line in (Path(directory)/'opponent_schedule.jsonl').read_text().splitlines()]
        historical=sum(row['roster']!=['rule']*3 for row in rows)
        return dict(rounds=len(rows),historical_rounds=historical,rule_rounds=len(rows)-historical,
                    historical_fraction=historical/len(rows))

    def point(self,arm,seed,entry,role='candidate'):
        actual=entry['actual_actions'];label=f'{arm}_s{seed}_a{actual}'
        self.update('development',active=dict(arm=arm,seed=seed,actions=actual))
        result=self.evaluate(Path(entry['checkpoint']),arm,label,self.m['development_seeds'],role)
        point=dict(**entry,arm=arm,seed=seed,evaluation=result,summary=result['summary'],failures=failures(result['raw']),role=role)
        point['comparison']=self.compare(result,self.state['parent_development'],label+'_parent')
        point['navigation']=self.navigation(result)
        point['q_drift']=self.q_drift(entry['checkpoint'])
        return point

    def q_drift(self,checkpoint):
        import numpy as np
        import torch
        from agent_code.dqn_agent.model import QNetwork
        rows=[]
        for directory in self.state['engineering']['directories']:
            path=Path(directory)/'q_probes.jsonl'
            if path.exists():rows.extend(json.loads(line) for line in path.read_text().splitlines())
        if not rows:raise ValueError('Missing fixed engineering Q corpus')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=QNetwork(84,6,128);net.load_state_dict(payload['policy']);net.eval()
        with torch.no_grad():values=net(torch.tensor([r['features'] for r in rows],dtype=torch.float32)).numpy()
        previous=np.asarray([r['q'] for r in rows]);masks=np.asarray([r['mask'] for r in rows],dtype=bool)
        if masks.shape!=values.shape:raise ValueError('Invalid Q probe masks')
        return dict(states=len(rows),mean_absolute_q_change=float(np.abs(values-previous).mean()),
                    admitted_action_change_fraction=float((np.where(masks,values,-np.inf).argmax(1)!=np.where(masks,previous,-np.inf).argmax(1)).mean()),
                    label='fixed parent engineering observations; diagnostic only')

    @staticmethod
    def navigation(result):
        counts={}; steps=0
        for d in result['directories']:
            for line in (Path(d)/'timing.jsonl').open():
                row=json.loads(line)
                if row.get('agent_name') != 'double_dqn_continuous_v2_agent':continue
                steps+=1
                action=row.get('action');counts[action]=counts.get(action,0)+1
                nav=row.get('navigation') or row.get('navigation_diagnostic') or {}
                for k,v in nav.items():
                    if isinstance(v,bool):counts[k]=counts.get(k,0)+int(v)
        return dict(steps=steps,counts=counts)

    @staticmethod
    def best(points):
        usable=[p for p in points if not p['failures']]
        return min(usable,key=lambda p:(-p['summary']['mean_score'],-p['summary']['first_place_rate'],p['actual_actions'])) if usable else None

    def seed(self,arm,seed):
        records=self.state['arms'].setdefault(arm,{})
        record=records.setdefault(str(seed),dict(points=[],best=None,complete=False))
        if record['complete']:return record
        cutoff(self.start,kind='selection')
        if arm=='C' and seed in (22,11):
            for entry in self.m['archived_controls'][str(seed)]:
                if any(p['actual_actions']==entry['actual_actions'] for p in record['points']):continue
                record['points'].append(self.point(arm,seed,entry,'archive'))
                record['best']=self.best(record['points'])
                self.update('checkpoint_complete',active=dict(arm=arm,seed=seed,actions=entry['actual_actions']))
        else:
            pending=self.state.setdefault('trained_points',{}).setdefault(arm,{}).setdefault(str(seed),{})
            previous=None
            for target in (20000,40000,60000):
                trained=pending.get(str(target))
                if trained is None:
                    trained=self.train(arm,seed,target,previous=previous)
                    pending[str(target)]=trained
                    self.update('training_segment_complete',active=dict(arm=arm,seed=seed,target_actions=target))
                elif sha256(trained['checkpoint'])!=trained['checkpoint_sha256']:
                    raise ValueError('Training segment checkpoint changed')
                previous=trained['run']
                if any(p['target_actions']==target for p in record['points']):continue
                entry=dict(trained)
                point=self.point(arm,seed,entry)
                record['points'].append(point);record['best']=self.best(record['points'])
                self.update('checkpoint_complete',active=dict(arm=arm,seed=seed,actions=trained['actual_actions']))
        record['complete']=True
        self.update('seed_complete',active=dict(arm=arm,seed=seed,best_actions=record['best']['actual_actions'] if record['best'] else None))
        return record

    @staticmethod
    def snapshot_entry(base,record,arm,seed):
        import torch
        path=base/record['checkpoint']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        return dict(checkpoint=str(path),checkpoint_sha256=sha256(path),actual_actions=int(payload['stage_action_steps']),
                    rounds=next(int(row['round']) for row in csv.DictReader((base.parent.parent/'training.csv').open()) if int(row['stage_action_steps'])==int(payload['stage_action_steps'])),seed=seed,arm=arm,
                    replay=payload['replay'].get('occupancy'),replay_counters=payload['replay'].get('kill_counters'))

    def estimate(self):
        ds=self.state['diagnostics']
        training=sum(ds[a]['wall_seconds']/ds[a]['actual_actions']*60000*2 for a in ('L','LP','LPK'))
        training+=max(d['wall_seconds']/d['actual_actions'] for d in ds.values())*60000*2
        per_world=self.state['engineering']['wall_seconds']/20
        selection=1.25*(training+per_world*200*(1+6+18+6))
        final=1.25*per_world*1200
        available=self.start+20*3600-time.time()
        self.update('throughput_estimate',throughput=dict(estimated_to_selection_seconds=selection,estimated_final_seconds=final,available_until_selection_seconds=available,margin=1.25))
        if selection>available or selection+final>self.start+24*3600-time.time():raise TimeoutError('Measured throughput exceeds unchanged experiment budget')

    def select(self):
        if 'engineering' not in self.state:
            self.state['engineering']=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'engineering',self.m['diagnostic_seeds'],'reference')
            self.update('engineering_complete')
        for arm in ('L','LP','LPK'):
            if arm not in self.state['diagnostics']:
                self.state['diagnostics'][arm]=self.train(arm,self.m['diagnostic_training_seeds'][arm],0,diagnostic=True)
                self.update('diagnostic_complete',active=dict(arm=arm))
        if 'throughput' not in self.state:self.estimate()
        if 'parent_development' not in self.state:
            self.state['parent_development']=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_development',self.m['development_seeds'],'reference')
            self.update('parent_development_complete')
        for seed in (22,11):self.seed('C',seed)
        for arm in ('L','LP','LPK'):
            if arm not in self.state['arms']:cutoff(self.start,kind='arm')
            for seed in (22,11):self.seed(arm,seed)
        self.development_comparisons()
        usable=[a for a in ('L','LP','LPK') if all(self.state['arms'][a][str(s)]['best'] for s in (22,11))]
        if not usable:self.update('screen_failed',error='No configuration has two usable seeds');return
        def rank(a):
            points=[self.state['arms'][a][str(s)]['best']['summary'] for s in (22,11)]
            return (-sum(p['mean_score'] for p in points),-sum(p['first_place_rate'] for p in points),-min(p['mean_score'] for p in points),('L','LP','LPK').index(a))
        winner=min(usable,key=rank);self.update('replication',winning_arm=winner)
        self.seed(winner,33)
        # C33 is the same registered C configuration, not an additional search arm.
        self.seed('C',33)
        cutoff(self.start,kind='selection')
        choices=[r['best'] for r in self.state['arms'][winner].values() if r['best']]
        candidate=min(choices,key=lambda p:(-p['summary']['mean_score'],-p['summary']['first_place_rate'],p['summary']['suicide_rate'],p['summary']['act_p95_seconds'],p['actual_actions'],p['seed']))
        control=self.state['arms']['C'][str(candidate['seed'])]['best']
        if control is None:self.update('screen_failed',error='Selected seed has no usable control');return
        self.update('final_frozen',final_candidate=candidate,matching_control=control)

    def development_comparisons(self):
        results={}
        for child,parent in [('L','C'),('LP','L'),('LPK','LP')]:
            for seed in (22,11):
                a=self.state['arms'][child][str(seed)];b=self.state['arms'][parent][str(seed)]
                for i,(x,y) in enumerate(zip(a['points'],b['points'])):
                    key=f'{child}_minus_{parent}_s{seed}_target{(i+1)*20000}'
                    results[key]=self.compare(x['evaluation'],y['evaluation'],key)
                if a['best'] and b['best']:
                    key=f'{child}_minus_{parent}_s{seed}_best'
                    results[key]=self.compare(a['best']['evaluation'],b['best']['evaluation'],key)
        self.update('three_step_comparison',development_comparisons=results)

    def run(self):
        try:
            if 'final_candidate' not in self.state:self.select()
            if 'final_candidate' not in self.state:return
            candidate=self.state['final_candidate'];control=self.state['matching_control']
            for p in (candidate,control):
                if sha256(p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Frozen endpoint changed')
            parent=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_final',self.m['final_seeds'],'reference')
            child=self.evaluate(Path(candidate['checkpoint']),candidate['arm'],'candidate_final',self.m['final_seeds'])
            matched=self.evaluate(Path(control['checkpoint']),'C','control_final',self.m['final_seeds'],control['role'])
            comparison=self.compare(child,parent,'final_parent');comparison_control=self.compare(child,matched,'final_control')
            delta=child['summary']['mean_score']-parent['summary']['mean_score']
            delta_control=child['summary']['mean_score']-matched['summary']['mean_score']
            bad=failures(parent['raw'])+failures(child['raw'])+failures(matched['raw'])
            status='final_failed' if bad else 'observed_improvement' if min(delta,delta_control)>0 else 'no_score_improvement'
            self.update(status,final_parent=parent,final_child=child,final_control=matched,
                        comparison=comparison,comparison_control=comparison_control,score_delta=delta,control_score_delta=delta_control)
        except TimeoutError as exc:self.update('budget_incomplete',error=str(exc))
        except (EngineeringFailure,ValueError) as exc:self.update('engineering_failure',error=str(exc))
        except BaseException as exc:
            self.update('infrastructure_interrupted',error=repr(exc));raise
        finally:
            self.report()
            if self.state['status'] in TERMINAL:self.archive_terminal()

    def archive_terminal(self):
        # Commit only this isolated experiment's terminal records, after workers exit.
        import shutil
        destination=ROOT/'experiments/results'/self.m['campaign_id']
        try:
            self.check_identity()
            destination.mkdir(parents=True,exist_ok=True)
            for source in (self.state_path,self.directory/'report.md',self.path,self.directory/'test_results.json',self.directory/'preflight/result.json'):
                shutil.copy2(source,destination/source.name)
            subprocess.run(['git','add','-f',str(destination.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m','Record Task4 three-step score experiment outcome'],cwd=ROOT,check=True)
        except (OSError,ValueError,subprocess.CalledProcessError) as exc:
            write(self.directory/'archive_error.json',{'error':str(exc)})

    def report(self):
        names=dict(observed_improvement='观察到本轮得分改进',no_score_improvement='未观察到改进',screen_failed='未观察到改进（无可用配对候选）',final_failed='验证失败',engineering_failure='验证失败',budget_incomplete='预算内未完成')
        lines=['# Task4 三步得分优化报告','',f"结论：{names.get(self.state['status'], self.state['status'])}",f'源码：{self.commit}',f'父权重 SHA-256：{PARENT_SHA256}',
            '三步分别为学习率、金币势函数、击杀片段保留与重采样；不承诺单调进化，不代表旧课程 Task4 合格。',
            '旧 C 快照仅作为新协议探索参照，旧固定终点失败结论不变。所有新训练不继承父 Replay。',
            '击杀标签只来自真实回调；死亡后的炸弹得分可能没有对应学习回调，不能声称训练回报等于最终得分。','',
            '| 配置 | 种子 | 实际动作 | 得分 | 金币 | 击杀 | 独占/并列第一 | 自杀率 | 炸弹存活 | P95/最大ms | 失败项 |',
            '|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|']
        lines += ['',f"测试证据：{self.state.get('tests',{})}；真实恢复对照：{self.state.get('preflight',{})}"]
        for arm,seeds in self.state['arms'].items():
            for seed,record in seeds.items():
                for p in record['points']:
                    s=p['summary']
                    lines.append(f"| {arm} | {seed} | {p['actual_actions']} | {s['mean_score']:.4f} | {s['mean_coins']:.4f} | {s['mean_kills']:.4f} | {s['exclusive_first_rate']:.4f}/{s['tied_first_rate']:.4f} | {s['suicide_rate']:.4f} | {s['bomb_survival_rate']:.4f} | {1000*s['act_p95_seconds']:.2f}/{1000*s['act_max_seconds']:.2f} | {p['failures']} |")
                    lines+=['',f"权重：{p['checkpoint']}；SHA-256：{p['checkpoint_sha256']}。"]
        for key,result in {**self.state.get('development_comparisons',{}),**{k:self.state[k] for k in ('comparison','comparison_control') if k in self.state}}.items():
            score=next(r for r in result['metrics'] if r['metric']=='score');first=result['first_place_difference']
            low,high=score['bootstrap_ci95_low'],score['bootstrap_ci95_high']
            lines+=['',f"{key}：得分差 {score['mean_difference']:.4f}，95%区间 [{low:.4f}, {high:.4f}]；第一名差 {first['mean']:.4f}，区间 {first['ci95']}。"]
            if low<=0<=high:lines.append('区间包含零，证据仍不确定。')
            if first['mean']<0:lines.append('第一名比例下降，保留此取舍。')
        if self.state.get('error'):lines+=['',self.state['error']]
        lines+=['','状态文件保存诊断、三种子实际范围、导航统计、Replay占用、最终候选和完整身份；日志目录保存逐批复现命令。',
                f'复现：`{sys.executable} experiments/task4_score_campaign.py --manifest {self.path}`。']
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    import fcntl
    manifest=json.loads(Path(args.manifest).read_text())
    directory=ROOT/'runs'/manifest['campaign_id'];directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Campaign(args.manifest,args.resume)
        (directory/'controller.pid').write_text(str(os.getpid())+'\n')
        def interrupted(signum,frame):raise OSError('Controller interrupted by signal '+str(signum))
        signal.signal(signal.SIGTERM,interrupted)
        controller.run()

if __name__=='__main__':main()

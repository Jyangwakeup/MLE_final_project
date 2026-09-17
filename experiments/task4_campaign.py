"""Bounded shared-parent Task 4 training and independent frozen validation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv
import json
import math
import numpy as np
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import task3_plateau_stopping as p
from experiments.task3_retention_prefix import summarize_evaluation, _evaluation_directories
from experiments.compare_evaluations import compare_evaluations
from experiments.task4_transfer import digest, PARENT_SHA256

VERSION='task4-campaign-v1'
ENGINEERING=('act_timeouts','act_skipped','avoidable_escape_collapses',
             'robust_guarantee_losses','robust_search_timeouts')
TERMINAL={'passed','stopped_engineering_failure','stopped_development_failure',
          'stopped_confirmation_failure','stopped_main_validation_failure','budget_exhausted'}


class EngineeringFailure(RuntimeError):
    pass


def validate_manifest(m):
    if m['schema_version']!=VERSION or m['training_seeds']!=[11,22,33]:
        raise ValueError('Task 4 protocol/training seeds mismatch')
    if set(m['arm_configs'])!={'A','B'} or m['parent']['checkpoint_sha256']!=PARENT_SHA256:
        raise ValueError('Task 4 parent/arms mismatch')
    groups=[m[k+'_seeds'] for k in ('diagnostic','development','confirmation','main_validation')]
    if list(map(len,groups))!=[20,60,100,100] or len(set(sum(groups,[])))!=280:
        raise ValueError('Task 4 requires disjoint 20/60/100/100 worlds')
    forbidden=set(m['excluded_seeds'])|set(range(20000,20100))|set(m['training_seeds'])|set(m['diagnostic_training_seeds'].values())
    forbidden.update(v[k] for v in m['rng_plan'].values() for k in ('environment_seed','official_opponent_seed'))
    if set(sum(groups,[])) & forbidden:
        raise ValueError('Task 4 validation worlds were used/reserved')
    if not 1<=len(m['cpus'])<=6 or len(set(m['cpus']))!=len(m['cpus']):
        raise ValueError('Task 4 requires 1..6 separate cores')
    return m


def evaluate_gates(parent,child):
    checks={}
    def check(name,value,threshold,op):
        checks[name]=dict(actual=value,threshold=threshold,operator=op,
            passed=(math.isclose(value,threshold,rel_tol=1e-12,abs_tol=1e-12)
                    or (value>threshold if op=='>=' else value<threshold)))
    for task,metric in ((1,'mean_score'),(2,'mean_coins'),(2,'mean_crates'),
                        (3,'mean_score'),(3,'mean_coins'),(3,'mean_crates')):
        a,b=parent[f'task{task}'][metric],child[f'task{task}'][metric]
        check(f'task{task}_{metric}_retention',b/a if a else (1. if b>=0 else 0.),.9,'>=')
    a,b=parent['task4'],child['task4']
    check('task4_score_gain',b['mean_score']-a['mean_score'],.5,'>=')
    kill=b['mean_kills']-a['mean_kills'];first=b['first_place_rate']-a['first_place_rate']
    checks['task4_combat_gain']=dict(kill_gain=kill,first_place_gain=first,
        kill_threshold=.1,first_place_threshold=.05,operator='OR',
        passed=kill>=.1-1e-12 or first>=.05-1e-12)
    for task,s in child.items():
        for metric in ENGINEERING:
            check(task+'_'+metric,s[metric],0.,'<=')
        check(task+'_act_p95',s['act_p95_seconds'],.25,'<=')
        check(task+'_act_max',s['act_max_seconds'],.48,'<=')
        check(task+'_invalid_action_rate',s['invalid_action_rate'],.01,'<=')
        if task=='task1':
            check('task1_bombs',s['mean_bombs'],0.,'<=')
        else:
            check(task+'_suicide_rate',s['suicide_rate'],.05,'<=')
            check(task+'_bomb_survival_rate',s['bomb_survival_rate'],.95,'>=')
        if task in ('task3','task4'):
            check(task+'_zero_bomb_round_rate',s['zero_bomb_round_rate'],.1,'<=')
    return all(c['passed'] for c in checks.values()),checks


def engineering_checks(summaries):
    return [f'{task}_{key}' for task,s in summaries.items()
            for key,bad in [(k,s[k]!=0) for k in ENGINEERING]+[
                ('act_p95',s['act_p95_seconds']>.25),('act_max',s['act_max_seconds']>.48),
                ('task1_bomb',task=='task1' and s['mean_bombs']!=0)] if bad]


def diagnostic_key(point):
    s=point['child']['summaries']['task4']
    return (sum(not c['passed'] for c in point['gate_checks'].values()),
            -s['mean_score'],-s['first_place_rate'],-s['mean_kills'],point['rounds'])


def b_eligible(results):
    failed=[r for r in results.values() if r.get('selected') is None]
    allowed={'task4_score_gain','task4_combat_gain'}
    return bool(failed) and all(any(
        {k for k,c in h['gate_checks'].items() if not c['passed']} <= allowed
        for h in r['history']) for r in failed)


def candidate_key(point):
    s=point['child']['summaries']['task4']
    retention=min(c['actual'] for k,c in point['gate_checks'].items() if k.endswith('_retention'))
    return (-s['mean_score'],s['suicide_rate'],-s['first_place_rate'],-s['mean_kills'],
            -retention,max(t['act_p95_seconds'] for t in point['child']['summaries'].values()),point['seed'])


def audit_training(directory,agent,*,check_timing=True):
    episodes=[json.loads(l) for l in (directory/'episodes.jsonl').read_text().splitlines()]
    deaths={e['round_index']:next(a for a in e['agents'] if a['name']==agent)
            for e in episodes if next(a for a in e['agents'] if a['name']==agent)['dead']}
    traces={str(k):[] for k in deaths}
    for line in (directory/'timing.jsonl').open():
        row=json.loads(line)
        if row['agent_name']==agent and row['round_index'] in deaths:
            traces[str(row['round_index'])].append(row)
    times=[]
    for line in (directory/'timing.jsonl').open():
        row=json.loads(line)
        if row['agent_name']==agent and row.get('think_time') is not None:times.append(row['think_time'])
    timing=dict(act_p95_seconds=float(np.percentile(times,95)),act_max_seconds=max(times)) if times else {}
    unresolved=[];classified={}
    for number,own in deaths.items():
        rows=traces[str(number)];last=rows[-1] if rows else {}
        placements=[r for r in rows if r['action']=='BOMB']
        explained=(placements and placements[-1].get('safety',{}).get('v1_to_physical_fallback'))
        category=('self_bomb_after_no_safe_placement' if explained else 'unexplained_self_death') if own['suicides'] else 'opponent_bomb'
        if own['suicides'] and not explained:unresolved.append(number)
        classified[str(number)]=dict(category=category,death_step=own['death_step'],causes=own['death_causes'],
            own_bomb_pending=last.get('safety',{}).get('own_bomb_pending'),
            fallback=last.get('safety',{}).get('v1_to_physical_fallback'))
    result=dict(rounds=len(episodes),deaths=classified,death_traces=traces,unresolved=unresolved,timing=timing)
    p._write_json(directory/'task4_training_audit.json',result)
    if check_timing and (not timing or timing['act_p95_seconds']>.25 or timing['act_max_seconds']>.48):
        raise EngineeringFailure(f'Training complete act timing gate: {directory} {timing}')
    if unresolved:raise EngineeringFailure(f'Unexplained training self-death: {directory} rounds {unresolved}')
    return dict(rounds=len(episodes),deaths=classified,timing=timing,audit=str(directory/'task4_training_audit.json'))


class Campaign:
    def __init__(self,root,manifest,resume=False):
        self.root=Path(root);self.manifest_path=Path(manifest).resolve();self.m=validate_manifest(p._json(manifest));self.commit=p._git(root,'rev-parse','HEAD')
        self.identity=dict(source_commit=self.commit,manifest_sha256=digest(self.m),
            config_sha256={a:p._sha256(self.root/c) for a,c in self.m['arm_configs'].items()},
            parent_sha256=self.m['parent']['checkpoint_sha256'],opponent_hashes=self.m['opponent_hashes'])
        self.started=datetime.fromisoformat(self.m['started_at'].replace('Z','+00:00')).timestamp()
        self.deadline=self.started+86400
        self.directory=self.root/'runs'/f"{self.m['campaign_id']}_{self.commit[:7]}"
        self.path=self.directory/'result.json'
        if self.path.exists():
            if not resume:raise FileExistsError('Use --resume for same-identity infrastructure recovery')
            self.state=self.read(self.path)
        else:self.state=dict(status='created',task4_qualified=False,arms={},diagnostics={})
        self.stop=threading.Event();self.check_identity()
        if self.m.get('admission_prerequisites'):
            from experiments.viability_prerequisites import validate
            validate(self.root,self.m)

    def write(self,path,value):
        value.update(self.identity);p._write_json(path,value)

    def read(self,path):
        value=p._json(path)
        if any(value.get(k)!=v for k,v in self.identity.items()):raise ValueError('Changed Task 4 cache/resume identity')
        return value

    def check_identity(self):
        if p._git(self.root,'rev-parse','HEAD')!=self.commit or p._git(self.root,'status','--porcelain'):
            raise ValueError('Task 4 requires clean unchanged committed source')
        for a,c in self.m['arm_configs'].items():
            if p._sha256(self.root/c)!=self.identity['config_sha256'][a]:raise ValueError('Task 4 config changed')
        if digest(p._json(self.manifest_path)) != self.identity['manifest_sha256']:
            raise ValueError('Task 4 manifest changed')
        if p._sha256(self.root/self.m['parent']['checkpoint'])!=self.identity['parent_sha256']:
            raise ValueError('Task 4 parent weight changed')
        for path,sha in self.m['opponent_hashes'].items():
            if p._sha256(self.root/path)!=sha:raise ValueError('Task 4 opponent changed')

    def check_time(self,evaluation=False,new_arm=False):
        hours=18 if new_arm else 23 if evaluation else 24
        if time.time()>=self.started+hours*3600:raise TimeoutError('Task 4 campaign cutoff reached')

    def execute(self,args,name,cpu):
        self.check_time();self.check_identity()
        log=self.directory/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        command=['taskset','-c',str(cpu),sys.executable,'experiments/task4_worker.py',*args]
        with log.open('w') as output:
            proc=subprocess.Popen(command,cwd=self.root,env={**os.environ,**p.THREAD_ENVIRONMENT},
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while proc.poll() is None:
                    if self.stop.is_set():raise EngineeringFailure('Cancelled after another worker failed')
                    if time.time()>=self.deadline:raise TimeoutError('Task 4 hard deadline')
                    time.sleep(.5)
                if proc.returncode:
                    runroot=self.root/'runs'/name
                    failures=list(runroot.rglob('task4_safety_failure.json'))
                    if failures:
                        raise EngineeringFailure(f'Safety failure: {failures[0]}')
                    if proc.returncode<0:raise OSError(f'Worker interrupted: {log}')
                    raise EngineeringFailure(f'Worker exception; inspect {log}')
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise

    def evaluate(self,arm,checkpoint,label,phase,worlds,tasks=(1,2,3,4),opponents=None):
        self.check_time(evaluation=True);self.check_identity()
        expected=dict(arm=arm,checkpoint_sha256=p._sha256(checkpoint),worlds=list(worlds),tasks=list(tasks),opponents=opponents)
        cache=self.directory/'evaluations'/(label+'.json')
        if cache.exists():
            value=self.read(cache)
            if value['request']!=expected:raise ValueError('Task 4 evaluation cache request mismatch')
            return value
        prefixes={};summaries={}
        # Fixed disjoint CPU waves; abort a wave on any engineering failure.
        for offset in range(0,len(tasks),len(self.m['cpus'])):
            batch=tasks[offset:offset+len(self.m['cpus'])]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures=[]
                for i,task in enumerate(batch):
                    base=self.root/'runs'/f"{self.m['campaign_id']}_{arm}_{self.identity['manifest_sha256'][:8]}_{label}_t{task}_{self.commit[:7]}"
                    target,_=p._unique_target(base);prefixes[str(task)]=target.name
                    args=['--config',self.m['arm_configs'][arm],'--mode','evaluate','--device','cpu',
                          '--task',str(task),'--agent',self.m['agent'],'--seeds',*map(str,worlds),
                          '--n-rounds','1','--checkpoint',str(checkpoint),'--run-id',target.name,'--replay-policy','all']
                    if opponents:args.extend(['--opponents',*opponents])
                    futures.append(pool.submit(self.execute,args,target.name,self.m['cpus'][i]))
                try:
                    for f in as_completed(futures):f.result()
                except BaseException:
                    self.stop.set();raise
            for task in batch:
                directory=self.root/'runs'/prefixes[str(task)]
                summary=summarize_evaluation(directory,self.m['agent'],worlds)
                with (directory/(directory.name+'_summary')/'summary.csv').open() as handle:
                    average=next(r for r in csv.DictReader(handle) if r['agent_name']==self.m['agent'] and r['run_id']=='AVERAGE')
                summary['mean_bombs']=float(average['mean_bombs'])
                summaries[f'task{task}']=summary
            bad=engineering_checks(summaries)
            if bad:raise EngineeringFailure('Frozen engineering gates: '+str(bad))
        result=dict(request=expected,summaries=summaries,runs=prefixes,phase=phase)
        self.write(cache,result)
        return result

    def train(self,arm,seed,rounds,previous=None,diagnostic=False):
        self.check_time();self.check_identity()
        count=20 if diagnostic else 50
        base=self.root/'runs'/f"{self.m['campaign_id']}_{arm}_{self.identity['manifest_sha256'][:8]}_{'diag' if diagnostic else 'train'}_s{seed}_c{rounds:04d}_{self.commit[:7]}"
        target,_=p._unique_target(base)
        args=['--config',self.m['arm_configs'][arm],'--mode','train','--device','cpu','--task','4',
              '--agent',self.m['agent'],'--seed',str(seed),'--n-rounds',str(count),
              '--run-id',target.name,'--replay-policy','all']
        if previous:args.extend(['--resume-from',str(previous)])
        else:args.extend(['--transfer-task4-from-checkpoint',str(self.root/self.m['parent']['checkpoint'])])
        self.execute(args,target.name,self.m['cpus'][0])
        audit=audit_training(target,self.m['agent'])
        from experiments.resume import load_training_snapshot, _load_generation
        snap=load_training_snapshot(target)
        latest=p._json(target/'resume/latest.json')
        if len(latest['generations'])!=2:raise EngineeringFailure('Missing two resume generations')
        for generation in latest['generations']:
            _load_generation(target,generation,expected_schema='training-resume-v11')
        if audit['rounds']!=count or snap.round_index!=rounds:raise EngineeringFailure('Training round mismatch')
        with (target/'training.csv').open() as f:rows=list(csv.DictReader(f))
        if not rows or any(r['loss'] and not math.isfinite(float(r['loss'])) for r in rows) or int(rows[-1]['updates'])<=0:
            raise EngineeringFailure('Invalid optimizer evidence')
        return dict(run=str(target),checkpoint=str(target/'checkpoints/final.pt'),
                    checkpoint_sha256=p._sha256(target/'checkpoints/final.pt'),audit=audit)

    def assess(self,arm,seed,trained,phase,parent,worlds):
        if p._sha256(Path(trained['checkpoint'])) != trained['checkpoint_sha256']:
            raise ValueError('Frozen Task 4 child weight changed')
        child=self.evaluate(arm,Path(trained['checkpoint']),f'{phase}_{arm}_s{seed}_c{trained["rounds"]}',phase,worlds)
        passed,checks=evaluate_gates(parent['summaries'],child['summaries'])
        comparisons={}
        for task in (1,2,3,4):
            comparisons[f'task{task}']=compare_evaluations(
                _evaluation_directories(self.root/'runs',child['runs'][str(task)],worlds),
                _evaluation_directories(self.root/'runs',parent['runs'][str(task)],worlds),
                self.directory/'comparisons'/f'{phase}_{arm}_{seed}_{trained["rounds"]}'/f'task{task}',bootstrap_samples=10000)
        return {**trained,'seed':seed,'arm':arm,'eligible':passed,'gate_checks':checks,
                'child':child,'bootstrap':comparisons}

    def seed(self,arm,seed,parent):
        path=self.directory/arm/f'seed_{seed}.json'
        result=self.read(path) if path.exists() else dict(history=[],selected=None)
        if result['selected'] is not None:return result
        for rounds in range(50,301,50):
            if any(h['rounds']==rounds for h in result['history']):continue
            previous=Path(result['history'][-1]['run']) if result['history'] else None
            if result['history'] and p._sha256(Path(result['history'][-1]['checkpoint'])) != result['history'][-1]['checkpoint_sha256']:
                raise ValueError('Task 4 resume parent checkpoint changed')
            training_record=self.directory/arm/f'trained_s{seed}_c{rounds}.json'
            if training_record.exists():
                trained=self.read(training_record)
            else:
                trained=self.train(arm,seed,rounds,previous);trained['rounds']=rounds
                self.write(training_record,trained)
            point=self.assess(arm,seed,trained,'development',parent,self.m['development_seeds'])
            result['history'].append(point)
            if point['eligible']:result['selected']=point
            self.write(path,result)
            print(f'{arm} seed={seed} rounds={rounds} eligible={point["eligible"]}',flush=True)
            if result['selected'] is not None:break
        result['diagnostic']=min(result['history'],key=diagnostic_key)
        self.write(path,result);return result

    def validation(self,arm,results):
        parent=self.evaluate('A',self.root/self.m['parent']['checkpoint'],'confirmation_parent','confirmation',self.m['confirmation_seeds'])
        confirmation={}
        for seed in (11,22,33):
            confirmation[str(seed)]=self.assess(arm,seed,results[str(seed)]['selected'],'confirmation',parent,self.m['confirmation_seeds'])
            self.state['confirmation']=confirmation
            self.write(self.path,self.state)
            if not confirmation[str(seed)]['eligible']:
                self.state['status']='stopped_confirmation_failure';return
        candidate=min(confirmation.values(),key=candidate_key)
        self.state['selected_training_seed']=candidate['seed'];self.write(self.path,self.state)
        parent=self.evaluate('A',self.root/self.m['parent']['checkpoint'],'main_parent','main_validation',self.m['main_validation_seeds'])
        main=self.assess(arm,candidate['seed'],results[str(candidate['seed'])]['selected'],
                         'main_validation',parent,self.m['main_validation_seeds'])
        self.state['main_validation']=main
        self.state.update(status='passed' if main['eligible'] else 'stopped_main_validation_failure',task4_qualified=main['eligible'])

    def run(self):
        if self.state['status'] in TERMINAL:return self.state
        try:
            self.state['status']='diagnostics';self.write(self.path,self.state)
            for seed in (22,11,33):
                if str(seed) not in self.state['diagnostics']:
                    actual=self.m['diagnostic_training_seeds'][str(seed)]
                    self.state['diagnostics'][str(seed)]=self.train('A',actual,20,diagnostic=True)
                    self.write(self.path,self.state)
            self.state['mixed_baseline']=self.evaluate('A',self.root/self.m['parent']['checkpoint'],'mixed_parent','diagnostic',self.m['diagnostic_seeds'],tasks=(4,),opponents=['rule_based_agent','peaceful_agent','coin_collector_agent'])
            parent=self.evaluate('A',self.root/self.m['parent']['checkpoint'],'development_parent','development',self.m['development_seeds'])
            self.state['parent_baseline']=parent;self.write(self.path,self.state)
            for arm in ('A','B'):
                if arm=='B' and not b_eligible(self.state['arms']['A']):break
                if arm not in self.state['arms']:
                    self.check_time(new_arm=True);self.state['arms'][arm]={};self.write(self.path,self.state)
                self.state['status']='development_'+arm;results=self.state['arms'][arm]
                results['22']=self.seed(arm,22,parent);self.write(self.path,self.state)
                if results['22']['selected'] is None:continue
                for seed in (11,33):
                    results[str(seed)]=self.seed(arm,seed,parent);self.write(self.path,self.state)
                if all(r['selected'] is not None for r in results.values()):
                    self.state['status']='confirmation';self.write(self.path,self.state)
                    self.validation(arm,results);return self.state
            self.state['status']='stopped_development_failure'
        except (TimeoutError,subprocess.TimeoutExpired) as exc:
            self.state.update(status='budget_exhausted',error=str(exc))
        except EngineeringFailure as exc:
            self.state.update(status='stopped_engineering_failure',error=str(exc))
        except Exception as exc:
            self.state.update(status='infrastructure_error',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            self.state['finished_at']=datetime.now().astimezone().isoformat();self.write(self.path,self.state)
        return self.state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('experiments/task4_campaign.json'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    v=Campaign(Path(__file__).resolve().parents[1],args.manifest,args.resume)
    result=v.run();print(result['status'],v.path,flush=True)
    return 0 if result['status']=='passed' else 2

if __name__=='__main__':raise SystemExit(main())

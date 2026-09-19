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
from experiments.task4_frozen_transfer import VERSION
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
    from experiments.frozen_opponents import VERSION as schedule
    if m['schema_version'] != VERSION or set(m['arm_configs']) != {'C','S'}:
        raise ValueError('Frozen protocol/arms mismatch')
    if m['training_seeds'] != [22,11,33] or m['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Frozen seeds/parent mismatch')
    groups=[m[k+'_seeds'] for k in ('diagnostic','development','heldout','final')]
    if list(map(len,groups)) != [20,200,100,400] or len(set(sum(groups,[]))) != 720:
        raise ValueError('Frozen data partition mismatch')
    forbidden=set(m['excluded_seeds']) | set(range(20000,20100))
    for values in m['rng_plan'].values():forbidden.update(values.values())
    if set(sum(groups,[])) & forbidden:raise ValueError('Reserved world overlap')
    if m['final_seeds'] != list(range(m['final_seeds'][0],m['final_seeds'][0]+400)):
        raise ValueError('Final worlds must be contiguous')
    if not 1 <= len(m['cpus']) <= 6 or len(set(m['cpus'])) != len(m['cpus']):
        raise ValueError('Invalid physical cores')
    if m['schedule_version']!=schedule or m['pool_sha256']!=digest(m['models']):
        raise ValueError('Frozen pool/schedule mismatch')
    if set(m['pool'])!={'parent','e1_22_40k','e1_11_40k','e1_33_20k'} or set(m['pool'])&set(m['heldout_ids']):
        raise ValueError('Training and heldout opponent identities overlap')
    if m['limits'] != dict(p95_seconds=.1,max_seconds=.3,search_budget_ms=400):
        raise ValueError('Unregistered safety limits')
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
            config_hashes={p:sha256(ROOT/p) for p in [*self.m['arm_configs'].values(),self.m['reference_config']]},
            parent_sha256=PARENT_SHA256,pool_sha256=self.m['pool_sha256'],schedule_version=self.m['schedule_version'])
        self.cancel=threading.Event()
        self.resuming=resume
        if self.state_path.exists():
            if not resume:raise ValueError('Existing campaign requires explicit infrastructure resume')
            self.state=json.loads(self.state_path.read_text())
            if self.state['identity']!=self.identity or self.state['status'] in TERMINAL:
                raise ValueError('Cannot resume terminal or different experiment')
        else:self.state=dict(status='created',identity=self.identity,arms={},diagnostics={},task4_qualified=False)
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
        for entry in self.m['models'].values():
            if sha256(entry['checkpoint'])!=entry['sha256']:raise ValueError('Frozen weight changed')

    def update(self, status, **values):
        self.state.update(status=status,updated_at=datetime.now().astimezone().isoformat(),**values)
        write(self.state_path,self.state)
        print(json.dumps(dict(status=status,**values)),flush=True)

    def execute(self, args, name, cpu, role, kind):
        cutoff(self.start,kind='work');self.check_identity()
        log=self.directory/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        command=['taskset','-c',str(cpu),sys.executable,'experiments/task4_frozen_worker.py',
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
        config=self.m['reference_config'] if role=='reference' else self.m['arm_configs'][arm]
        request=dict(identity=self.identity,role=role,kind=kind,arm=arm,checkpoint_sha256=sha256(checkpoint),worlds=worlds,config=config,safety=self.m['target_safety'])
        cache=self.directory/'evaluations'/(label+'.json')
        if cache.exists():
            value=json.loads(cache.read_text())
            if value['request']!=request:raise ValueError('Evaluation cache identity mismatch')
            if any(sha256(p)!=h for p,h in value['artifact_hashes'].items()):raise ValueError('Evaluation cache artifact changed')
            if role=='candidate' and failures(value['raw']):
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
                    artifact_hashes={str(Path(d)/f):sha256(Path(d)/f) for d in directories for f in ('metadata.json','episodes.jsonl','timing.jsonl','opponent_schedule.jsonl')})
        write(cache,result)
        if role=='candidate' and failures(raw):
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
        from experiments.task4_frozen_transfer import validate_resume
        import torch
        key=f'{arm}_{seed}_{"diag" if diagnostic else target}'
        attempts=self.state.setdefault('training_attempts',{}).setdefault(key,[])
        complete=False;previous_round=0
        if attempts:
            if not self.resuming:raise ValueError('An existing training attempt requires infrastructure resume')
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
            else:args+=['--transfer-task4-frozen-opponents-from-checkpoint',str(ROOT/self.m['parent']['checkpoint'])]
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

    def seed(self,arm,seed):
        records=self.state['arms'].setdefault(arm,{})
        if str(seed) in records:return records[str(seed)]
        cutoff(self.start,kind='selection')
        self.update('training',active=dict(arm=arm,seed=seed,target_actions=60000))
        pending=self.state.setdefault('trained_endpoints',{}).setdefault(arm,{})
        trained=pending.get(str(seed))
        if trained is None:
            trained=self.train(arm,seed,60000);pending[str(seed)]=trained
        elif sha256(trained['checkpoint'])!=trained['checkpoint_sha256']:
            raise ValueError('Saved endpoint changed during infrastructure recovery')
        self.update('development',active=dict(arm=arm,seed=seed,target_actions=60000))
        rules=self.evaluate(Path(trained['checkpoint']),arm,f'{arm}_s{seed}_rules',self.m['development_seeds'])
        heldout=self.evaluate(Path(trained['checkpoint']),arm,f'{arm}_s{seed}_heldout',self.m['heldout_seeds'],kind='heldout')
        comparison=self.compare(rules,self.state['parent_development'],f'{arm}_s{seed}_parent')
        historical_comparison=self.compare(heldout,self.state['parent_heldout'],f'{arm}_s{seed}_heldout_parent')
        bad=failures(rules['raw'])+['heldout:'+f for f in failures(heldout['raw'])]
        point=dict(**trained,comparison=comparison,historical_comparison=historical_comparison,
                   summary=rules['summary'],failures=bad,evaluation=rules,heldout=heldout)
        if arm=='S':point['control_comparison']=self.compare(rules,self.state['arms']['C'][str(seed)]['evaluation'],f'S_s{seed}_control')
        records[str(seed)]=point
        self.update('seed_complete',active=dict(arm=arm,seed=seed,summary=point['summary'],failures=bad))
        if bad:raise EngineeringFailure('Fixed endpoint unusable: '+arm+str(seed)+': '+str(bad))
        return point

    def estimate(self):
        diagnostics=self.state['diagnostics']
        train_seconds=sum(diagnostics[a]['wall_seconds']/diagnostics[a]['actual_actions']*60000*3 for a in ('C','S'))
        # Four neural seats are the expensive case. Use their measured parallel
        # full-game throughput for every future evaluation, with a 25% margin.
        per_world=self.state['engineering']['wall_seconds']/20
        selection=1.25*(train_seconds+per_world*(300+6*300))
        final=1.25*per_world*3*400
        selection_available=self.start+20*3600-time.time()
        total_available=self.start+24*3600-time.time()
        self.update('throughput_estimate',throughput=dict(estimated_to_selection_seconds=selection,
                    estimated_final_seconds=final,available_until_selection_seconds=selection_available,
                    available_until_hard_stop_seconds=total_available,margin=1.25,training_seconds=train_seconds))
        if selection>selection_available or selection+final>total_available:
            raise TimeoutError('Measured throughput cannot complete unchanged experiment in budget')

    def select(self):
        if 'engineering' not in self.state:
            self.update('engineering')
            self.state['engineering']=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'engineering',self.m['diagnostic_seeds'],'reference','engineering')
            if failures(self.state['engineering']['raw']):raise EngineeringFailure('Engineering learner safety gates failed')
            self.update('engineering_complete')
        for arm in ('C','S'):
            if arm not in self.state['diagnostics']:
                cutoff(self.start,kind='arm')
                self.update('diagnostic',active=dict(arm=arm))
                self.state['diagnostics'][arm]=self.train(arm,self.m['diagnostic_training_seeds'][arm],0,diagnostic=True)
                self.update('diagnostic_complete',active=dict(arm=arm))
        if 'throughput' not in self.state:self.estimate()
        if 'parent_development' not in self.state:
            self.update('parent_development')
            self.state['parent_development']=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_development',self.m['development_seeds'],'reference')
            self.update('parent_development_complete')
        if 'parent_heldout' not in self.state:
            self.update('parent_heldout')
            self.state['parent_heldout']=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_heldout',self.m['heldout_seeds'],'reference','heldout')
            self.update('parent_heldout_complete')
        for seed in (22,11):
            for arm in ('C','S'):
                if arm not in self.state['arms']:cutoff(self.start,kind='arm')
                self.seed(arm,seed)
        controls=[self.state['arms']['C'][str(seed)]['summary']['mean_score'] for seed in (22,11)]
        historical=[self.state['arms']['S'][str(seed)]['summary']['mean_score'] for seed in (22,11)]
        parent=self.state['parent_development']['summary']['mean_score']
        if not (all(score>parent for score in historical) and sum(historical)>sum(controls)):
            self.update('screen_failed',screen=dict(C=controls,S=historical,parent=parent));return
        self.update('replication')
        self.seed('C',33);self.seed('S',33)
        cutoff(self.start,kind='selection')
        candidate=min(self.state['arms']['S'].values(),key=lambda p:(-p['summary']['mean_score'],
            -p['summary']['first_place_rate'],p['summary']['suicide_rate'],p['summary']['act_p95_seconds'],p['seed']))
        self.update('final_frozen',final_candidate=candidate,
                    matching_control=self.state['arms']['C'][str(candidate['seed'])])

    def run(self):
        try:
            if 'final_candidate' not in self.state:self.select()
            if 'final_candidate' not in self.state:return
            candidate=self.state['final_candidate'];control=self.state['matching_control']
            for p in (candidate,control):
                if sha256(p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Frozen endpoint changed')
            parent=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_final',self.m['final_seeds'],'reference')
            child=self.evaluate(Path(candidate['checkpoint']),'S','candidate_final',self.m['final_seeds'])
            matched=self.evaluate(Path(control['checkpoint']),'C','control_final',self.m['final_seeds'])
            comparison=self.compare(child,parent,'final_parent');comparison_control=self.compare(child,matched,'final_control')
            delta=child['summary']['mean_score']-parent['summary']['mean_score']
            delta_control=child['summary']['mean_score']-matched['summary']['mean_score']
            bad=failures(child['raw'])+failures(matched['raw'])
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
            for source in (self.state_path,self.directory/'report.md',self.path):
                shutil.copy2(source,destination/source.name)
            subprocess.run(['git','add','-f',str(destination.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m','Record frozen historical opponent experiment outcome'],cwd=ROOT,check=True)
        except (OSError,ValueError,subprocess.CalledProcessError) as exc:
            write(self.directory/'archive_error.json',{'error':str(exc)})

    def report(self):
        names=dict(observed_improvement='观察到历史对手训练收益',no_score_improvement='未观察到收益',
                   screen_failed='未观察到收益（初筛未晋级）',final_failed='验证失败',
                   engineering_failure='验证失败',budget_incomplete='预算内未完成')
        lines=['# Task4 冻结历史对手对照报告','',f"结论：{names.get(self.state['status'],self.state['status'])}",
               f'源码：{self.commit}',f'父权重 SHA-256：{PARENT_SHA256}',
               '仅比较 Task4 固定终点；不代表旧课程协议的 Task4 合格，也未启动分代自博弈。',
               '留出的历史策略与训练池同源，不称为独立未知对手。旧权重和原提交包不变。',
               '初始化清空全部 Replay 与优化器；规则得分优先，历史对局只作迁移诊断。','',
               '| 臂 | 种子 | 实际动作 | 回合 | 历史曝光 | 得分 | 金币 | 击杀 | 独占/并列第一 | 自杀率 | 炸弹存活 | P95/最大ms | 失败项 |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|']
        for arm,seeds in self.state['arms'].items():
            for seed,p in seeds.items():
                s=p['summary']
                lines.append(f"| {arm} | {seed} | {p['actual_actions']} | {p['rounds']} | {p['exposure']['historical_fraction']:.3f} | {s['mean_score']:.4f} | {s['mean_coins']:.4f} | {s['mean_kills']:.4f} | {s['exclusive_first_rate']:.4f}/{s['tied_first_rate']:.4f} | {s['suicide_rate']:.4f} | {s['bomb_survival_rate']:.4f} | {1000*s['act_p95_seconds']:.2f}/{1000*s['act_max_seconds']:.2f} | {p['failures']} |")
                lines.extend(['',f"{arm}/{seed} 权重：{p['checkpoint']}；SHA-256：{p['checkpoint_sha256']}。",
                              f"留出历史策略结果：{json.dumps(p['heldout']['summary'],ensure_ascii=False)}"])
        for key,title in [('comparison','相对原父模型'),('comparison_control','相对匹配 C 模型')]:
            if key not in self.state:continue
            score=next(r for r in self.state[key]['metrics'] if r['metric']=='score')
            low,high=score['bootstrap_ci95_low'],score['bootstrap_ci95_high']
            first=self.state[key]['first_place_difference']
            lines+=['',f"{title}得分差：{score['mean_difference']:.4f}，95%配对区间 [{low:.4f}, {high:.4f}]。",
                    f"第一名比例差：{first['mean']:.4f}，95%区间 {first['ci95']}。"]
            if low<=0<=high:lines.append('区间包含零，收益证据仍不确定。')
            if first['mean']<0:lines.append('第一名比例下降，需保留这一取舍。')
        if all('33' in self.state['arms'].get(a,{}) for a in ('C','S')):
            third=self.state['arms']['S']['33']['summary']['mean_score']
            control=self.state['arms']['C']['33']['summary']['mean_score']
            parent=self.state['parent_development']['summary']['mean_score']
            lines+=['',f'第三种子开发得分：S={third:.4f}，C={control:.4f}，父基线={parent:.4f}。',
                    '第三种子同时复现两项正向差值。' if third>max(control,parent) else '第三种子未同时复现两项正向差值。']
        for failed in self.state.get('failed_evaluations',[]):
            lines+=['',f"失败终点 {failed['label']}：{json.dumps(failed['result']['summary'],ensure_ascii=False)}；失败项 {failures(failed['result']['raw'])}"]
        if self.state.get('error'):lines+=['',self.state['error']]
        lines+=['','实际动作、回合、冻结对手审计、全部中间结果见 status.json；复现命令见 logs/*.command.json。',
                f"控制器命令：`{sys.executable} experiments/task4_frozen_campaign.py --manifest {self.path}`。"]
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

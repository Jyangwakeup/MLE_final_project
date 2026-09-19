"""Bounded E1/E2/E3 specialist campaign; never invokes a curriculum campaign."""
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
from agent_code.learning_common.effective_learning import VERSION
from experiments.task4_campaign import audit_training, EngineeringFailure, engineering_checks
from experiments.task4_protocol import CANDIDATE
from experiments.compact_audit import audit_run
from experiments.compare_evaluations import compare_evaluations

BEHAVIOR = {'suicide_rate','bomb_survival_rate','invalid_action_rate'}
TERMINAL = {'observed_improvement','no_score_improvement','final_failed','engineering_failure','budget_incomplete','no_eligible_arm'}
THREADS = {k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def failures(raw):
    # Bomb usage and crate yield remain diagnostics, never elimination criteria.
    return [f for f in raw['failures'] if f != 'zero_bomb_round_rate']


def checkpoint_key(point):
    s=point['summary']
    return (-s['mean_score'], -s['first_place_rate'], point['actual_actions'])


def final_key(point):
    s=point['summary']
    return (-s['mean_score'], -s['first_place_rate'], s['suicide_rate'],
            s['act_p95_seconds'], point['actual_actions'], point['seed'])


def arm_key(pair, arm):
    scores=[p['summary']['mean_score'] for p in pair]
    first=[p['summary']['first_place_rate'] for p in pair]
    return (-sum(scores)/2, -sum(first)/2, -min(scores), arm)


def cutoff(start, *, kind, now=None):
    hours={'arm':18,'selection':20,'evaluation':23,'work':24}[kind]
    if (time.time() if now is None else now) >= start + hours*3600:
        raise TimeoutError('Specialist '+kind+' deadline')


def validate_manifest(m):
    if m['schema_version'] != VERSION or set(m['arm_configs']) != {'E1','E2','E3'}:
        raise ValueError('Exploration protocol/arms mismatch')
    if m['training_seeds'] != [22,11,33] or m['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Exploration seeds/parent mismatch')
    groups=[m[k+'_seeds'] for k in ('diagnostic','development','final')]
    if list(map(len,groups)) != [20,100,200] or len(set(sum(groups,[]))) != 320:
        raise ValueError('Exploration data partition mismatch')
    forbidden=set(m['excluded_seeds']) | set(range(20000,20100))
    for values in m['rng_plan'].values(): forbidden.update(values.values())
    if set(sum(groups,[])) & forbidden:
        raise ValueError('Reserved world overlap')
    if len(m['final_seeds']) != 200 or m['final_seeds'] != list(range(m['final_seeds'][0],m['final_seeds'][0]+200)):
        raise ValueError('Final worlds must be contiguous')
    if not 1 <= len(m['cpus']) <= 6 or len(set(m['cpus'])) != len(m['cpus']):
        raise ValueError('Invalid physical cores')
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
            parent_sha256=PARENT_SHA256)
        self.cancel=threading.Event()
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

    def update(self, status, **values):
        self.state.update(status=status,updated_at=datetime.now().astimezone().isoformat(),**values)
        write(self.state_path,self.state)
        print(json.dumps(dict(status=status,**values)),flush=True)

    def execute(self, args, name, cpu, role):
        cutoff(self.start,kind='work');self.check_identity()
        log=self.directory/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        command=['taskset','-c',str(cpu),sys.executable,'experiments/task4_exploration_worker.py',
                 '--exploration-manifest',str(self.path),'--role',role,*args]
        write(log.with_suffix('.command.json'),command)
        with log.open('w') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**THREADS},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while proc.poll() is None:
                    if self.cancel.is_set():raise EngineeringFailure('Cancelled after peer failure')
                    cutoff(self.start,kind='work');time.sleep(.5)
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
        bad=failures(raw)
        write(Path(directory)/'exploration_audit.json',raw)
        fatal=bad if diagnostic else [f for f in bad if f not in BEHAVIOR]
        if fatal:raise EngineeringFailure(str(directory)+': '+str(fatal))
        return raw

    def evaluate(self, checkpoint, arm, label, worlds, role='candidate'):
        cutoff(self.start,kind='evaluation');self.check_identity()
        config=self.m['reference_config'] if role=='reference' else self.m['arm_configs'][arm]
        request=dict(identity=self.identity,role=role,arm=arm,checkpoint_sha256=sha256(checkpoint),worlds=worlds,config=config)
        cache=self.directory/'evaluations'/(label+'.json')
        if cache.exists():
            value=json.loads(cache.read_text())
            if value['request']!=request:raise ValueError('Evaluation cache identity mismatch')
            if any(sha256(p)!=h for p,h in value['artifact_hashes'].items()):raise ValueError('Evaluation cache artifact changed')
            return value
        # Disjoint worlds on disjoint physical cores; parent/child use identical seeds.
        chunks=[worlds[i::len(self.m['cpus'])] for i in range(len(self.m['cpus']))]
        jobs=[]
        for i,chunk in enumerate(chunks):
            name=self.unique(f'{label}_p{i}')
            args=['--config',config,'--mode','evaluate','--device','cpu','--task','4','--agent',self.m['agent'],
                  '--opponents',*(['rule_based_agent']*3),'--seeds',*map(str,chunk),'--n-rounds','1',
                  '--checkpoint',str(checkpoint),'--run-id',name,'--replay-policy','all']
            jobs.append((name,chunk,args,self.m['cpus'][i]))
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures=[pool.submit(self.execute,args,name,cpu,role) for name,chunk,args,cpu in jobs]
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
        summary['first_place_rate']=sum(a['score']==max(x['score'] for x in e['agents']) for a,e in zip(own,episodes))/len(own)
        result=dict(request=request,summary=summary,raw=raw,directories=directories,
                    artifact_hashes={str(Path(d)/f):sha256(Path(d)/f) for d in directories for f in ('metadata.json','episodes.jsonl','timing.jsonl')})
        write(cache,result)
        return result

    def compare(self, child, parent, label):
        result=compare_evaluations(child['directories'],parent['directories'],self.directory/'comparisons'/label,bootstrap_samples=10000)
        from experiments.compare_evaluations import _load_group, _bootstrap_ci
        _,_,a=_load_group(child['directories']);_,_,b=_load_group(parent['directories'])
        differences=[a[k]['exclusive_win']+a[k]['tied_first']-b[k]['exclusive_win']-b[k]['tied_first'] for k in sorted(a)]
        low,high=_bootstrap_ci(differences,'first_place',10000)
        result['first_place_difference']={'mean':sum(differences)/len(differences),'ci95':[low,high]}
        write(self.directory/'comparisons'/label/'exploration_comparison.json',result)
        return result

    def train(self, arm, seed, target, previous=None, diagnostic=False):
        cutoff(self.start,kind='work');self.check_identity()
        name=self.unique(f'{arm}_s{seed}_{"diag" if diagnostic else target}')
        args=['--config',self.m['arm_configs'][arm],'--mode','train','--device','cpu','--task','4',
              '--agent',self.m['agent'],'--opponents',*(['rule_based_agent']*3),'--seed',str(seed),
              '--n-rounds',str(20 if diagnostic else 100000),'--run-id',name,'--replay-policy','all']
        if not diagnostic:args+=['--target-stage-action-steps',str(target)]
        if previous:args+=['--resume-from',previous]
        else:args+=['--transfer-task4-exploration-from-checkpoint',str(ROOT/self.m['parent']['checkpoint'])]
        self.execute(args,name,self.m['cpus'][0],'candidate')
        directory=ROOT/'runs'/name
        audit=audit_training(directory,self.m['agent'],policy=CANDIDATE)
        raw=self.audit(directory,diagnostic)
        from experiments.resume import load_training_snapshot
        snap=load_training_snapshot(directory)
        with (directory/'training.csv').open() as stream:rows=list(csv.DictReader(stream))
        actions=int(rows[-1]['stage_action_steps'])
        if (diagnostic and audit['rounds']!=20) or (not diagnostic and actions<target):
            raise EngineeringFailure('Incomplete training segment')
        if int(rows[-1]['updates'])<=0:raise EngineeringFailure('No training updates')
        checkpoint=directory/'checkpoints/final.pt'
        return dict(run=str(directory),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                    seed=seed,arm=arm,target_actions=target,actual_actions=actions,rounds=snap.round_index,audit=audit,raw=raw)

    def seed(self,arm,seed):
        records=self.state['arms'].setdefault(arm,{}).setdefault(str(seed),dict(history=[]))
        if records.get('complete'):return records['selected']
        previous=records['history'][-1]['run'] if records['history'] else None
        for target in range(20000,100001,20000):
            if any(p['target_actions']==target for p in records['history']):continue
            cutoff(self.start,kind='selection')
            self.update('training',active=dict(arm=arm,seed=seed,target_actions=target))
            trained=self.train(arm,seed,target,previous)
            self.update('development',active=dict(arm=arm,seed=seed,target_actions=target))
            evaluated=self.evaluate(Path(trained['checkpoint']),arm,f'{arm}_s{seed}_a{target}',self.m['development_seeds'])
            comparison=self.compare(evaluated,self.state['parent_development'],f'{arm}_s{seed}_a{target}')
            point=dict(**trained,comparison=comparison,summary=evaluated['summary'],failures=failures(evaluated['raw']),evaluation=evaluated)
            point['eligible']=not point['failures'];records['history'].append(point);previous=point['run']
            self.update('checkpoint_complete',active=dict(arm=arm,seed=seed,target_actions=target,summary=point['summary'],failures=point['failures']))
        usable=[p for p in records['history'] if p['eligible']]
        records.update(complete=True,selected=min(usable,key=checkpoint_key) if usable else None)
        self.update('seed_complete',active=dict(arm=arm,seed=seed))
        return records['selected']

    def select(self):
        for arm in ('E1','E2','E3'):
            if arm not in self.state['diagnostics']:
                cutoff(self.start,kind='arm')
                self.update('diagnostic',active=dict(arm=arm))
                self.state['diagnostics'][arm]=self.train(arm,self.m['diagnostic_training_seeds'][arm],0,diagnostic=True)
                self.update('diagnostic_complete',active=dict(arm=arm))
        if 'parent_development' not in self.state:
            self.update('parent_development')
            self.state['parent_development']=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_development',self.m['development_seeds'],'reference')
            self.update('parent_development_complete')
        pairs={}
        for arm in ('E1','E2','E3'):
            if arm not in self.state['arms']:cutoff(self.start,kind='arm')
            pair=[self.seed(arm,seed) for seed in (22,11)]
            if all(p is not None for p in pair):pairs[arm]=pair
        if not pairs:self.update('no_eligible_arm');return
        selected=min(pairs,key=lambda arm:arm_key(pairs[arm],arm))
        self.update('replication',selected_arm=selected)
        third=self.seed(selected,33)
        cutoff(self.start,kind='selection')
        points=pairs[selected]+([third] if third else [])
        candidate=min(points,key=final_key)
        if 'final_candidate' in self.state and self.state['final_candidate']!=candidate:raise ValueError('Final candidate changed')
        self.update('final_frozen',final_candidate=candidate)

    def run(self):
        try:
            if "final_candidate" not in self.state:
                self.select()
            if "final_candidate" not in self.state:return
            candidate=self.state["final_candidate"];selected=candidate["arm"]
            if sha256(candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('Frozen candidate changed')
            parent=self.evaluate(ROOT/self.m['parent']['checkpoint'],None,'parent_final',self.m['final_seeds'],'reference')
            child=self.evaluate(Path(candidate['checkpoint']),selected,'candidate_final',self.m['final_seeds'])
            comparison=self.compare(child,parent,'final')
            delta=child['summary']['mean_score']-parent['summary']['mean_score']
            status='final_failed' if failures(child['raw']) else 'observed_improvement' if delta>0 else 'no_score_improvement'
            self.update(status,final_parent=parent,final_child=child,comparison=comparison,score_delta=delta)
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
            subprocess.run(['git','commit','-m','Record Task4 specialist exploration outcome'],cwd=ROOT,check=True)
        except (OSError,ValueError,subprocess.CalledProcessError) as exc:
            write(self.directory/'archive_error.json',{'error':str(exc)})

    def report(self):
        lines=['# Task4 专项探索报告','',f"状态：{self.state['status']}",f'源码：{self.commit}',
               f'父权重 SHA-256：{PARENT_SHA256}','',
               '仅评估 Task4；不代表旧课程协议的 Task4 合格。专项初始化不继承父 Replay。',
               'E1 为整体专项配置，E2 仅改变奖励，E3 再改变 gamma。',
               '纯得分奖励按框架回调事件计数；死亡后击杀可能被归入最后一步，不能保证无偏的动作信用分配。','',
               '| 实验 | 种子 | 动作数 | 得分 | 金币 | 击杀 | 第一名 | 自杀率 | 炸弹存活 | P95/ms | 最大/ms | 失败项 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for arm,seeds in self.state['arms'].items():
            for seed,record in seeds.items():
                for p in record['history']:
                    s=p['summary'];lines.append(f"| {arm} | {seed} | {p['actual_actions']} | {s['mean_score']:.4f} | {s['mean_coins']:.4f} | {s['mean_kills']:.4f} | {s['first_place_rate']:.4f} | {s['suicide_rate']:.4f} | {s['bomb_survival_rate']:.4f} | {s['act_p95_seconds']*1000:.2f} | {s['act_max_seconds']*1000:.2f} | {', '.join(p['failures']) or '无'} |")
        if 'comparison' in self.state:
            score=next(r for r in self.state['comparison']['metrics'] if r['metric']=='score')
            low,high=score['bootstrap_ci95_low'],score['bootstrap_ci95_high']
            first=self.state['comparison']['first_place_difference']
            lines+=['',f"最终得分差：{score['mean_difference']:.4f}，95% 配对区间 [{low:.4f}, {high:.4f}]。",
                    f"第一名比例差：{first['mean']:.4f}，95% 区间 {first['ci95']}。"]
            if low<=0<=high:lines.append('区间跨零，得分改善证据仍不确定。')
            if first['mean']<0:lines.append('第一名比例下降，需要明确这一取舍。')
        if self.state.get('error'):lines+=['',self.state['error']]
        lines+=['','完整安全、耗时、权重哈希、回合数和比较区间见同目录 status.json；全部实际命令见 logs/*.command.json。']
        (self.directory/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();Campaign(args.manifest,args.resume).run()

if __name__=='__main__':main()

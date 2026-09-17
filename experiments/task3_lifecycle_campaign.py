"""Bounded, preregistered Task 3 lifecycle repair campaign.

Old plateau evidence is immutable. This controller reuses its metric and stopping
functions, but owns arm identities, the deadline, diagnostic admission and held-out
stages. It never chooses another candidate after confirmation/main failure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import task3_plateau_stopping as plateau
from experiments.task3_retention_prefix import evaluate_gates

VERSION = 'task3-lifecycle-campaign-v1'
ARMS = ('A', 'P', 'R', 'PR')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def diagnostic_point(result):
    selected = result.get('decision', {}).get('selected_checkpoint_run')
    if selected:
        return next(h for h in result['history'] if h['checkpoint_run'] == selected)
    return min(result['history'], key=lambda h: (h['failed_gate_count'], *plateau._candidate_key(h)))


def failure_counts(results):
    counts = dict(retention=0, combat=0, safety=0)
    for result in results.values():
        for name, check in diagnostic_point(result)['gate_checks'].items():
            if check['passed']:
                continue
            kind = ('retention' if name.endswith('_retention') else
                    'combat' if name in ('task3_score_gain', 'task3_combat_gain') else 'safety')
            counts[kind] += 1
    return counts


def next_arm(arms):
    if not arms:
        return 'A'
    baseline = failure_counts(arms['A'])
    # An unresolved safety/engineering failure cannot be tuned away by rewards.
    if any(failure_counts(value)['safety'] for value in arms.values()):
        return None
    order = ('P', 'R') if baseline['retention'] else ('R', 'P')
    for arm in order:
        if arm not in arms:
            return arm
    if ('PR' not in arms
            and failure_counts(arms['P'])['retention'] < baseline['retention']
            and failure_counts(arms['R'])['combat'] < baseline['combat']):
        return 'PR'
    return None


def inspect_diagnostic(run, agent):
    """Audit live own-bomb responsibility independently of guarantee counters."""
    episodes = [json.loads(l) for l in (run / 'episodes.jsonl').read_text().splitlines()]
    deaths = {}
    for episode in episodes:
        own = next(a for a in episode['agents'] if a['name'] == agent)
        if own['dead']:
            deaths[episode['round_index']] = dict(
                causes=own['death_causes'], step=own['death_step'], suicide=bool(own['suicides']))
    failures, traces, previous = [], {}, None
    pending_count = 0
    for line in (run / 'timing.jsonl').open():
        row = json.loads(line)
        if row['agent_name'] != agent:
            continue
        safety = row.get('safety', {})
        pending_count += int(safety.get('own_bomb_pending', False))
        if safety.get('robust_search_timed_out') or safety.get('robust_guarantee_loss'):
            failures.append({'round': row['round_index'], 'step': row['step'], 'reason': 'safety_exception'})
        if (previous is not None and previous['round_index'] == row['round_index']
                and row['step'] == previous['step'] + 1
                and previous['action'] == 'BOMB'
                and not safety.get('own_bomb_pending')):
            # A blocked BOMB may leave capacity available; only fail when it is
            # still unavailable, which rules out a failed placement.
            if not safety.get('physical_mask', [False]*6)[5]:
                failures.append({'round': row['round_index'], 'step': row['step'], 'reason': 'lost_bomb_history'})
        if row['round_index'] in deaths:
            target = traces.setdefault(str(row['round_index']), [])
            target.append({'step': row['step'], 'action': row['action'], 'safety': safety})
            del target[:-12]
        previous = row
    # Conservatively stop on a self-death pending a separate trace diagnosis;
    # neither zero counters nor an unexplained fallback is accepted as admission.
    for number, death in deaths.items():
        if death['suicide']:
            failures.append({'round': number, 'step': death['step'], 'reason': 'self_death_requires_diagnosis'})
    return dict(status='passed' if not failures and pending_count else 'gate_failed',
                rounds=len(episodes), pending_decisions=pending_count, failures=failures,
                deaths=deaths, death_traces=traces)


class Campaign:
    def __init__(self, root, path, resume=False):
        self.root = root
        self.manifest = plateau._json(path)
        m = self.manifest
        if m.get('schema_version') != VERSION:
            raise ValueError('Unsupported campaign protocol')
        if set(m['arm_configs']) != set(ARMS):
            raise ValueError('Campaign requires exactly A/P/R/PR')
        groups = [m[k] for k in ('development_seeds', 'confirmation_seeds', 'main_validation_seeds')]
        if [len(x) for x in groups] != [60, 100, 100] or len(set(sum(groups, []))) != 260:
            raise ValueError('Campaign seed sets must be distinct 60/100/100')
        if set(sum(groups, [])) & set(m['reserved_final_test_seeds']):
            raise ValueError('Campaign overlaps reserved final test worlds')
        if any(set(g) & set(range(19300, 19320)) for g in groups):
            raise ValueError('Old confirmation worlds cannot be reused')
        if m['gates'] != plateau._json(root / 'experiments/task3_plateau_stopping.json')['gates']:
            raise ValueError('Campaign must preserve all original gates')
        self.commit = plateau._git(root, 'rev-parse', 'HEAD')
        if plateau._git(root, 'status', '--porcelain'):
            raise ValueError('Campaign requires a clean committed worktree')
        if not plateau._git_is_ancestor(root, m['source_base'], self.commit):
            raise ValueError('Campaign source is not a descendant of frozen baseline')
        self.identity = dict(source_commit=self.commit, manifest_sha256=digest(m),
                             config_sha256={a: plateau._sha256(root / p) for a, p in m['arm_configs'].items()})
        self.started = datetime.fromisoformat(m['started_at'].replace('Z', '+00:00')).timestamp()
        self.deadline = self.started + m['budget_hours'] * 3600
        self.directory = root / 'runs' / f"{m['campaign_id']}_{self.commit[:7]}"
        self.path = self.directory / 'result.json'
        if self.path.exists():
            if not resume:
                raise FileExistsError('Campaign already exists; use --resume')
            self.state = self.read(self.path)
        else:
            self.state = dict(status='created', arms={}, diagnostics={}, qualified_for_task4=False)
            self.write(self.path, self.state)
        self.resume = resume
        for spec in m['task2_parents'].values():
            plateau._validate_parent(Path(spec['run']), spec['checkpoint_sha256'])

    def write(self, path, value):
        value.update(self.identity)
        plateau._write_json(path, value)

    def read(self, path):
        value = plateau._json(path)
        if any(value.get(k) != v for k, v in self.identity.items()):
            raise ValueError(f'Cannot resume changed campaign identity: {path}')
        return value

    def check_time(self, evaluation=False, new_arm=False):
        limit = self.deadline
        if evaluation:
            limit = min(limit, self.started + self.manifest['evaluation_cutoff_hours'] * 3600)
        if new_arm:
            limit = min(limit, self.started + self.manifest['new_arm_cutoff_hours'] * 3600)
        if time.time() >= limit:
            raise TimeoutError('Preregistered campaign deadline/cutoff reached')

    def arm_manifest(self, arm, diagnostic=False):
        m = json.loads(json.dumps(self.manifest))
        m['run_namespace'] = f"{m['campaign_id']}_{arm}{'_diag' if diagnostic else ''}_{self.identity['manifest_sha256'][:8]}"
        m['deadline_epoch'] = self.deadline
        m['configs'] = {k: m['arm_configs'][arm] for k in ('training','development','confirmation','main_validation')}
        if diagnostic:
            m['plateau_stopping']['interval_rounds'] = m['diagnostic_rounds']
        return m

    def evaluate(self, arm, seed, checkpoint, role, phase, seeds):
        self.check_time(evaluation=True)
        m = self.arm_manifest(arm)
        config = self.manifest['arm_configs']['A' if role == 'parent' else arm]
        return plateau._run_evaluation_pair(
            project_root=self.root, manifest=m, training_seed=seed,
            checkpoint=checkpoint, role=role, phase=phase, evaluation_seeds=seeds,
            config_path=config, cpu_offset={33:0,11:6,22:12}[seed] + (3 if role=='child' else 0))

    def diagnostics(self):
        m = self.arm_manifest('A', diagnostic=True)
        for seed in (33,11,22):
            if str(seed) in self.state['diagnostics']:
                result = self.state['diagnostics'][str(seed)]
            else:
                self.check_time()
                run = plateau._train_segment(
                    project_root=self.root, manifest=m, seed=seed,
                    cumulative_round=m['diagnostic_rounds'],
                    parent_run=Path(m['task2_parents'][str(seed)]['run']), previous_run=None)
                result = inspect_diagnostic(run, m['agent'])
                result.update(run=str(run), checkpoint_sha256=plateau._sha256(run/'checkpoints/final.pt'))
                self.state['diagnostics'][str(seed)] = result
                self.write(self.path,self.state)
        return all(r['status']=='passed' for r in self.state['diagnostics'].values())

    def seed(self, arm, seed):
        m = self.arm_manifest(arm)
        path = self.directory / arm / f'seed_{seed}.json'
        result = self.read(path) if path.exists() else dict(arm=arm, training_seed=seed, history=[], parent=None, decision={})
        if result.get('arm') != arm or result.get('training_seed') != seed:
            raise ValueError('Development arm/training seed identity mismatch')
        parent = Path(m['task2_parents'][str(seed)]['run'])
        worlds=m['development_seeds']
        if result['parent'] is None:
            summaries,runs=self.evaluate(arm,seed,parent/'checkpoints/final.pt','parent','development',worlds)
            result['parent']=dict(summaries=summaries,evaluation_runs=runs)
            self.write(path,result)
        for item in result['history']:
            if plateau._sha256(self.root/'runs'/item['checkpoint_run']/'checkpoints/final.pt') != item['checkpoint_sha256']:
                raise ValueError('Development checkpoint was modified')
        tracker=plateau.PlateauTracker(m['plateau_stopping'],history=result['history'])
        while not tracker.state['stop']:
            self.check_time()
            number=(result['history'][-1]['cumulative_round'] if result['history'] else 0)+50
            if result.get('pending_segment'):
                pending=result['pending_segment'];run=Path(pending['path'])
                if plateau._sha256(run/'checkpoints/final.pt') != pending['sha256']:
                    raise ValueError('Pending segment changed')
            else:
                run=plateau._train_segment(project_root=self.root,manifest=m,seed=seed,cumulative_round=number,
                    parent_run=parent,previous_run=None if not result['history'] else self.root/'runs'/result['history'][-1]['checkpoint_run'])
                result['pending_segment']=dict(path=str(run),sha256=plateau._sha256(run/'checkpoints/final.pt'))
                self.write(path,result)
            child,runs=self.evaluate(arm,seed,run/'checkpoints/final.pt','child',f'development_c{number:04d}',worlds)
            assessment=plateau.build_assessment(cumulative_round=number,checkpoint_run=run.name,
                checkpoint=run/'checkpoints/final.pt',parent=result['parent']['summaries'],child=child,gates=m['gates'])
            assessment['evaluation_runs']=runs
            assessment['bootstrap']=plateau._compare_pair(project_root=self.root,evidence_root=path.parent/f's{seed}',
                phase=f'c{number:04d}',evaluation_seeds=worlds,parent_prefixes=result['parent']['evaluation_runs'],
                child_prefixes=runs,bootstrap_samples=m['bootstrap_samples'])
            result['decision']=tracker.record(assessment);result['history'].append(assessment)
            result.pop('pending_segment',None);self.write(path,result)
            print(f"development arm={arm} seed={seed} rounds={number} eligible={assessment['eligible']} score={assessment['task3_score']:.3f} failures={assessment['failed_gate_count']}",flush=True)
        result['status']='qualified' if result['decision']['selected_checkpoint_run'] else 'gate_failed'
        self.write(path,result)
        return result

    def stage(self, arm, results, stage, worlds):
        path=self.directory/arm/f'{stage}.json'
        result=self.read(path) if path.exists() else dict(arm=arm,stage=stage,evaluation_seeds=worlds,per_seed={},status='running')
        if (result.get('evaluation_seeds') != worlds or result.get('arm') != arm
                or result.get('stage') != stage):
            raise ValueError('Evaluation seed set changed')
        for seed,seed_result in sorted(results.items()):
            checkpoint=plateau._selected_checkpoint(self.root,seed_result)
            if str(seed) in result['per_seed']:
                if result['per_seed'][str(seed)]['checkpoint_sha256'] != plateau._sha256(checkpoint):
                    raise ValueError('Frozen evaluation checkpoint changed')
                continue
            parent=Path(self.manifest['task2_parents'][str(seed)]['run'])/'checkpoints/final.pt'
            parent_metrics,parent_runs=self.evaluate(arm,seed,parent,'parent',stage,worlds)
            child_metrics,child_runs=self.evaluate(arm,seed,checkpoint,'child',stage,worlds)
            passed,checks=evaluate_gates(parent_metrics,child_metrics,self.manifest['gates'])
            result['per_seed'][str(seed)]=dict(status='passed' if passed else 'gate_failed',
                gate_checks=checks,summaries=dict(parent=parent_metrics,child=child_metrics),
                checkpoint=str(checkpoint),checkpoint_sha256=plateau._sha256(checkpoint),
                evaluation_runs=dict(parent=parent_runs,child=child_runs),
                bootstrap=plateau._compare_pair(project_root=self.root,evidence_root=path.parent/stage,
                phase=f's{seed}',evaluation_seeds=worlds,parent_prefixes=parent_runs,child_prefixes=child_runs,
                bootstrap_samples=self.manifest['bootstrap_samples']))
            self.write(path,result)
            print(f'{stage} arm={arm} seed={seed} passed={passed}',flush=True)
        passed=all(v['status']=='passed' for v in result['per_seed'].values())
        result['status']='passed' if passed else 'gate_failed';self.write(path,result)
        return passed,result,str(path)

    def run(self):
        terminal={'passed','stopped_diagnostic_failure','stopped_confirmation_failure','stopped_main_validation_failure',
                  'stopped_development_failure','budget_exhausted'}
        if self.state['status'] in terminal:
            return self.state
        try:
            self.state['status']='running_diagnostics';self.write(self.path,self.state)
            if not self.diagnostics():
                self.state['status']='stopped_diagnostic_failure'
                return self.state
            while True:
                selected=self.state.get('selected_arm')
                if selected:
                    arm=selected;results={int(k):v for k,v in self.state['arms'][arm].items()}
                    break
                active=self.state.get('active_arm')
                arm=active or next_arm(self.state['arms'])
                if arm is None:
                    self.state['status']='stopped_development_failure';return self.state
                if active is None:
                    self.check_time(new_arm=True)
                    self.state['active_arm']=arm
                self.state['status']=f'running_development_{arm}';self.write(self.path,self.state)
                results={33:self.seed(arm,33)}
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures={seed:executor.submit(self.seed,arm,seed) for seed in (11,22)}
                    results.update({seed:f.result() for seed,f in futures.items()})
                self.state['arms'][arm]={str(k):v for k,v in results.items()}
                self.state.pop('active_arm',None)
                if all(v['status']=='qualified' for v in results.values()):
                    self.state['selected_arm']=arm;self.write(self.path,self.state);break
                self.write(self.path,self.state)
            self.state['status']='running_confirmation';self.write(self.path,self.state)
            passed,confirmation,path=self.stage(arm,results,'confirmation',self.manifest['confirmation_seeds'])
            self.state['confirmation']=path
            if not passed:
                self.state['status']='stopped_confirmation_failure';return self.state
            seed=min(results,key=lambda seed:plateau._confirmation_key(confirmation['per_seed'][str(seed)],seed))
            self.state['selected_training_seed']=seed
            self.state['status']='running_main_validation';self.write(self.path,self.state)
            passed,main,path=self.stage(arm,{seed:results[seed]},'main_validation',self.manifest['main_validation_seeds'])
            self.state['main_validation']=path
            self.state['qualified_for_task4']=bool(passed)
            self.state['status']='passed' if passed else 'stopped_main_validation_failure'
        except (TimeoutError,subprocess.TimeoutExpired) as error:
            self.state.update(status='budget_exhausted',error=str(error))
        except Exception as error:
            self.state.update(status='infrastructure_error',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.state['elapsed_hours']=(time.time()-self.started)/3600
            self.write(self.path,self.state)
            self.report()
        return self.state

    def report(self):
        lines=['# Task 3 生命周期修复实验结果','',f"状态：`{self.state['status']}`。",
               f"源码：`{self.commit}`；manifest SHA-256：`{self.identity['manifest_sha256']}`。",'',
               '旧任务Replay保留以隔离修复变量；不证明历史父任务数据无实现缺陷。','',
               '| 实验臂 | 训练种子 | 开发状态 | 选中轮次 | 得分 | 击杀 | 失败门槛 |',
               '|---|---:|---|---:|---:|---:|---|']
        for arm,results in self.state['arms'].items():
            for seed,value in results.items():
                point=diagnostic_point(value)
                failed=', '.join(k for k,v in point['gate_checks'].items() if not v['passed']) or '无'
                lines.append(f"| {arm} | {seed} | {value['status']} | {point['cumulative_round']} | {point['task3_score']:.3f} | {point['task3_kills']:.3f} | {failed} |")
        lines += ['', '未合格模型的表中轮次仅为诊断点，不能作为合格候选。','',
                  '## 诊断准入','']
        for seed,value in self.state['diagnostics'].items():
            lines.append(f"- seed {seed}: {value['status']}；责任存续决策 {value['pending_decisions']}；失败 {json.dumps(value['failures'],ensure_ascii=False)}")
        for stage in ('confirmation','main_validation'):
            lines += ['',f'## {stage}','']
            if not self.state.get(stage):
                lines.append('尚未完成，不计作通过。');continue
            p=Path(self.state[stage]);data=plateau._json(p)
            lines.append(f'完整指标、配对bootstrap和每项门槛：[result]({p})')
            for seed,v in data['per_seed'].items():
                c=v['summaries']['child']['task3'];pmetrics=v['summaries']['parent']['task3']
                lines.append(f"- seed {seed}: {v['status']}，score {c['mean_score']:.3f}（父 {pmetrics['mean_score']:.3f}），kills {c['mean_kills']:.3f}，first {c['first_place_rate']:.1%}，suicide {c['suicide_rate']:.1%}。")
                lines.append(f"  权重 `{v['checkpoint']}`；SHA-256 `{v['checkpoint_sha256']}`。")
                for task,boot in v['bootstrap'].items():
                    for metric in boot['metrics']:
                        if metric['metric'] in ('score','coins','crates','kills'):
                            lines.append(f"  {task}/{metric['metric']}: Δ={metric['mean_difference']:.3f}, CI95=[{metric['bootstrap_ci95_low']:.3f},{metric['bootstrap_ci95_high']:.3f}]")
        lines += ['', 'Task 4 未启动；最终保留测试集未使用。','']
        (self.directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('experiments/task3_lifecycle_campaign.json'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    campaign=Campaign(Path(__file__).resolve().parents[1],args.manifest,args.resume)
    result=campaign.run()
    print(json.dumps({'status':result['status'],'result':str(campaign.path)},indent=2),flush=True)
    return 0 if result['status']=='passed' else 2

if __name__=='__main__':
    raise SystemExit(main())

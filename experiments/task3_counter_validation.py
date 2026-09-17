"""Frozen-only revalidation after the escape-collapse diagnostic correction."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys
import subprocess
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import task3_plateau_stopping as p
from experiments.task3_lifecycle_campaign import digest
from experiments.task3_retention_prefix import evaluate_gates

VERSION = 'task3-counter-validation-v1'


def validate_manifest(m):
    if m.get('schema_version') != VERSION:
        raise ValueError('Unsupported counter validation protocol')
    if set(m['candidates']) != {'11', '22', '33'}:
        raise ValueError('Exactly three frozen candidates are required')
    a, b = m['confirmation_seeds'], m['main_validation_seeds']
    if len(a) != 100 or len(b) != 100 or len(set(a+b)) != 200:
        raise ValueError('Two independent 100-world sets are required')
    if set(a+b) & set(m['excluded_seeds']):
        raise ValueError('Previously used/reserved worlds cannot be blind validation')
    return m


class Validation:
    def __init__(self, root, manifest, resume=False):
        self.root=root;self.m=validate_manifest(p._json(manifest))
        self.commit=p._git(root,'rev-parse','HEAD')
        self.identity=dict(source_commit=self.commit,manifest_sha256=digest(self.m),
                           config_sha256=p._sha256(root/self.m['config']))
        self.deadline=datetime.fromisoformat(self.m['deadline'].replace('Z','+00:00')).timestamp()
        self.directory=root/'runs'/f"{self.m['campaign_id']}_{self.commit[:7]}"
        self.path=self.directory/'result.json'
        if self.path.exists():
            if not resume:raise FileExistsError('Use --resume only for an interrupted same-source run')
            self.state=self.read(self.path)
        else:self.state=dict(status='created',qualified_for_task3=False)
        self.check()

    def read(self,path):
        value=p._json(path)
        if any(value.get(k)!=v for k,v in self.identity.items()):
            raise ValueError('Frozen validation identity changed')
        return value

    def write(self,path,value):
        value.update(self.identity);p._write_json(path,value)

    def check(self):
        cutoff=datetime.fromisoformat(self.m['evaluation_cutoff'].replace('Z','+00:00')).timestamp()
        if time.time()>=min(self.deadline,cutoff):raise TimeoutError('Validation deadline/cutoff reached')
        if p._git(self.root,'rev-parse','HEAD')!=self.commit or p._git(self.root,'status','--porcelain'):
            raise ValueError('Validation requires unchanged clean committed source')
        if p._sha256(self.root/self.m['config'])!=self.identity['config_sha256']:
            raise ValueError('Frozen config changed')
        for item in self.m['candidates'].values():
            for role in ('parent','child'):
                if p._sha256(Path(item[role]))!=item[role+'_sha256']:
                    raise ValueError('Frozen checkpoint changed')

    def seed(self,stage,seed):
        path=self.directory/stage/f'seed_{seed}.json'
        if path.exists():return self.read(path)
        self.check()
        worlds=self.m[stage+'_seeds']; metrics={};runs={}
        m=dict(self.m,run_namespace=self.m['campaign_id']+'_'+self.identity['manifest_sha256'][:8],
               deadline_epoch=self.deadline)
        for role in ('parent','child'):
            self.check()
            metrics[role],runs[role]=p._run_evaluation_pair(project_root=self.root,manifest=m,
                training_seed=seed,checkpoint=Path(self.m['candidates'][str(seed)][role]),
                role=role,phase=stage,evaluation_seeds=worlds,config_path=self.m['config'],
                cpu_offset={11:0,22:6,33:12}[seed]+(3 if role=='child' else 0))
        passed,checks=evaluate_gates(metrics['parent'],metrics['child'],self.m['gates'])
        result=dict(status='passed' if passed else 'gate_failed',training_seed=seed,
                    evaluation_seeds=worlds,checkpoint=self.m['candidates'][str(seed)]['child'],
                    checkpoint_sha256=self.m['candidates'][str(seed)]['child_sha256'],
                    summaries=metrics,gate_checks=checks,evaluation_runs=runs,
                    bootstrap=p._compare_pair(project_root=self.root,evidence_root=self.directory/stage,
                    phase=f's{seed}',evaluation_seeds=worlds,parent_prefixes=runs['parent'],
                    child_prefixes=runs['child'],bootstrap_samples=10000))
        self.write(path,result)
        print(f'{stage} seed={seed} {result["status"]}',flush=True)
        return result

    def run(self):
        if self.state['status'] in {'passed','stopped_confirmation_failure','stopped_main_validation_failure','budget_exhausted'}:
            return self.state
        try:
            self.state['status']='running_confirmation';self.write(self.path,self.state)
            with ThreadPoolExecutor(max_workers=3) as ex:
                tasks={seed:ex.submit(self.seed,'confirmation',seed) for seed in (11,22,33)}
                confirmation={str(seed):task.result() for seed,task in tasks.items()}
            self.state['confirmation']=confirmation
            if not all(x['status']=='passed' for x in confirmation.values()):
                self.state['status']='stopped_confirmation_failure'
                return self.state
            seed=self.m['selected_training_seed']
            self.state.update(status='running_main_validation',selected_training_seed=seed)
            self.write(self.path,self.state)
            main=self.seed('main_validation',seed);self.state['main_validation']=main
            ok=main['status']=='passed'
            self.state.update(status='passed' if ok else 'stopped_main_validation_failure',qualified_for_task3=ok)
        except (TimeoutError, subprocess.TimeoutExpired) as error:self.state.update(status='budget_exhausted',error=str(error))
        except Exception as error:
            self.state.update(status='infrastructure_error',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.state['finished_at']=datetime.now().astimezone().isoformat()
            self.write(self.path,self.state)
        return self.state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('experiments/task3_counter_validation.json'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    campaign=Validation(Path(__file__).resolve().parents[1],args.manifest,args.resume)
    result=campaign.run();print(result['status'],campaign.path,flush=True)
    return 0 if result['status']=='passed' else 2

if __name__=='__main__':raise SystemExit(main())

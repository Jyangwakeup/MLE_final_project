"""Four-hour engineering admission only; never starts formal Task 4 training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys
import threading
import time

if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.task4_campaign import Campaign, EngineeringFailure, p
from experiments.task4_transfer import digest, PARENT_SHA256

VERSION='order-equivalence-admission-v1'
TERMINAL={'admission_passed','admission_failed','budget_exhausted'}


def validate_evidence(root,manifest):
    for name,record in manifest['prerequisites'].items():
        path=root/record['path']
        if p._sha256(path)!=record['sha256']:raise ValueError('Changed admission evidence: '+name)
    benchmark=p._json(root/manifest['prerequisites']['benchmark']['path'])
    if (benchmark['status']!='historical_passed' or benchmark.get('failed_steps')
            or benchmark.get('search_budget_ms')!=400 or benchmark.get('repetitions')!=10
            or benchmark.get('warmups')!=1 or len(benchmark.get('cpu',[]))!=1):
        raise ValueError('Historical search did not pass the complete protocol')
    records=benchmark.get('states',[])
    if [r['step'] for r in records]!=list(range(38,58)):
        raise ValueError('Historical corpus incomplete')
    for record in records:
        if set(record['warmup'])!={'old','new'} or record['warmup']['new']['search_result']['timed_out']:
            raise ValueError('Invalid historical warmup')
        samples=record['samples']
        expected={(v,i) for v in ('old','new') for i in range(10)}
        if len(samples)!=20 or {(s['implementation'],s['repeat']) for s in samples}!=expected:
            raise ValueError('Incomplete paired benchmark samples')
        if any(s['search_result']['timed_out'] for s in samples if s['implementation']=='new'):
            raise ValueError('Optimized search timed out')
    from experiments.run import _source_hash
    if benchmark['runtime_source_hash']!=_source_hash('double_dqn_continuous_v2_agent'):
        raise ValueError('Benchmark runtime source changed')
    for path,sha in benchmark['benchmark_source_hashes'].items():
        if p._sha256(root/path)!=sha:raise ValueError('Benchmark helper source changed')
    if benchmark['corpus_sha256']!=p._sha256(root/'experiments/results/task4_shared_parent_20260917/failure_states.json'):
        raise ValueError('Historical corpus changed')
    if benchmark['production_sha256']!=p._sha256(root/'agent_code/team_agent/opponent_transitions.py'):
        raise ValueError('Benchmark used different production source')
    if benchmark['reference_sha256']!=p._sha256(root/'tests/reference_opponent_transitions.py'):
        raise ValueError('Benchmark reference changed')
    log=(root/manifest['prerequisites']['unittest']['path']).read_text()
    match=re.search(r'Ran (\d+) tests in [^\n]+\n\nOK(?: \(skipped=\d+\))?\s*$',log)
    if not match or int(match[1])!=manifest['prerequisites']['unittest']['test_count'] or int(match[1])<408:
        raise ValueError('Complete unittest did not pass')


class OrderAdmission(Campaign):
    def __init__(self,root,manifest,resume=False):
        self.root=Path(root);self.manifest_path=Path(manifest).resolve();self.m=p._json(self.manifest_path)
        if (self.m['schema_version']!=VERSION or self.m['budget_hours']!=4
                or self.m['training_seeds']!=[11,22,33] or set(self.m['arm_configs'])!={'A'}
                or self.m['parent']['checkpoint_sha256']!=PARENT_SHA256
                or self.m['diagnostic_training_seeds']!={'11':22411,'22':22422,'33':22433}):
            raise ValueError('Unexpected order-equivalence admission contract')
        validate_evidence(self.root,self.m)
        self.commit=p._git(self.root,'rev-parse','HEAD')
        self.identity=dict(source_commit=self.commit,manifest_sha256=digest(self.m),
            config_sha256={a:p._sha256(self.root/c) for a,c in self.m['arm_configs'].items()},
            parent_sha256=PARENT_SHA256,opponent_hashes=self.m['opponent_hashes'])
        self.started=datetime.fromisoformat(self.m['started_at'].replace('Z','+00:00')).timestamp()
        self.deadline=self.started+4*3600
        self.directory=self.root/'runs'/f"{self.m['campaign_id']}_{self.commit[:7]}"
        self.path=self.directory/'result.json';self.stop=threading.Event()
        if self.path.exists():
            if not resume:raise FileExistsError('Use --resume only for same-identity infrastructure recovery')
            self.state=self.read(self.path)
        else:self.state=dict(status='created',diagnostics={},task4_qualified=False)
        self.check_identity()

    def check_time(self,**kwargs):
        if time.time()>=self.deadline:raise TimeoutError('Four-hour order-equivalence deadline')

    def run(self):
        if self.state['status'] in TERMINAL:return self.state
        try:
            self.state['status']='diagnostics';self.write(self.path,self.state)
            for seed in (22,11,33):
                if str(seed) in self.state['diagnostics']:continue
                validate_evidence(self.root,self.m)
                self.state['active_seed']=seed;self.write(self.path,self.state)
                value=self.train('A',self.m['diagnostic_training_seeds'][str(seed)],20,diagnostic=True)
                self.state['diagnostics'][str(seed)]=value;self.write(self.path,self.state)
                print('diagnostic completed',seed,flush=True)
            self.check_time()
            self.state['status']='admission_passed'
        except TimeoutError as exc:self.state.update(status='budget_exhausted',error=str(exc))
        except EngineeringFailure as exc:self.state.update(status='admission_failed',error=str(exc))
        except Exception as exc:
            self.state.update(status='infrastructure_error',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            self.state['finished_at']=datetime.now(timezone.utc).isoformat();self.write(self.path,self.state)
        return self.state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('experiments/order_equivalence_admission.json'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    campaign=OrderAdmission(Path(__file__).resolve().parents[1],args.manifest,args.resume)
    result=campaign.run();print(result['status'],campaign.path,flush=True)
    return 0 if result['status']=='admission_passed' else 2

if __name__=='__main__':raise SystemExit(main())

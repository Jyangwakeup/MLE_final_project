"""Paired uncached/cached full-search timings on three immutable failure corpora."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import os,sys,time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.benchmark_opponent_orders import sha, write
from experiments.run import _source_hash
from tests.test_vectorized_viability import records, search_arguments, CORPORA
from tests import reference_vectorized_survival as old
from agent_code.team_agent import controllable_survival as new
ROOT=Path(__file__).resolve().parents[1]

def measure(record,module):
    actions,remaining=search_arguments(record);start=time.perf_counter()
    result=module.controllable_survival_actions(record['state'],actions,remaining_steps=remaining,budget_ms=400)
    return dict(search_seconds=time.perf_counter()-start,search_result=asdict(result))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/viability_cache/benchmark.json'))
    parser.add_argument('--deadline',default='2026-09-17T21:17:31Z');args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp()
    if len(os.sched_getaffinity(0))!=1:raise ValueError('Pin to a single physical CPU')
    paths=[Path(__file__).resolve(),ROOT/'tests/test_vectorized_viability.py',ROOT/'tests/test_order_equivalence.py',ROOT/'tests/reference_vectorized_survival.py',ROOT/'experiments/benchmark_opponent_orders.py']
    result=dict(status='running',protocol='cached-viability-v2',search_budget_ms=400,repetitions=10,warmups=1,
        cpu=list(os.sched_getaffinity(0)),runtime_source_hash=_source_hash('double_dqn_continuous_v2_agent'),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        corpus_hashes={name:sha(ROOT/'experiments/results'/name/'failure_states.json') for name in CORPORA},states=[])
    try:
        for record in records():
            entry=dict(corpus=record['corpus'],step=record['state']['step'],warmup={},samples=[]);result['states'].append(entry)
            for name,module in [('old',old),('new',new)]:
                if time.time()>=deadline:raise TimeoutError('Optimization deadline')
                entry['warmup'][name]=measure(record,module)
            for repeat in range(10):
                for name in (('old','new') if repeat%2==0 else ('new','old')):
                    if time.time()>=deadline:raise TimeoutError('Optimization deadline')
                    entry['samples'].append(dict(repeat=repeat,implementation=name,**measure(record,old if name=='old' else new)))
            write(args.output,result);print(record['corpus'],entry['step'],flush=True)
        if time.time()>=deadline:raise TimeoutError('Optimization deadline')
        bad=[(r['corpus'],r['step']) for r in result['states'] if r['warmup']['new']['search_result']['timed_out'] or any(s['search_result']['timed_out'] for s in r['samples'] if s['implementation']=='new')]
        result.update(status='historical_passed' if not bad else 'historical_timeout_failure',failed_states=bad)
    except TimeoutError as exc:result.update(status='budget_exhausted',error=str(exc))
    finally:result['finished_at']=datetime.now(timezone.utc).isoformat();write(args.output,result)
    return 0 if result['status']=='historical_passed' else 2
if __name__=='__main__':raise SystemExit(main())

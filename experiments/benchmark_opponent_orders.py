"""Paired historical-state benchmark; this does not evaluate or train a model."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, time
from unittest.mock import patch

if __package__ in (None, ''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent_code.team_agent import controllable_survival as survival
from agent_code.team_agent import opponent_transitions as optimized
from experiments.run import _source_hash
from tests import reference_opponent_transitions as reference
from tests.test_order_equivalence import historical_states, search_arguments, native, CORPUS

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(native(value),indent=2)+'\n');temporary.replace(path)

def measure(record,implementation):
    actions,remaining=search_arguments(record)
    start=perf_counter()
    scenarios=[implementation.enumerate_opponent_transition_scenarios(record['state'],a,progress_world=True) for a in actions]
    enumeration=perf_counter()-start
    with patch.object(survival,'enumerate_opponent_transition_scenarios',implementation.enumerate_opponent_transition_scenarios):
        start=perf_counter()
        result=survival.controllable_survival_actions(record['state'],actions,remaining_steps=remaining,budget_ms=400)
        elapsed=perf_counter()-start
    return dict(enumeration_seconds=enumeration,search_seconds=elapsed,
                unique_scenarios=sum(map(len,scenarios)),search_result=asdict(result))

def simulation_count(record,implementation):
    actions,_=search_arguments(record);count=0;original=implementation._apply_action_phase
    def counted(*args,**kwargs):
        nonlocal count
        count+=1;return original(*args,**kwargs)
    with patch.object(implementation,'_apply_action_phase',counted):
        for a in actions:implementation.enumerate_opponent_transition_scenarios(record['state'],a,progress_world=True)
    return count

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/order_admission/benchmark.json'))
    parser.add_argument('--deadline',default='2026-09-17T21:17:31Z')
    args=parser.parse_args();deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp()
    if len(os.sched_getaffinity(0))!=1:raise ValueError('Pin benchmark to one physical CPU with taskset')
    result=dict(status='running',search_budget_ms=400,
        runtime_source_hash=_source_hash('double_dqn_continuous_v2_agent'),
        benchmark_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'tests/test_order_equivalence.py',ROOT/'tests/reference_opponent_transitions.py']},cpu=list(os.sched_getaffinity(0)),repetitions=10,warmups=1,
        corpus_sha256=sha(CORPUS),production_sha256=sha(ROOT/'agent_code/team_agent/opponent_transitions.py'),
        reference_sha256=sha(ROOT/'tests/reference_opponent_transitions.py'),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},states=[])
    implementations={'old':reference,'new':optimized}
    try:
        for record in historical_states():
            entry=dict(step=record['state']['step'],simulation_counts={},warmup={},samples=[])
            result['states'].append(entry)
            for name,impl in implementations.items():
                if time()>=deadline:raise TimeoutError('Four-hour deadline')
                entry['simulation_counts'][name]=simulation_count(record,impl)
                entry['warmup'][name]=measure(record,impl)
            for repeat in range(10):
                for name in (('old','new') if repeat%2==0 else ('new','old')):
                    if time()>=deadline:raise TimeoutError('Four-hour deadline')
                    entry['samples'].append(dict(repeat=repeat,implementation=name,**measure(record,implementations[name])))
            write(args.output,result)
            print('historical step',entry['step'],'simulations',entry['simulation_counts'],flush=True)
        bad=[x['step'] for x in result['states'] if x['warmup']['new']['search_result']['timed_out'] or
             any(s['search_result']['timed_out'] for s in x['samples'] if s['implementation']=='new')]
        if time()>=deadline:raise TimeoutError('Four-hour deadline before final result')
        result.update(status='historical_timeout_failure' if bad else 'historical_passed',failed_steps=bad)
    except TimeoutError as exc:result.update(status='budget_exhausted',error=str(exc))
    finally:
        result['finished_at']=datetime.now(timezone.utc).isoformat();write(args.output,result)
    return 0 if result['status']=='historical_passed' else 2

if __name__=='__main__':raise SystemExit(main())

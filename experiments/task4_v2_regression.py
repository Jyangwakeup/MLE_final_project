"""Offline complete proofs for the newly observed 262.60ms failure prefix."""
import gc
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.team_agent.controllable_survival import controllable_survival_actions as solve
from agent_code.team_agent.opponent_transitions import ACTIONS
from tests.reference_grid_survival import controllable_survival_actions as reference
from experiments.task4_transfer import sha256

CORPUS = 'experiments/results/compact_admission_20260917/frozen/run_evidence/compact_ae2b981_diagnostic_33/task4_failure_states.pkl'
OUT = 'experiments/results/task4_300ms_20260918/regression'


def main():
    import numpy as np
    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=False)
    rows = pickle.loads((ROOT / CORPUS).read_bytes())
    assert len(rows) == 20 and rows[-1]['state']['step'] == 88
    samples = []
    result = dict(status='running', states=len(rows), repetitions=10,
                  corpus_sha256=sha256(ROOT / CORPUS), affinity=sorted(os.sched_getaffinity(0)),
                  gc_enabled=gc.isenabled(), search_budget_ms=400, offline_budget_ms=60000)
    try:
        assert len(result['affinity']) == 1 and result['gc_enabled']
        for index, row in enumerate(rows):
            state, diagnostic = row['state'], row['safety']
            placed = diagnostic['own_bomb_placed_step']
            horizon = max(1, 7 - (state['step'] - placed)) if diagnostic['own_bomb_pending'] else 7
            kw = dict(remaining_steps=horizon, consider_opponent_rearming=True)
            old = reference(state, ACTIONS, budget_ms=60000, **kw)
            new = solve(state, ACTIONS, budget_ms=60000, **kw)
            assert not old.timed_out and not new.timed_out
            assert asdict(old) == asdict(new), (index, old, new)
            solve(state, ACTIONS, budget_ms=400, **kw)  # Unmeasured warmup.
            for repeat in range(10):
                stats = {}
                cpu, start = time.process_time(), time.perf_counter()
                actual = solve(state, ACTIONS, budget_ms=400, work_stats=stats, **kw)
                sample = dict(index=index, round=state['round'], step=state['step'],
                              repeat=repeat, seconds=time.perf_counter()-start,
                              cpu_seconds=time.process_time()-cpu, work=stats,
                              timed_out=actual.timed_out)
                samples.append(sample)
                assert asdict(actual) == asdict(old)
            print('complete', index, 'step', state['step'], flush=True)
        seconds = [s['seconds'] for s in samples]
        result.update(p95_ms=float(np.percentile(seconds,95)*1000), max_ms=max(seconds)*1000,
                      status='passed')
        assert result['p95_ms'] <= 100 and result['max_ms'] <= 300
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        (out / 'measurements.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
        (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

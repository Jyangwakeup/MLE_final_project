"""Fail-fast campaign worker; model and game execution remain unchanged."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import task4_worker
from experiments.task4_protocol import VERSION, OBSERVED_VERSION, ALIGNED_VERSION, MODERN, limits


def checked_failure(original, policy, state, action, safety, **kwargs):
    reason = original(state, action, safety, **kwargs)
    if reason:
        return reason
    if kwargs['elapsed'] > policy.maximum:
        return 'campaign_act_max_exceeded'
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--campaign-manifest', required=True)
    parser.add_argument('--evaluation-role', choices=('candidate', 'reference', 'historical_reference'), required=True)
    args, remaining = parser.parse_known_args(argv)
    manifest = json.loads(Path(args.campaign_manifest).read_text())
    if manifest['schema_version'] not in MODERN:
        raise ValueError('v2 worker requires v2 manifest')
    if args.evaluation_role=='historical_reference' and manifest['schema_version']!=ALIGNED_VERSION:
        raise ValueError('Historical role is exclusive to v4 retention')
    if args.evaluation_role=='historical_reference':
        start=remaining.index('--seeds')+1;end=remaining.index('--n-rounds')
        if list(map(int,remaining[start:end]))!=list(range(24000,24060)) or remaining[remaining.index('--task')+1] not in ('1','2','3'):
            raise ValueError('Historical reference only allowed on registered retention worlds')
    policy = limits(manifest['schema_version'], args.evaluation_role)
    config = remaining[remaining.index('--config') + 1]
    allowed = ([manifest['historical_reference_config']] if args.evaluation_role=='historical_reference' and manifest['schema_version']==ALIGNED_VERSION else
               [manifest['reference_config']] if args.evaluation_role == 'reference'
               else list(manifest['arm_configs'].values()))
    if Path(config).resolve() not in [(ROOT / value).resolve() for value in allowed]:
        raise ValueError('Worker role/config mismatch')
    if args.evaluation_role != 'candidate' and remaining[remaining.index('--mode') + 1] != 'evaluate':
        raise ValueError('Reference may only evaluate')
    replay = None
    if manifest['schema_version']!=ALIGNED_VERSION and '--seed' in remaining and remaining[remaining.index('--seed') + 1] == '22433':
        import pickle
        replay = {row['state']['step']: row for row in pickle.loads((ROOT / manifest['failure_corpus']).read_bytes())}
    matched = []
    from experiments.task4_reference_observation import ReferenceObservation
    observation=ReferenceObservation() if manifest['schema_version']==OBSERVED_VERSION and args.evaluation_role=='reference' else None
    def checked(state, action, safety, **kw):
        reason = checked_failure(original, policy, state, action, safety, **kw)
        if observation:
            reason,record=observation.inspect(state,action,safety,reason,original,**kw)
            if record:
                target = ROOT / 'runs' / remaining[remaining.index('--run-id') + 1]
                # Evaluation uses one child directory per registered world.
                active=[p for p in target.glob('*/timing.jsonl') if p.parent.joinpath('metadata.json').exists()]
                current=max(active,key=lambda p:p.stat().st_mtime).parent
                with (current/'reference_observations.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
        if replay and state['round'] == 13 and state['step'] in replay:
            from experiments.task4_v2_evidence import fingerprint
            expected = replay[state['step']]
            same = fingerprint(state) == fingerprint(expected['state']) and action == expected['action']
            if same: matched.append(state['step'])
            target = ROOT / 'runs' / remaining[remaining.index('--run-id') + 1]
            (target / 'failure_regression.json').write_text(json.dumps(dict(
                matched=same, matched_steps=matched, round=13, step=state['step'],
                think_time=kw['elapsed'], state_sha256=fingerprint(state)), indent=2)+'\n')
            if not same: return 'diagnostic_trajectory_mismatch'
        return reason
    original = task4_worker.safety_failure
    task4_worker.safety_failure = checked
    try:
        return task4_worker.main(remaining)
    finally:
        task4_worker.safety_failure = original


if __name__ == '__main__':
    raise SystemExit(main())

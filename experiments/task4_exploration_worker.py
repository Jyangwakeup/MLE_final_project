"""Specialist fail-fast worker with unchanged v9 decision implementation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import task4_worker
from experiments.task4_transfer import digest, sha256
from agent_code.learning_common.effective_learning import VERSION


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exploration-manifest', required=True)
    parser.add_argument('--role', choices=['candidate','reference'], required=True)
    args, remaining = parser.parse_known_args(argv)
    manifest = json.loads(Path(args.exploration_manifest).read_text())
    if manifest['schema_version'] != VERSION:
        raise ValueError('Wrong specialist worker protocol')
    config_path = Path(remaining[remaining.index('--config')+1]).resolve()
    allowed = ([manifest['reference_config']] if args.role == 'reference' else list(manifest['arm_configs'].values()))
    if config_path not in [(ROOT/p).resolve() for p in allowed]:
        raise ValueError('Specialist role/config mismatch')
    if args.role == 'reference':
        if remaining[remaining.index('--mode')+1] != 'evaluate':
            raise ValueError('Reference may only evaluate')
        checkpoint = remaining[remaining.index('--checkpoint')+1]
        if sha256(checkpoint) != manifest['parent']['checkpoint_sha256']:
            raise ValueError('Reference checkpoint mismatch')
    elif remaining[remaining.index('--mode')+1] == 'evaluate':
        import torch
        from experiments.task4_exploration_transfer import validate_resume
        payload=torch.load(remaining[remaining.index('--checkpoint')+1],map_location='cpu',weights_only=True)
        validate_resume(payload.get('transfer_contract'),root=ROOT,config_path=config_path,seed=payload['agent_seed'])
    original = task4_worker.safety_failure
    def checked(state, action, safety, **kwargs):
        return original(state, action, safety, **kwargs) or (
            'exploration_act_max_exceeded' if kwargs['elapsed'] > .300 else None)
    task4_worker.safety_failure = checked
    try:
        return task4_worker.main(remaining)
    finally:
        task4_worker.safety_failure = original

if __name__ == '__main__':
    raise SystemExit(main())

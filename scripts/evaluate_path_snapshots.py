"""Freeze every saved policy and select by Task 1 development performance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def rank(row, steps):
    completion = float(row['mean_all_coins_completion_steps'] or 'inf')
    loop = float(row['long_wait_loop_rate']) + float(row['long_ping_pong_loop_rate'])
    return (float(row['all_coins_rate']), float(row['mean_coins']),
            float(row['coins_per_100_steps']), -completion, -loop, -steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--config', default='experiments/configs/cnn_path_task1_e2_r3.json')
    args = parser.parse_args()
    run = Path(args.run)
    candidates = sorted((run / 'checkpoints/snapshots').glob('policy_*.pt'))
    candidates.append(run / 'checkpoints/final.pt')
    results = []
    import torch
    for checkpoint in candidates:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        steps = int(payload['stage_action_steps'])
        del payload
        run_id = f'{run.name}_dev_{checkpoint.stem}'
        subprocess.run([sys.executable, '-m', 'experiments.run', '--config', args.config,
                        '--mode', 'evaluate', '--task', '1', '--agent',
                        'cnn_path_double_dqn_agent', '--device', 'cpu',
                        '--checkpoint', str(checkpoint), '--seeds',
                        '10001', '10002', '10003', '10004', '10005',
                        '--n-rounds', '20', '--run-id', run_id], check=True)
        summaries = list((Path('runs') / run_id).rglob('summary.csv'))
        aggregate = []
        for summary in summaries:
            with summary.open() as handle:
                aggregate.extend(r for r in csv.DictReader(handle) if r['run_id'] == 'AVERAGE')
        if len(aggregate) != 1:
            raise RuntimeError(f'Expected one aggregate for {run_id}, found {len(aggregate)}')
        row = aggregate[0]
        hasher = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(block)
        digest = hasher.hexdigest()
        results.append({'checkpoint': str(checkpoint), 'steps': steps,
                        'sha256': digest, 'metrics': row})
        (run / 'snapshot_evaluations.json').write_text(json.dumps(results, indent=2))
    best = max(results, key=lambda r: rank(r['metrics'], r['steps']))
    shutil.copy2(best['checkpoint'], run / 'checkpoints/best_task1.pt')
    (run / 'best_task1_selection.json').write_text(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()

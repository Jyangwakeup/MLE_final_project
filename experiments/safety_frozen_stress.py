"""Frozen engineering stress test; no training, held-out qualification, or retries."""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run import _source_hash
from experiments.task4_campaign import engineering_checks
from experiments.task3_retention_prefix import summarize_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'experiments/safety_v6_stress_manifest.json')
    manifest_path = parser.parse_args(argv).manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True):
        raise RuntimeError('Freeze a clean source before evaluation')
    if _source_hash('double_dqn_continuous_v2_agent') != manifest['runtime_source_sha256']:
        raise RuntimeError('Runtime identity changed')
    identities = {manifest['config']: manifest['config_sha256'],
                  manifest['parent_checkpoint']: manifest['parent_sha256'],
                  **manifest['opponent_hashes']}
    for path, expected in identities.items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Identity mismatch: ' + path)
    deadline = datetime.fromisoformat(manifest['deadline_utc'].replace('Z', '+00:00')).timestamp()
    if time.time() >= deadline:
        raise RuntimeError('Budget expired')
    directory = ROOT / 'runs' / ('safety_stress_' + commit[:7])
    directory.mkdir(exist_ok=False)
    state = dict(status='running', source_commit=commit, manifest_sha256=sha(manifest_path),
                 jobs={}, task4_qualified=False)
    stop = threading.Event()
    def save():
        temporary = directory / 'result.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(directory / 'result.json')
    env = {**os.environ, **{key: '1' for key in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    def work(task, cpu):
        name = f'safety_stress_task{task}_{commit[:7]}'
        target = ROOT / 'runs' / name
        if target.exists():
            raise FileExistsError(target)
        args = ['taskset', '-c', str(cpu), sys.executable, 'experiments/task4_worker.py',
                '--config', manifest['config'], '--mode', 'evaluate', '--device', 'cpu',
                '--task', str(task), '--agent', 'double_dqn_continuous_v2_agent',
                '--seeds', *map(str, manifest['engineering_seeds']), '--n-rounds', '1',
                '--checkpoint', str(ROOT / manifest['parent_checkpoint']),
                '--run-id', name, '--replay-policy', 'all']
        (directory / f'task{task}_command.json').write_text(json.dumps(args, indent=2)+'\n')
        audited = set()
        def check_deaths():
            for path in target.rglob('episodes.jsonl'):
                if path in audited:
                    continue
                lines = path.read_text().splitlines()
                if not lines:
                    continue
                for line in lines:
                    episode = json.loads(line)
                    own = next(a for a in episode['agents'] if a['name']=='double_dqn_continuous_v2_agent')
                    if own['suicides']:
                        raise RuntimeError(f'Self death requires investigation: {path}')
                audited.add(path)
        with (directory / f'task{task}.log').open('w') as log:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if stop.is_set() or time.time() >= deadline:
                        raise RuntimeError('Peer failure or budget cutoff')
                    check_deaths()
                    time.sleep(1)
                check_deaths()
                if process.returncode:
                    raise RuntimeError(f'Task {task} worker failed: {process.returncode}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        summary = summarize_evaluation(target, 'double_dqn_continuous_v2_agent', manifest['engineering_seeds'])
        episodes = [json.loads(line) for path in target.rglob('episodes.jsonl')
                    for line in path.read_text().splitlines()]
        if len(episodes) != len(manifest['engineering_seeds']):
            raise RuntimeError('Incomplete evaluation')
        summary['mean_bombs'] = sum(a['bombs'] for e in episodes for a in e['agents']
                    if a['name']=='double_dqn_continuous_v2_agent') / len(episodes)
        failures = engineering_checks({f'task{task}': summary})
        if summary['invalid_action_rate'] > .01:
            failures.append('invalid_action_rate')
        if task > 1 and (summary['suicide_rate'] > .05 or summary['bomb_survival_rate'] < .95):
            failures.append('bomb_survival_or_suicide')
        if task > 2 and summary['zero_bomb_round_rate'] > .1:
            failures.append('zero_bomb_round_rate')
        return str(task), dict(run=str(target), command=args, summary=summary, failures=failures)
    save()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work, task, cpu) for task, cpu in zip(manifest['tasks'], manifest['cpus'])]
            try:
                for future in as_completed(futures):
                    task, result = future.result()
                    state['jobs'][task] = result
                    save()
                    if result['failures']:
                        raise RuntimeError(f'Task {task} gates failed: {result["failures"]}')
                    print('Task', task, 'passed', flush=True)
            except BaseException:
                stop.set()
                raise
        state['status'] = 'passed'
    except Exception as exc:
        state.update(status='stopped_failure', error=repr(exc))
    finally:
        save()
        print(state['status'], state.get('error', ''), flush=True)
    return 0 if state['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

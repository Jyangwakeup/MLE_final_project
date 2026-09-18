"""Detached campaign owner: retain status and commit terminal evidence locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.task4_campaign import Campaign, TERMINAL


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def finish(campaign):
    state = campaign.state
    out = ROOT/'experiments/results/task4_300ms_20260918/terminal'
    if out.exists():
        raise FileExistsError('Terminal archive already exists')
    out.mkdir(parents=True)
    shutil.copytree(campaign.directory, out/'controller')
    inventory = []
    checkpoints = []
    for directory in sorted((ROOT/'runs').glob(campaign.m['campaign_id']+'_*')):
        if directory == campaign.directory:
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file():
                continue
            entry = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path))
            inventory.append(entry)
            if path.name in ('final.pt', 'learner.pt'):
                checkpoints.append(entry)
            if (path.name in ('task4_safety_failure.json','task4_failure_states.pkl','v2_raw_audit.json',
                              'failure_regression.json','episodes.jsonl','metadata.json')
                    or path.name.startswith('task4_death_states')):
                target = out/'run_evidence'/path.relative_to(ROOT/'runs')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    write(out/'raw_artifact_inventory.json', inventory)
    write(out/'checkpoints.json', checkpoints)
    status = state['status']
    conclusion = ('Task4通过' if status == 'passed' and state.get('task4_qualified') else
                  '预算内未完成验证' if status == 'budget_exhausted' else '验证失败')
    report = [f'# Task4 300ms：{conclusion}', '',
              f"Source: `{campaign.commit}`; manifest digest: `{campaign.identity['manifest_sha256']}`.",
              f"Terminal status: `{status}`; finished: {state.get('finished_at')}; error: `{state.get('error','')}`.",
              '', 'The previous 250ms admission remains failed. Reused engineering evidence is not new validation.',
              'The reference is the original v5 parent; candidate v9 gains include both controller changes and learning.',
              'Candidate complete-act P95/max limits are100/300ms; reference limits250/480ms; search budget400ms.',
              'Historical parent replay is retained without retroactive certification.', '',
              '## Diagnostics', '']
    for seed, result in state.get('diagnostics', {}).items():
        audit = result['audit']
        report.append(f"- Logical seed{seed}: {audit['rounds']} rounds; timing `{audit['timing']}`; run `{result['run']}`.")
    report += ['', '## Development, confirmation and main validation', '']
    for arm, replicas in state.get('arms', {}).items():
        for seed, result in replicas.items():
            report.append(f"- Arm{arm} seed{seed}: selected rounds `{(result.get('selected') or {}).get('rounds')}`.")
            for point in result.get('history', []):
                failures=[key for key,value in point['gate_checks'].items() if not value['passed']]
                report.append(f"  - c{point['rounds']}: failed gates `{failures}`; checkpoint `{point['checkpoint']}`; SHA `{point['checkpoint_sha256']}`.")
    if state.get('main_validation'):
        report.append(f"\nMain candidate: `{state['main_validation']['checkpoint']}`; SHA `{state['main_validation']['checkpoint_sha256']}`.")
    report += ['', 'Full parent/child metrics and10,000-resample paired intervals are in `controller/evaluations`,',
               '`controller/comparisons`, per-arm seed records and `controller/result.json`.',
               'If a stage was not reached, it has no result; incomplete runs remain in the raw inventory.',
               'Exact taskset/Python commands are retained in `controller/logs/*.command.json`.',
               'All raw timing, replay and training state paths/SHA-256 are in `raw_artifact_inventory.json`.',
               'Diagnostic weights cannot continue formal training. No archive was repackaged or pushed.', '']
    (out/'report.md').write_text('\n'.join(report))
    # A terminal controller no longer checks the source HEAD. Never stage unrelated changes.
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Source changed; terminal evidence retained but automatic commit refused')
    subprocess.run(['git','add','-f',str(out.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',f'Record terminal Task4 300ms result: {status}'],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    campaign=Campaign(ROOT,args.manifest,args.resume)
    owner=campaign.directory/'background.json'
    value=dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
               source_commit=campaign.commit,manifest=str(campaign.manifest_path),
               result=str(campaign.path),status='running')
    write(owner,value)
    try:
        campaign.run()
    finally:
        value.update(status=campaign.state['status'],ended_at=datetime.now(timezone.utc).isoformat())
        write(owner,value)
    if campaign.state['status'] in TERMINAL:
        try:
            finish(campaign)
        except Exception as exc:
            value['archive_error']=repr(exc);write(owner,value)
            raise


if __name__=='__main__':
    main()

"""Remove only fully inventoried, verified worktrees and their accounted backup."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from experiments import consolidate_worktrees as c


def load(name):return json.loads((c.RECOVERY/name).read_text())


def gates():
    r=load('registry.json')
    assert len(c.manifests())==len(r['worktrees'])==20
    assert load('migration-complete.json')['complete']
    assert load('bundle-verification.json')['passed']
    assert load('validation/result.json')['status']=='passed'
    assert (c.RECOVERY/'backup-destinations.json').exists()
    assert c.git(c.ROOT,'branch','--show-current')=='main'
    assert c.git(c.ROOT,'branch','--format=%(refname:short)')=='main'
    frozen=json.loads((c.ROOT/'docs/research/branch-integration/frozen-files.json').read_text())
    for p,h in frozen.items():assert c.sha(c.ROOT/'agent_code'/p)==h
    assert c.sha(c.ROOT/'output/die-hardest-submission/final-project-agent-code.zip')=='0f64069f1ebdf7e7097780686514d123b245e6f71c1920e76ab60372303ac928'
    return r


def remove_worktrees():
    r=gates()
    journal=c.RECOVERY/'deletion-journal.json'
    state=json.loads(journal.read_text()) if journal.exists() else {'removed':[],'backup_removed':False}
    assert load('source-verification.json')['passed']
    live={w['worktree']:w for w in c.worktrees()}
    registered={w['worktree'] for w in r['worktrees']}
    assert not (set(live)-registered),'Unregistered worktree appeared'
    for row in r['worktrees']:
        if row['worktree']==str(c.ROOT):continue
        path=c.removal_guard(row['worktree'],r)
        if str(path) in state['removed']:
            assert not path.exists() and str(path) not in live
            continue
        assert str(path) in live and path.is_dir(),'Missing source without completed deletion record'
        busy=c.active_users([path])
        if busy:raise RuntimeError(f'Worktree still in use: {busy}')
        # Hashes were checked in the immediately preceding full audit. Re-inventory
        # sources again at the destructive boundary; any change blocks deletion.
        c.verify_one_source(row,check_hashes=False)
        c.verify_links()
        state['next']=str(path);c.write_json(journal,state)
        p=subprocess.run(['git','-C',str(c.ROOT),'worktree','remove','--force',str(path)],capture_output=True,text=True)
        with (c.RECOVERY/'deletion.log').open('a') as f:f.write(str(path)+'\n'+p.stdout+p.stderr+'\n')
        if p.returncode:raise RuntimeError(p.stderr)
        assert not path.exists()
        state['removed'].append(str(path));state['next']=None;c.write_json(journal,state)
        print('REMOVED',path.name,flush=True)
    assert len(c.worktrees())==1
    state['worktrees_completed_at']=time.time();c.write_json(journal,state)


def remove_backup():
    r=gates();journal=c.RECOVERY/'deletion-journal.json';state=load('deletion-journal.json')
    assert len(state['removed'])==19 and len(c.worktrees())==1
    c.verify_links()
    backup=Path(r['external_backup'])
    assert backup==Path('/export/data/sfan/branch-integration-backups/20260922T145758Z')
    assert not backup.is_symlink() and not backup.is_relative_to(c.ROOT)
    if state['backup_removed']:
        assert not backup.exists();return
    busy=c.active_users([backup])
    if busy:raise RuntimeError(f'Backup still in use: {busy}')
    locations=load('backup-destinations.json');old=json.loads((backup/'inventory.json').read_text())
    assert len(locations)==len(old['files'])
    for row in locations:
        p=c.ROOT/row['destination']
        # Account-backup freshly hashed every destination. Detect any subsequent
        # edits without repeatedly reading all 97GB at each deletion boundary.
        if c.identity(p)!=row['destination_identity'] or not p.is_file():
            raise ValueError(f'Backup destination changed: {p}')
    for p in backup.iterdir():
        if p.is_file():assert c.sha(p)==c.sha(c.RECOVERY/'previous-backup'/p.name)
    for p in (backup/'integration-records').rglob('*'):
        if p.is_file():assert c.sha(p)==c.sha(c.RECOVERY/'previous-backup/integration-records'/p.relative_to(backup/'integration-records'))
    # Verify the old bundle, including the original branch refs, before discarding
    # its redundant restore-check bare clone.
    subprocess.run(['git','-C',str(c.ROOT),'bundle','verify',str(c.RECOVERY/'previous-backup/all-branches.bundle')],check=True,capture_output=True)
    shutil.rmtree(backup)
    if not any(backup.parent.iterdir()):backup.parent.rmdir()
    state['backup_removed']=True;state['completed_at']=time.time();c.write_json(journal,state)
    print('REMOVED external backup',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['worktrees','backup'])
    parser.add_argument('--delete-verified',action='store_true',required=True)
    args=parser.parse_args()
    {'worktrees':remove_worktrees,'backup':remove_backup}[args.operation]()


if __name__=='__main__':main()

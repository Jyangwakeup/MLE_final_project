"""Remove only fully inventoried, verified worktrees and their accounted backup."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from experiments import consolidate_worktrees as c


def load(name):return json.loads((c.RECOVERY/name).read_text())


def require(condition, message="Deletion prerequisite failed"):
    if not condition: raise ValueError(message)


def gates():
    r=load('registry.json')
    require(len(c.manifests())==len(r['worktrees'])==20)
    require(load('migration-complete.json')['complete'])
    require(load('bundle-verification.json')['passed'])
    require(load('validation/result.json')['status']=='passed')
    require((c.RECOVERY/'backup-destinations.json').exists())
    require(load('external-backup-audit.json')['passed'])
    require(c.git(c.ROOT,'branch','--show-current')=='main')
    require(c.git(c.ROOT,'branch','--format=%(refname:short)')=='main')
    frozen=json.loads((c.ROOT/'docs/research/branch-integration/frozen-files.json').read_text())
    for p,h in frozen.items():require(c.sha(c.ROOT/'agent_code'/p)==h)
    require(c.sha(c.ROOT/'output/die-hardest-submission/final-project-agent-code.zip')=='0f64069f1ebdf7e7097780686514d123b245e6f71c1920e76ab60372303ac928')
    return r


def verify_saved_destinations(path):
    m=load('worktree-manifests/'+path.name+'.json')
    identities=load('destination-verifications/'+path.name+'.json')
    for f in m['files']:
        dest=Path(m['archive'])/f['path']
        require(dest.is_file() and not dest.is_symlink(),f'Archive missing: {dest}')
        require(c.identity(dest)==identities.get(f['path']),f'Archive changed: {dest}')
    c.verify_links()


def verify_backup_snapshot(backup, partial=False):
    audit=load('external-backup-audit.json');require(audit['passed'])
    expected={f['path']:f for f in audit['files']}
    files,links=c.backup_entries(backup)
    require(set(files)<=set(expected) if partial else set(files)==set(expected),'Backup file set changed')
    expected_links={l['path']:l for l in audit['links']}
    actual_links={l['path']:l for l in links}
    require(set(actual_links)<=set(expected_links) if partial else set(actual_links)==set(expected_links),'Backup link set changed')
    for rel in files:
        require(c.identity(backup/rel)==expected[rel]['identity'],f'Backup source changed: {rel}')
    for rel,l in actual_links.items():require(l['text']==expected_links[rel]['text'],f'Backup link changed: {rel}')
    for f in audit['files']:
        dest=c.ROOT/f['destination']
        require(dest.is_file() and not dest.is_symlink() and c.identity(dest)==f['destination_identity'],f'Backup destination changed: {dest}')
    for l in audit['links']:
        dest=c.ROOT/l['destination'];target=c.ROOT/l['target']
        require(dest.is_symlink() and dest.resolve()==target.resolve() and target.exists(),f'Saved link changed: {dest}')


def remove_worktrees():
    r=gates()
    journal=c.RECOVERY/'deletion-journal.json'
    state=json.loads(journal.read_text()) if journal.exists() else {'removed':[],'backup_removed':False}
    require(load('source-verification.json')['passed'])
    live={w['worktree']:w for w in c.worktrees()}
    registered={w['worktree'] for w in r['worktrees']}
    require(not (set(live)-registered),'Unregistered worktree appeared')
    for row in r['worktrees']:
        if row['worktree']==str(c.ROOT):continue
        path=c.removal_guard(row['worktree'],r)
        if str(path) in state['removed']:
            require(not path.exists() and str(path) not in live)
            continue
        if state.get('next')==str(path) and not path.exists() and str(path) not in live:
            verify_saved_destinations(path)
            state['removed'].append(str(path));state['next']=None;c.write_json(journal,state)
            continue
        require(str(path) in live and path.is_dir(),'Missing source without completed deletion record')
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
        require(not path.exists())
        state['removed'].append(str(path));state['next']=None;c.write_json(journal,state)
        print('REMOVED',path.name,flush=True)
    require(len(c.worktrees())==1)
    state['worktrees_completed_at']=time.time();c.write_json(journal,state)


def remove_backup():
    r=gates();journal=c.RECOVERY/'deletion-journal.json';state=load('deletion-journal.json')
    require(len(state['removed'])==19 and len(c.worktrees())==1)
    c.verify_links()
    backup=Path(r['external_backup'])
    require(backup==Path('/export/data/sfan/branch-integration-backups/20260922T145758Z'))
    require(not backup.is_symlink() and not backup.is_relative_to(c.ROOT))
    if state['backup_removed']:
        require(not backup.exists());return
    busy=c.active_users([backup])
    if busy:raise RuntimeError(f'Backup still in use: {busy}')
    verify_backup_snapshot(backup, partial=state.get('backup_removing',False))
    # Verify the old bundle, including the original branch refs, before discarding
    # its redundant restore-check bare clone.
    subprocess.run(['git','-C',str(c.ROOT),'bundle','verify',str(c.RECOVERY/'previous-backup/all-branches.bundle')],check=True,capture_output=True)
    state['backup_removing']=True;c.write_json(journal,state)
    if backup.exists():shutil.rmtree(backup)
    if backup.parent.exists() and not any(backup.parent.iterdir()):backup.parent.rmdir()
    state['backup_removed']=True;state['completed_at']=time.time();c.write_json(journal,state)
    print('REMOVED external backup',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['worktrees','backup'])
    parser.add_argument('--delete-verified',action='store_true',required=True)
    args=parser.parse_args()
    {'worktrees':remove_worktrees,'backup':remove_backup}[args.operation]()


if __name__=='__main__':main()

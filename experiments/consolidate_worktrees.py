"""Inventory and archive registered worktrees; destructive cleanup is a separate command.

Only paths in the persisted registry can be removed. Source files and manifests
are never rewritten. Cached environments are recorded, rather than copied.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archive'
RECOVERY = ARCHIVE / 'recovery'
CACHE_NAMES = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]


def checked_copy(source, target):
    """Idempotent, stable-source copying; a differing existing target is an error."""
    before = identity(source)
    digest = sha(source)
    if target.exists():
        if target.is_symlink() or sha(target) != digest:
            raise ValueError(f'Conflicting archive destination: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.archive-part')
        shutil.copy2(source, temporary)
        if sha(temporary) != digest or identity(source) != before:
            raise ValueError(f'Source changed during copy: {source}')
        temporary.replace(target)
    if identity(source) != before:
        raise ValueError(f'Source changed during verification: {source}')
    return {'bytes': before[0], 'sha256': digest, 'identity': before}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def worktrees():
    return [dict(line.split(' ', 1) if ' ' in line else (line, True)
                 for line in block.splitlines())
            for block in git(ROOT, 'worktree', 'list', '--porcelain').split('\n\n')]


def map_path(path, registry):
    """Map lexical absolute paths; longest root wins, without resolving links."""
    path = Path(os.path.abspath(path))
    for row in sorted(registry['worktrees'], key=lambda w: len(w['worktree']), reverse=True):
        old = Path(row['worktree'])
        if path.is_relative_to(old):
            return Path(row['archive']) / path.relative_to(old)
    raise ValueError(f'Unregistered link target: {path}')


def removal_guard(path, registry):
    path = Path(path)
    allowed = {Path(w['worktree']) for w in registry['worktrees'] if w['worktree'] != str(ROOT)}
    if path not in allowed or path == ROOT or path.is_symlink():
        raise ValueError(f'Refusing unregistered deletion: {path}')
    return path


def active_users(paths):
    """Inspect same-user process cwd, executable, open files and path arguments."""
    own = {os.getpid()}
    pid = os.getppid()
    while pid > 1:
        own.add(pid)
        try: pid = int(Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[1])
        except (OSError, ValueError): break
    hits = []
    def inside(s): return any(s == str(p) or s.startswith(str(p) + '/') for p in paths)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in own:
            continue
        try:
            if p.stat().st_uid != os.getuid(): continue
            values = []
            for name in ('cwd', 'exe'):
                try: values.append(os.readlink(p / name))
                except OSError: pass
            for f in (p / 'fd').iterdir():
                try: values.append(os.readlink(f))
                except OSError: pass
            values += [x.decode(errors='replace') for x in (p / 'cmdline').read_bytes().split(b'\0') if x.startswith(b'/')]
            matched = sorted(set(v for v in values if inside(v)))
            if matched: hits.append({'pid': int(p.name), 'paths': matched})
        except FileNotFoundError: pass
        except PermissionError:
            # PAM's non-dumpable session helper has no project workload.
            if (p / 'comm').read_text().strip() != '(sd-pam)':
                hits.append({'pid': int(p.name), 'error': 'same-user process cannot be inspected'})
    return hits


def environment_record(path):
    cfg = path / 'pyvenv.cfg'
    record = {'path': str(path), 'kind': 'rebuildable_environment', 'pyvenv_cfg': cfg.read_text()}
    packages = []
    for p in path.glob('lib/python*/site-packages/*.dist-info/METADATA'):
        name = version = None
        with p.open(errors='replace') as f:
            for line in f:
                if line.startswith('Name: '): name = line[6:].strip()
                if line.startswith('Version: '): version = line[9:].strip()
                if name and version: break
        packages.append({'name': name, 'version': version})
    record['packages'] = packages
    return record


def scan(root):
    tracked = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0'))
    dirty = set(subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--name-only', '-z']).decode().split('\0'))
    files, links, excluded = [], [], []
    for base, dirs, names in os.walk(root, followlinks=False):
        base = Path(base)
        for name in list(dirs):
            p = base / name
            if p.is_symlink():
                dirs.remove(name); names.append(name)
            elif name in CACHE_NAMES or (root == ROOT and p in (ARCHIVE, ROOT / '.scratch/archive-consolidation')):
                dirs.remove(name); excluded.append({'path': str(p), 'kind': 'cache_or_control'})
            elif (p / 'pyvenv.cfg').is_file():
                dirs.remove(name); excluded.append(environment_record(p))
        for name in names:
            p = base / name
            rel = str(p.relative_to(root))
            if name == '.git' or name.endswith('.pyc'):
                excluded.append({'path': str(p), 'kind': 'git_pointer_or_bytecode'}); continue
            if p.is_symlink():
                links.append({'path': rel, 'text': os.readlink(p), 'resolved': str(p.resolve()), 'exists': p.exists()})
            elif rel not in tracked or rel in dirty:
                if not p.is_file(): raise ValueError(f'Unsupported file kind: {p}')
                files.append(rel)
    return sorted(files), links, excluded, sorted(dirty - {''})


def init():
    registry_path = RECOVERY / 'registry.json'
    if registry_path.exists(): return json.loads(registry_path.read_text())
    rows = worktrees()
    if len(rows) != 20 or git(ROOT, 'branch', '--show-current') != 'main':
        raise ValueError('Expected the 20 registered worktrees and current main')
    for row in rows:
        root = Path(row['worktree'])
        row['archive'] = str(root if root == ROOT else ARCHIVE / 'worktrees' / root.name)
        row['status_before'] = git(root, 'status', '--porcelain')
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', row['HEAD'], 'main'], check=True)
    busy = active_users([Path(w['worktree']) for w in rows if w['worktree'] != str(ROOT)])
    # Preserve the initial occupancy evidence; pause affected roots in migrate.
    if any('error' in x for x in busy): raise RuntimeError(f'Uninspectable processes: {busy}')
    backup = Path((ROOT / '.scratch/branch-integration/backup-location.txt').read_text().strip())
    registry = {'worktrees': rows, 'external_backup': str(backup), 'main_at_start': git(ROOT, 'rev-parse', 'HEAD'),
                'initial_active_processes': busy, 'created': time.time(), 'filesystem_available_before': shutil.disk_usage(ROOT).free}
    write_json(registry_path, registry)
    return registry


def archive_one(row, registry):
    root, target = Path(row['worktree']), Path(row['archive'])
    output = RECOVERY / 'worktree-manifests' / (root.name + '.json')
    files, links, excluded, dirty = scan(root)
    records = []
    for rel in files:
        source, dest = root / rel, target / rel
        if root == ROOT:
            before = identity(source); h = sha(source)
            if identity(source) != before: raise ValueError(f'Changed source: {source}')
            record = {'bytes': before[0], 'sha256': h, 'identity': before}
        else: record = checked_copy(source, dest)
        records.append({'path': rel, **record})
    for link in links:
        if not link['exists']: raise ValueError(f'Broken source link: {root / link["path"]}')
        source = root / link['path']
        mapped = map_path(link['resolved'], registry)
        link['destination_target'] = str(mapped)
        if root != ROOT:
            dest = target / link['path']; dest.parent.mkdir(parents=True, exist_ok=True)
            text = os.path.relpath(mapped, dest.parent)
            if dest.is_symlink():
                if os.readlink(dest) != text: raise ValueError(f'Changed link destination: {dest}')
            elif dest.exists(): raise ValueError(f'File/link collision: {dest}')
            else: dest.symlink_to(text)
        elif mapped != Path(link['resolved']):
            raise ValueError(f'Main link requires separate migration: {source}')
    patch = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD'])
    patch_path = output.with_suffix('.patch');patch_path.parent.mkdir(parents=True, exist_ok=True);patch_path.write_bytes(patch)
    manifest = {'worktree': str(root), 'head': row['HEAD'], 'archive': str(target), 'files': records,
                'links': links, 'excluded': excluded, 'dirty_paths': dirty, 'verified': True}
    write_json(output, manifest)
    print('ARCHIVED', root.name, len(records), flush=True)
    return manifest


def manifests():
    return [json.loads(p.read_text()) for p in sorted((RECOVERY / 'worktree-manifests').glob('*.json'))]


def migrate():
    registry = init()
    # Preserve both current integrated history and the original branch bundle.
    bundle = RECOVERY / 'main.bundle'
    if not bundle.exists():
        subprocess.run(['git', '-C', str(ROOT), 'bundle', 'create', str(bundle), 'main'], check=True)
    busy = active_users([Path(w['worktree']) for w in registry['worktrees'] if w['worktree'] != str(ROOT)])
    if any('error' in x for x in busy): raise RuntimeError(f'Uninspectable processes: {busy}')
    pending = [w['worktree'] for w in registry['worktrees'] if any(p == w['worktree'] or p.startswith(w['worktree'] + '/') for x in busy for p in x.get('paths', []))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(archive_one, row, registry) for row in registry['worktrees'] if row['worktree'] not in pending and not (RECOVERY / 'worktree-manifests' / (Path(row['worktree']).name + '.json')).exists()]
        for f in as_completed(futures): f.result()
    if not pending: verify_links()
    write_json(RECOVERY / 'migration-complete.json', {'complete': not pending, 'pending': pending, 'worktrees': len(manifests())})


def verify_links():
    count = 0
    for m in manifests():
        root = Path(m['archive'])
        for row in m['links']:
            p = root / row['path']; target = Path(row['destination_target'])
            if not p.is_symlink() or not p.exists() or p.resolve() != target.resolve():
                raise ValueError(f'Unresolved archive link: {p} -> {target}')
            if not p.resolve().is_relative_to(ROOT): raise ValueError(f'External archive link: {p}')
            count += 1
    return count


def account_backup():
    registry = init(); backup = Path(registry['external_backup'])
    old = json.loads((backup / 'inventory.json').read_text())
    # Content destinations verified by migrate; root-relative identities remain explicit.
    records = {str(Path(m['worktree']) / f['path']): (Path(m['archive']) / f['path'], f['sha256'])
               for m in manifests() for f in m['files']}
    def account(f):
        original = f['path']; candidate = records.get(original)
        dest = candidate[0] if candidate else map_path(original, registry)
        # Main logs may have changed since inventory; never trust only a cached hash.
        if dest.is_file() and not dest.is_symlink() and dest.stat().st_size == f['bytes'] and sha(dest) == f['sha256']:
            kind = 'canonical'
        else:
            source = backup / f['backup']; dest = ARCHIVE / 'backup-only' / f['backup']
            r = checked_copy(source, dest)
            if r['sha256'] != f['sha256']: raise ValueError(f'Corrupt original backup: {source}')
            kind = 'backup-only'
        return {'original': original, 'backup': f['backup'], 'destination': str(dest.relative_to(ROOT)),
                'sha256': f['sha256'], 'bytes': f['bytes'], 'kind': kind,
                'destination_identity': identity(dest)}
    # Bounded batches avoid a future object per historical file.
    locations = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(old['files']), 1000):
            locations.extend(pool.map(account, old['files'][start:start+1000]))
            if start % 20000 == 0: print('ACCOUNTED', len(locations), flush=True)
    # Retain backup metadata and execution records, not a duplicate bare clone.
    for p in backup.iterdir():
        if p.is_file(): checked_copy(p, RECOVERY / 'previous-backup' / p.name)
    for p in (backup / 'integration-records').rglob('*'):
        if p.is_file(): checked_copy(p, RECOVERY / 'previous-backup/integration-records' / p.relative_to(backup / 'integration-records'))
    write_json(RECOVERY / 'backup-destinations.json', locations)
    print('BACKUP ACCOUNTED', len(locations), 'extra copies', sum(x['kind']=='backup-only' for x in locations), flush=True)


def backup_entries(backup):
    files=[];links=[]
    for base,dirs,names in os.walk(backup,followlinks=False):
        base=Path(base)
        for name in list(dirs):
            p=base/name
            if p==backup/'restore-check.git':dirs.remove(name)
            elif p.is_symlink():dirs.remove(name);names.append(name)
        for name in names:
            p=base/name;rel=str(p.relative_to(backup))
            if p.is_symlink():links.append({'path':rel,'text':os.readlink(p)})
            elif p.is_file():files.append(rel)
            else:raise ValueError(f'Unsupported backup file: {p}')
    return sorted(files),sorted(links,key=lambda x:x['path'])


def audit_external_backup():
    registry=init();backup=Path(registry['external_backup'])
    expected={x['backup']:x for x in json.loads((RECOVERY/'backup-destinations.json').read_text())}
    files,links=backup_entries(backup)
    def inspect(rel):
        source=backup/rel;before=identity(source);h=sha(source)
        if identity(source)!=before:raise ValueError(f'Backup changed: {source}')
        prior=expected.get(rel)
        if prior and prior['sha256']==h:
            dest=ROOT/prior['destination']
            if identity(dest)!=prior['destination_identity']:raise ValueError(f'Destination changed: {dest}')
        elif not rel.startswith('worktrees/'):
            dest=RECOVERY/'previous-backup'/rel
            checked_copy(source,dest)
        else:
            dest=ARCHIVE/'backup-only/current-backup'/rel;checked_copy(source,dest)
        return {'path':rel,'sha256':h,'bytes':before[0],'identity':before,
                'destination':str(dest.relative_to(ROOT)),'destination_identity':identity(dest)}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(files),1000):
            records.extend(pool.map(inspect,files[start:start+1000]))
            if start%20000==0:print('BACKUP AUDITED',len(records),flush=True)
    dest_by_source={str(backup/x['path']):ROOT/x['destination'] for x in records}
    for link in links:
        p=backup/link['path']
        target=Path(os.path.abspath(p.parent/link['text']))
        if str(target) in dest_by_source:mapped=dest_by_source[str(target)]
        else:mapped=map_path(p.resolve(),registry)
        if not mapped.exists() or not mapped.resolve().is_relative_to(ROOT):
            raise ValueError(f'Unpreserved backup link: {p}')
        dest=ARCHIVE/'backup-only/links'/link['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        text=os.path.relpath(mapped,dest.parent)
        if dest.is_symlink():
            if os.readlink(dest)!=text:raise ValueError(f'Conflicting saved link: {dest}')
        elif dest.exists():raise ValueError(f'Conflicting saved link: {dest}')
        else:dest.symlink_to(text)
        link.update(destination=str(dest.relative_to(ROOT)),target=str(mapped.relative_to(ROOT)))
    # All refs in the redundant restore-test clone must exist in the integrated history.
    clone=backup/'restore-check.git'
    if clone.exists():
        for tip in subprocess.check_output(['git','--git-dir='+str(clone),'for-each-ref','--format=%(objectname)'],text=True).splitlines():
            subprocess.run(['git','-C',str(ROOT),'cat-file','-e',tip],check=True)
    write_json(RECOVERY/'external-backup-audit.json',{'files':records,'links':links,'excluded_rebuildable_clone':str(clone),'passed':True})


def verify_one_source(row, check_hashes=True):
    root = Path(row['worktree'])
    m = json.loads((RECOVERY/'worktree-manifests'/(root.name+'.json')).read_text())
    if git(root,'rev-parse','HEAD') != m['head']: raise ValueError(f'HEAD changed: {root}')
    files,links,excluded,dirty = scan(root)
    if files != [f['path'] for f in m['files']] or dirty != m['dirty_paths']:
        raise ValueError(f'New or removed source files: {root}')
    if [(x['path'],x['text']) for x in links] != [(x['path'],x['text']) for x in m['links']]:
        raise ValueError(f'Source links changed: {root}')
    audit_path=RECOVERY/'destination-verifications'/(root.name+'.json')
    previous=json.loads(audit_path.read_text()) if not check_hashes else {}
    verified={}
    for f in m['files']:
        source=root/f['path'];dest=Path(m['archive'])/f['path']
        if identity(source)!=f['identity'] or not dest.is_file():
            raise ValueError(f'Source/archive changed: {source}')
        if dest.is_symlink():raise ValueError(f'Archive file replaced by link: {dest}')
        if check_hashes:
            before=identity(dest)
            if sha(dest)!=f['sha256'] or identity(dest)!=before:
                raise ValueError(f'Archive hash changed: {dest}')
            verified[f['path']]=before
        elif previous.get(f['path'])!=identity(dest):
            raise ValueError(f'Archive changed after hash audit: {dest}')
    patch = subprocess.check_output(['git','-C',str(root),'diff','--binary','HEAD'])
    if patch != (RECOVERY/'worktree-manifests'/(root.name+'.patch')).read_bytes():
        raise ValueError(f'Tracked changes appeared: {root}')
    if check_hashes:write_json(audit_path,verified)
    return {'root':str(root),'files':len(files),'verified':True,'at':time.time()}


def verify_sources():
    registry = init()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(verify_one_source,[r for r in registry['worktrees'] if r['worktree']!=str(ROOT)]))
    count=verify_links()
    write_json(RECOVERY/'source-verification.json',{'passed':True,'results':results,'links':count})
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['migrate','account-backup','verify-sources','verify-links','audit-backup'])
    args=parser.parse_args()
    {'migrate':migrate,'account-backup':account_backup,'verify-sources':verify_sources,'verify-links':verify_links,'audit-backup':audit_external_backup}[args.operation]()


if __name__ == '__main__': main()

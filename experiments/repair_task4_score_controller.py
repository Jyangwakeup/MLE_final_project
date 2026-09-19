"""One-shot, explicitly authorized controller-only revision of the interrupted campaign.
Does not permit ordinary training resume across source identities.
"""
import copy
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.task4_transfer import sha256,digest
from experiments.run import _source_hash
from experiments.task4_score_campaign import write
OLD='a301c5991447543426dc1f6041ac72c5abc902fc'
ALLOWED={'experiments/task4_score_campaign.py','tests/test_task4_score.py','experiments/repair_task4_score_controller.py'}

def main():
    base=ROOT/'runs/task4_score_20260920'
    with (base/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Source must be clean')
        new=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        changed=set(subprocess.check_output(['git','diff','--name-only',OLD,new],cwd=ROOT,text=True).splitlines())
        if not changed or not changed <= ALLOWED:raise ValueError('Not a controller-only revision')
        state=json.loads((base/'status.json').read_text());old_identity=copy.deepcopy(state['identity'])
        if (state['status']!='infrastructure_interrupted' or old_identity['source_commit']!=OLD
                or state['error']!='TypeError("dict() got multiple values for keyword argument \'arm\'")'):
            raise ValueError('Not the registered result-assembly interruption')
        if set(state.get('arms',{}))-{'C'} or state.get('trained_points'):raise ValueError('Formal training already started')
        manifest=json.loads((ROOT/'experiments/task4_score_manifest.json').read_text())
        if time.time()>=datetime.fromisoformat(manifest['started_at']).timestamp()+20*3600:raise ValueError('Original selection budget expired')
        if digest(manifest)!=old_identity['manifest_sha256']:raise ValueError('Manifest changed')
        for p,h in {**old_identity['config_hashes'],**manifest['algorithm_hashes'],**manifest['opponent_hashes']}.items():
            if sha256(ROOT/p)!=h:raise ValueError('Runtime/config changed: '+p)
        if sha256(ROOT/manifest['parent']['checkpoint'])!=old_identity['parent_sha256']:raise ValueError('Parent changed')
        runtime=_source_hash(manifest['agent'])
        proof=json.loads((base/'preflight/result.json').read_text())
        if json.loads((Path(proof['whole'])/'metadata.json').read_text())['source_hash']!=runtime:raise ValueError('Runtime differs from tested source')
        testlog=base/'restart_tests.log'
        if not testlog.read_text().rstrip().endswith('\nOK'):raise ValueError('Regression tests incomplete')
        caches=[]
        for path in sorted((base/'evaluations').glob('*.json')):
            value=json.loads(path.read_text())
            if value['request']['identity']!=old_identity:raise ValueError('Unexpected cache identity')
            if any(sha256(p)!=h for p,h in value['artifact_hashes'].items()):raise ValueError('Evaluation evidence changed')
            caches.append((path,value))
        backup=base/'revisions/result_assembly_v1'
        backup.mkdir(parents=True,exist_ok=False)
        originals={}
        for path in [base/'status.json',base/'controller.log',base/'report.md',*[p for p,v in caches]]:
            if not path.exists():continue
            relative=path.relative_to(base);dest=backup/'original'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest);originals[str(relative)]=sha256(path)
        new_identity=dict(old_identity,source_commit=new)
        receipt=dict(kind='explicit_controller_revision_v1',authorized_by='user restart request',old_identity=old_identity,
                     new_identity=new_identity,runtime_source_hash=runtime,changed_files=sorted(changed),
                     original_hashes=originals,regression_log_sha256=sha256(testlog),
                     budget_started_at=manifest['started_at'],created_at=datetime.now().astimezone().isoformat(),
                     semantics='Only result assembly changed; completed rollout artifacts and all thresholds remain unchanged; no old training state resumed')
        write(backup/'receipt.json',receipt)
        def rebind(value):
            if isinstance(value,dict):
                if value.get('identity')==old_identity:value['identity']=copy.deepcopy(new_identity)
                for child in value.values():rebind(child)
            elif isinstance(value,list):
                for child in value:rebind(child)
        for path,value in caches:
            rebind(value);value['reuse_provenance']=dict(original_source=OLD,revision_receipt=str(backup/'receipt.json'))
            write(path,value)
        rebind(state)
        state.setdefault('controller_revisions',[]).append(receipt)
        state['restart_validation']=dict(log=str(testlog),sha256=sha256(testlog),tests=10,passed=True)
        write(base/'status.json',state)
        print(json.dumps(dict(source=new,receipt=str(backup/'receipt.json'),reused_evaluations=len(caches))))

if __name__=='__main__':main()

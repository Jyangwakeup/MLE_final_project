"""Build Die Hardest only from the frozen, tested B33 submission archive."""
import argparse,hashlib,json,shutil,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD='double_dqn_continuous_v2_agent';NEW='die_hardest'
SOURCE_SHA='3fbda6106e810ef49aa2e8fcb8097f66cb8738628faed229177a8de1989e71d5'
WEIGHT_SHA='ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a'
def sha(data):return hashlib.sha256(data).hexdigest()
def build(source):
 data=source.read_bytes()
 if sha(data)!=SOURCE_SHA:raise ValueError('Source is not the frozen B33 archive')
 files={}
 with zipfile.ZipFile(source) as z:
  if z.testzip():raise ValueError('Corrupt source archive')
  for n in z.namelist():
   p=Path(n)
   if p.is_absolute() or '..' in p.parts or p.parts[0]!=OLD:raise ValueError(n)
   if n.endswith('/'):continue
   rel=p.relative_to(OLD);content=z.read(n)
   if rel.suffix=='.py':content=content.decode().replace('agent_code.'+OLD,'agent_code.'+NEW).encode()
   files[str(rel)]=content
 assert sha(files['final.pt'])==WEIGHT_SHA
 assert [p for p in files if p.endswith('.pt')]==['final.pt']
 assert files['requirements.txt']==b'numpy==2.2.6\ntorch==2.5.1\n'
 manifest=json.loads(files['SUBMISSION_MANIFEST.json']);manifest.update(agent=NEW,competition_name='Die Hardest',source_agent=OLD,source_archive_sha256=SOURCE_SHA,source_candidate='Task4 B seed33/c200',known_limitations=['27155 fixed-deadline versus rolling-admission inconsistency remains unresolved'],safety_fix_applied=False)
 files['SUBMISSION_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
 files['README.md']='''# Die Hardest

Frozen competition candidate: Task4 B seed33/c200, Double DQN, survival-mask-v9.
Tournament registration name: **Die Hardest**. Module and default framework display name: `die_hardest`.

Install `requirements.txt` in the course environment. Copy this entire directory into the original framework's `agent_code/`, then run from the framework root:

```bash
python main.py play --agents die_hardest rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --no-gui
```

Inference loads the included `final.pt` relative to this directory, uses CPU and one Torch thread, and needs no sibling project agents. Run without `--train`; the training module is retained from the verified source, but this frozen copy is intended for inference. For the verified default configuration, do not set `BOMBERMAN_*` overrides.

The original B33 checkpoint is unchanged. Renaming updates only package references and documentation. The algorithm, feature, reward and checkpoint contract identifiers are unchanged. Requirements retain the validated NumPy 2.2.6 and Torch 2.5.1 versions. See SUBMISSION_MANIFEST.json for provenance and hashes.

The original B33 completed a separate 1000-world evaluation versus three rule-based opponents: mean score 4.231, first including ties 50.2%, sole first 37.8%, survival 94.7%, no observed timeouts or safety alerts. These are results of the source candidate, not a new 1000-world evaluation after renaming.

Known limitation: the historical 27155 safety-contract counterexample remains unresolved. Packaging/renaming checks are not full safety certification. Docker and official tournament hardware were not verified here. No isolated repair prototype is included.
'''.encode()
 dest=ROOT/'agent_code'/NEW;out=ROOT/'output/die-hardest-submission';archive=out/'final-project-agent-code.zip'
 if dest.exists():
  existing={str(p.relative_to(dest)):p.read_bytes() for p in dest.rglob('*') if p.is_file()}
  if existing!=files:raise FileExistsError('Refusing to overwrite a different agent directory')
 else:
  with tempfile.TemporaryDirectory(dir=dest.parent,prefix='.die-hardest-') as tmp:
   stage=Path(tmp)/NEW;stage.mkdir()
   for rel,content in files.items():p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
   stage.rename(dest)
 out.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=out) as tmp:
  staged=Path(tmp)/archive.name
  with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for rel,content in sorted(files.items()):
    info=zipfile.ZipInfo(NEW+'/'+rel,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,content)
  if archive.exists():
   if staged.read_bytes()!=archive.read_bytes():raise FileExistsError('Refusing to overwrite different submission ZIP')
  else:shutil.copy2(staged,archive)
 (out/'SHA256SUMS').write_text(sha(archive.read_bytes())+'  '+archive.name+'\n')
 print(json.dumps({'archive':str(archive),'sha256':sha(archive.read_bytes()),'files':len(files),'checkpoint_sha256':WEIGHT_SHA},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);build(p.parse_args().source)

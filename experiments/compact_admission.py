"""One frozen admission only. Never starts a formal Task4 campaign."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run import _source_hash
from experiments.task3_retention_prefix import summarize_evaluation, _evaluation_directories
from experiments.task4_campaign import engineering_checks, audit_training
from experiments.compare_evaluations import compare_evaluations
AGENT='double_dqn_continuous_v2_agent'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timestamp(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def gates(task,summary):
    bad=engineering_checks({f'task{task}':summary})
    for key,limit in [('act_p95_seconds',.1),('act_max_seconds',.25),('invalid_action_rate',.01)]:
        if summary[key]>limit:bad.append(key)
    if task>1:
        if summary['suicide_rate']>.05:bad.append('suicide_rate')
        if summary['bomb_survival_rate']<.95:bad.append('bomb_survival_rate')
    if task>2 and summary['zero_bomb_round_rate']>.1:bad.append('zero_bomb_round_rate')
    return bad


class Admission:
    def __init__(self,manifest):
        self.manifest_path=Path(manifest).resolve();self.m=json.loads(self.manifest_path.read_text());self.commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.identity={'source_commit':self.commit,'manifest_sha256':sha(self.manifest_path),'runtime_sha256':_source_hash(AGENT)}
        self.check_identity()
        preflight=json.loads((ROOT/self.m['preflight_result']).read_text())
        if preflight['status']!='passed' or preflight['act_p95_ms']>100 or preflight['act_max_ms']>250 or preflight['historical_states']<160:
            raise ValueError('Incomplete preflight')
        import re
        log=(ROOT/self.m['unittest_log']).read_text()
        if not re.search(r'Ran \d+ tests in [^\n]+\n\nOK(?: \(skipped=\d+\))?\s*$',log):raise ValueError('Incomplete full unittest')
        if time.time()>=timestamp(self.m['freeze_deadline']):raise RuntimeError('Freeze cutoff elapsed')
        self.out=ROOT/'runs'/('compact_admission_'+self.commit[:7]);self.out.mkdir(parents=True,exist_ok=False)
        self.stop=threading.Event();self.state={**self.identity,'status':'running','formal_training_started':False,'task4_qualified':False,'jobs':{}}
        self.save()

    def check_identity(self):
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Frozen source must be clean')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=self.commit:raise ValueError('Frozen commit changed')
        if self.m['runtime_source_sha256']!=_source_hash(AGENT):raise ValueError('Runtime changed')
        if sha(self.manifest_path)!=self.identity['manifest_sha256']:raise ValueError('Manifest changed')
        for path,expected in self.m['identities'].items():
            if sha(ROOT/path)!=expected:raise ValueError('Identity changed: '+path)

    def save(self):
        temp=self.out/'result.tmp';temp.write_text(json.dumps(self.state,indent=2)+'\n');temp.replace(self.out/'result.json')

    def execute(self,label,args,cpu):
        self.check_identity()
        if time.time()>=timestamp(self.m['batch_deadline']):raise TimeoutError('No new batches after hour23')
        command=['taskset','-c',str(cpu),sys.executable,*args]
        (self.out/(label+'_command.json')).write_text(json.dumps(command,indent=2)+'\n')
        env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        with (self.out/(label+'.log')).open('w') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while p.poll() is None:
                    if self.stop.is_set() or time.time()>=timestamp(self.m['deadline']):raise TimeoutError('Peer failure or deadline')
                    time.sleep(.5)
                if p.returncode:raise RuntimeError(f'{label}: worker exit {p.returncode}')
            finally:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

    def evaluate(self,label,task,seeds,cpu,reference=False):
        name='compact_'+self.commit[:7]+'_'+label
        if (ROOT/'runs'/name).exists():raise FileExistsError(name)
        config=self.m['reference_config'] if reference else self.m['candidate_config']
        program=['-m','experiments.run'] if reference else ['experiments/compact_worker.py']
        self.execute(label,[*program,'--config',config,'--mode','evaluate','--device','cpu','--task',str(task),'--agent',AGENT,'--seeds',*map(str,seeds),'--n-rounds','1','--checkpoint',str(ROOT/self.m['parent']['checkpoint']),'--run-id',name,'--replay-policy','all'],cpu)
        target=ROOT/'runs'/name;s=summarize_evaluation(target,AGENT,seeds)
        episodes=[json.loads(l) for p in target.glob('*/episodes.jsonl') for l in p.read_text().splitlines()]
        if len(episodes)!=len(seeds):raise RuntimeError('Incomplete world set')
        s['mean_bombs']=sum(a['bombs'] for e in episodes for a in e['agents'] if a['name']==AGENT)/len(episodes)
        if not reference:
            if any(a['suicides'] for e in episodes for a in e['agents'] if a['name']==AGENT):raise RuntimeError('Self death requires explanation')
            from experiments.compact_audit import audit_run
            raw=audit_run(target,task)
            (self.out/(label+'_audit.json')).write_text(json.dumps(raw,indent=2)+'\n')
            bad=raw['failures']
            if bad:raise RuntimeError(label+': '+str(bad))
        return {'run':str(target),'name':name,'summary':s,'task':task,'reference':reference}

    def wave(self,specs):
        results={};cpus=self.m['cpus']
        with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
            # Distinct cores per concurrent wave; never oversubscribe.
            for start in range(0,len(specs),len(cpus)):
                pending={pool.submit(self.evaluate,*spec,cpu=cpu):spec[0] for spec,cpu in zip(specs[start:start+len(cpus)],cpus)}
                try:
                    for f in as_completed(pending):
                        label=pending[f];results[label]=f.result();self.state['jobs'][label]=results[label];self.save();print(label,'passed',flush=True)
                except BaseException:self.stop.set();raise
        return results

    def run(self):
        try:
            for case in self.m['regression_cases']:
                seed,task=case['seed'],case['task']
                label=f'regression_task{task}_{seed}';r=self.evaluate(label,task,[seed],self.m['cpus'][0]);self.state['jobs'][label]=r;self.save()
            # Recompute both baselines on the same source; no mismatched cache reuse.
            ret={}
            for task in (1,2,3):
                for ref in (True,False):
                    label=f'retention_{task}_{"parent" if ref else "child"}'
                    ret[label]=self.evaluate(label,task,list(range(24000,24060)),self.m['cpus'][0],reference=ref)
                    self.state['jobs'][label]=ret[label];self.save()
                parent=ret[f'retention_{task}_parent'];child=ret[f'retention_{task}_child']
                metrics={1:['mean_score'],2:['mean_coins','mean_crates'],3:['mean_score','mean_coins','mean_crates']}[task]
                checks={k:child['summary'][k]/parent['summary'][k] if parent['summary'][k] else 1.0 for k in metrics}
                self.state.setdefault('retention',{})[str(task)]=checks
                self.state.setdefault('bootstrap',{})[str(task)]=compare_evaluations(_evaluation_directories(ROOT/'runs',child['name'],range(24000,24060)),_evaluation_directories(ROOT/'runs',parent['name'],range(24000,24060)),self.out/'comparisons'/str(task),bootstrap_samples=10000)
                self.save()
                if any(v<.9 for v in checks.values()):raise RuntimeError('Retention gate failed')
            self.wave([(f'fresh_task{task}',task,self.m['engineering_seeds']) for task in (1,2,3,4)])
            for logical in (22,11,33):
                seed=self.m['diagnostic_training_seeds'][str(logical)];label='diagnostic_'+str(logical);name='compact_'+self.commit[:7]+'_'+label
                self.execute(label,['experiments/compact_worker.py','--config',self.m['candidate_config'],'--mode','train','--device','cpu','--task','4','--agent',AGENT,'--seed',str(seed),'--n-rounds','20','--transfer-task4-from-checkpoint',str(ROOT/self.m['parent']['checkpoint']),'--run-id',name,'--replay-policy','all'],self.m['cpus'][0])
                target=ROOT/'runs'/name;audit=audit_training(target,AGENT)
                if audit['rounds']!=20:raise RuntimeError('Incomplete diagnostic')
                # The pooled metrics path accepts per-seed evaluation dirs; train
                # has one directory and multiple rounds, summarized directly below.
                from experiments.compact_audit import audit_run
                result=audit_run(target,task=4)
                if result['failures']:raise RuntimeError(str(result['failures']))
                self.state['jobs'][label]=result;self.save()
            self.state['status']='admission_passed'
        except TimeoutError as exc:self.state.update(status='incomplete_budget',error=repr(exc))
        except Exception as exc:self.state.update(status='admission_failed',error=repr(exc))
        finally:self.save()
        print(self.state['status'],self.state.get('error',''),flush=True)
        return 0 if self.state['status']=='admission_passed' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();raise SystemExit(Admission(a.manifest).run())

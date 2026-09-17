from pathlib import Path
import sys,json,hashlib,subprocess
root=Path('/export/data/sfan/MLE_final_project_safety');sys.path.insert(0,str(root))
from experiments.analyze import analyze_runs
from experiments.task3_retention_prefix import summarize_evaluation
from experiments.task4_campaign import engineering_checks
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert commit.startswith('0baf79d')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
original=root/'runs/safety_stress_4aa3812';audit=json.loads((original/'independent_audit.json').read_text());assert not audit['failures']
out=original/('reanalysis_'+commit[:7]);out.mkdir(exist_ok=False)
r={'execution_source_commit':audit['source_commit'],'analysis_source_commit':commit,'reason':'400 episodes completed; chart generation failed after execution','original_controller_status':'stopped_failure','tasks':{},'task4_qualified':False}
for task in (1,2,3,4):
 directory=root/'runs'/f'safety_stress_task{task}_4aa3812';files=sorted(directory.rglob('episodes.jsonl'));assert len(files)==100
 for relative,expected in audit['tasks'][str(task)]['raw_sha256'].items():assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected
 mirror=out/directory.name;summary_out=mirror/(directory.name+'_summary')
 analyze_runs([p.parent for p in files],summary_out)
 s=summarize_evaluation(mirror,'double_dqn_continuous_v2_agent',range(24100,24200))
 episodes=[json.loads(line) for p in files for line in p.read_text().splitlines()]
 s['mean_bombs']=sum(a['bombs'] for e in episodes for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')/100
 independent=audit['tasks'][str(task)];assert s['act_p95_seconds']==independent['pooled_p95'];assert s['act_max_seconds']==independent['max']
 failures=engineering_checks({f'task{task}':s})
 if s['invalid_action_rate']>.01:failures.append('invalid_action_rate')
 if task>1 and (s['suicide_rate']>.05 or s['bomb_survival_rate']<.95):failures.append('bomb_survival_or_suicide')
 if task>2 and s['zero_bomb_round_rate']>.1:failures.append('zero_bomb_round_rate')
 r['tasks'][str(task)]={'summary':s,'failures':failures,'raw_run':str(directory),'analysis_output':str(summary_out)}
 print(task,s,failures,flush=True)
r['status']='passed' if not any(t['failures'] for t in r['tasks'].values()) else 'gate_failure'
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print('STATUS',r['status'],flush=True)

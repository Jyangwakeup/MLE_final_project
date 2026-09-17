from pathlib import Path
import json,math,hashlib,time
root=Path('/export/data/sfan/MLE_final_project_safety');path=root/'runs/safety_stress_4aa3812/result.json'
while True:
 state=json.loads(path.read_text())
 if state['status']!='running':break
 time.sleep(2)
out=root/'runs/safety_stress_4aa3812/independent_audit.json';assert not out.exists()
report={'source_commit':state['source_commit'],'controller_status':state['status'],'percentile_definition':'nearest-rank ceil(0.95*N)-1','tasks':{},'failures':[],'task4_qualified':False}
for task in (1,2,3,4):
 d=root/'runs'/f'safety_stress_task{task}_4aa3812';times=[];episodes=[];hashes={};issues=[];placements=0;pending=0;deaths=[]
 for p in sorted(d.rglob('episodes.jsonl')):
  es=[json.loads(l) for l in p.read_text().splitlines()];episodes+=es
  for e in es:
   a=next(a for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')
   if a['dead']:deaths.append(dict(world=e['environment_seed'],agent=a,trace_files=list(map(str,p.parent.glob('task4_death_states*')))))
   if a['suicides']:issues.append(dict(world=e['environment_seed'],issue='self_death'))
 for p in sorted(d.rglob('timing.jsonl')):
  for l in p.open():
   row=json.loads(l)
   if row['agent_name']!='double_dqn_continuous_v2_agent':continue
   if row.get('think_time') is not None:times.append(row['think_time'])
   s=row.get('safety',{})
   if row['action']=='BOMB':
    placements+=1;n=s['opponent_scenario_counts'][5];passed=s['opponent_passing_counts'][5]
    if not(n>0 and n==passed and not s['robust_search_timed_out'] and not s['v1_to_physical_fallback']):issues.append(dict(file=str(p),step=row.get('step'),issue='unproved_placement'))
   if s['own_bomb_pending']:
    pending+=1
    if s['v1_to_physical_fallback']:issues.append(dict(file=str(p),step=row.get('step'),issue='pending_physical_fallback'))
 for filename in ('metadata.json','episodes.jsonl','timing.jsonl'):
  for p in d.rglob(filename):hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 ordered=sorted(times);p95=ordered[math.ceil(.95*len(ordered))-1] if times else None
 if len(episodes)!=100:issues.append({'issue':'incomplete_episodes','count':len(episodes)})
 if p95 is None or p95>.25 or max(times)>.48:issues.append({'issue':'pooled_timing_gate'})
 report['tasks'][str(task)]=dict(episodes=len(episodes),act_count=len(times),pooled_p95=p95,max=max(times) if times else None,placements=placements,pending_decisions=pending,issues=issues,deaths=deaths,raw_sha256=hashes)
 report['failures']+= [{'task':task,**issue} for issue in issues]
report['status']='passed' if state['status']=='passed' and not report['failures'] else 'failed_or_incomplete'
out.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['failures'],flush=True)
for task,r in report['tasks'].items():print(task,{k:r[k] for k in ('episodes','act_count','pooled_p95','max','placements')},flush=True)

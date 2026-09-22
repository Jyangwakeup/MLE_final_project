import json,math
from pathlib import Path
root=Path('/export/data/sfan/MLE_final_project_safety')
base=root/'runs/safety_stress_task4_0dc8dba'
result={'completed':0,'actions':0,'placements':0,'pending_actions':0,'issues':[],'deaths':[]};times=[]
for ep in sorted(base.glob('*/episodes.jsonl')):
 text=ep.read_text()
 if not text.strip() or not text.endswith('\n'):continue
 episodes=[json.loads(l) for l in text.splitlines()]
 rows=[json.loads(l) for l in (ep.parent/'timing.jsonl').read_text().splitlines()]
 rows=[r for r in rows if r['agent_name']=='double_dqn_continuous_v2_agent']
 origin=None
 for r in rows:
  s=r['safety'];step=r['step'];times.append(r['think_time']*1000)
  if s['own_bomb_pending']:
   result['pending_actions']+=1
   if origin is None or s['own_bomb_placed_step']!=origin or not 0<step-origin<7:result['issues'].append([ep.parent.name,step,'clock'])
   if s['v1_to_physical_fallback']:result['issues'].append([ep.parent.name,step,'pending_fallback'])
  else:origin=None
  if r['action']=='BOMB':
   origin=step;result['placements']+=1
   if not s['decision_mask'][5] or s['robust_to_v1_fallback'] or s['v1_to_physical_fallback']:result['issues'].append([ep.parent.name,step,'placement'])
  for k in ('robust_guarantee_loss','robust_search_timed_out','avoidable_escape_collapse'):
   if s[k]:result['issues'].append([ep.parent.name,step,k])
  if r['timed_out'] or r['skipped']:result['issues'].append([ep.parent.name,step,'framework_timing'])
 for e in episodes:
  result['completed']+=1;a=next(a for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')
  if a['dead']:
   last=rows[-1];result['deaths'].append({'seed':e['seed'],'causes':a['death_causes'],'step':a['death_step'],'self_death':a['killed_by_self'],'last_action':last['action'],'pending_before':last['safety']['own_bomb_pending'],'physical_fallback':last['safety']['v1_to_physical_fallback']})
times.sort();result.update(actions=len(times),p95_ms=times[math.ceil(.95*len(times))-1] if times else None,max_ms=max(times,default=None))
(root/'runs/safety_stress_0dc8dba/raw_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""Read-only audit of completed fresh engineering games; no qualification claim."""
import json,math
from pathlib import Path
ROOT=Path('/export/data/sfan/MLE_final_project_safety_movement')
ACTIONS=('UP','RIGHT','DOWN','LEFT','WAIT','BOMB')
report={}
for task in (4,):
 result={'completed':0,'actions':0,'placements':0,'pending_actions':0,'issues':[],'deaths':[]};times=[]
 for ep in sorted((ROOT/'runs/gc_context_7efa03b_worlds').glob('*/episodes.jsonl')):
  text=ep.read_text()
  if not text.strip() or not text.endswith('\n'):continue
  episodes=[json.loads(l) for l in text.splitlines()]
  rows=[json.loads(l) for l in (ep.parent/'timing.jsonl').read_text().splitlines()]
  rows=[r for r in rows if r['agent_name']=='double_dqn_continuous_v2_agent']
  origin=None;last_proof=None
  for r in rows:
   s=r['safety'];step=r['step'];i=ACTIONS.index(r['action']);times.append(r['think_time']*1000)
   def issue(reason):result['issues'].append([ep.parent.name,step,reason])
   if not s['decision_mask'][i]:issue('action_mask_mismatch')
   if not s['opponent_to_v3_fallback'] and not s['v1_to_physical_fallback']:
    if not s['opponent_passing_counts'][i]:issue('unproved_selection')
    end=s['own_bomb_placed_step']+7 if s['own_bomb_pending'] else step+7
    last_proof={'step':step,'action':r['action'],'pending':s['own_bomb_pending'],'endpoint_exclusive':end}
   if s['own_bomb_pending']:
    result['pending_actions']+=1
    if origin is None or s['own_bomb_placed_step']!=origin or not 0<step-origin<7:issue('clock')
    if s['v1_to_physical_fallback']:issue('pending_fallback')
   else:
    if origin is not None and step-origin<7:issue('premature_responsibility_release')
    origin=None
   if r['action']=='BOMB':
    origin=step;result['placements']+=1
    if task==1:issue('task1_bomb')
    if s['robust_to_v1_fallback'] or s['v1_to_physical_fallback'] or not s['opponent_passing_counts'][i]:issue('uncertified_bomb')
   for key in ('robust_guarantee_loss','robust_search_timed_out','avoidable_escape_collapse'):
    if s[key]:issue(key)
   if r['timed_out'] or r['skipped']:issue('framework_timing')
  for e in episodes:
   result['completed']+=1;a=next(a for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')
   if a['dead']:
    result['deaths'].append({'seed':e['seed'],'death_step':a['death_step'],'causes':a['death_causes'],'self_death':a['killed_by_self'],'pending_before':rows[-1]['safety']['own_bomb_pending'],'last_proof':last_proof,'death_before_endpoint':last_proof is not None and a['death_step']<last_proof['endpoint_exclusive']})
 times.sort();result.update(actions=len(times),p95_ms=times[math.ceil(.95*len(times))-1] if times else None,max_ms=max(times,default=None));report[f'task{task}']=result
(ROOT/'runs/gc_context_7efa03b/raw_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

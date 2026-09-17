import json, math
from pathlib import Path
ROOT=Path('/export/data/sfan/MLE_final_project_safety')
result={}
for task in (1,2,3):
 for version in ('v5','v8'):
  base=ROOT/'runs'/f'safety_retention_{version}_task{task}_5ff8d4c'
  entries=[];issues=[];times=[];bombs=0;pending=0
  for ep in sorted(base.glob('*/episodes.jsonl')):
   text=ep.read_text()
   if not text.endswith('\n') or not text.strip():continue
   episodes=[json.loads(l) for l in text.splitlines()]
   rows=[json.loads(l) for l in (ep.parent/'timing.jsonl').read_text().splitlines()]
   rows=[r for r in rows if r['agent_name']=='double_dqn_continuous_v2_agent']
   origin=None
   for r in rows:
    times.append(r['think_time']*1000)
    if version!='v8':continue
    s=r['safety'];step=r['step']
    if s['own_bomb_pending']:
     pending+=1
     if origin is None or s['own_bomb_placed_step']!=origin or not 0<step-origin<7:issues.append([ep.parent.name,step,'clock'])
     if s['v1_to_physical_fallback']:issues.append([ep.parent.name,step,'pending_physical_fallback'])
    else:origin=None
    if r['action']=='BOMB':
     origin=step;bombs+=1
     if not s['decision_mask'][5] or s['robust_to_v1_fallback'] or s['v1_to_physical_fallback']:issues.append([ep.parent.name,step,'placement'])
    for key in ('robust_guarantee_loss','robust_search_timed_out','avoidable_escape_collapse'):
     if s[key]:issues.append([ep.parent.name,step,key])
    if r['timed_out'] or r['skipped']:issues.append([ep.parent.name,step,'framework_timing'])
   for e in episodes:
    a=next(a for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')
    entries.append({'seed':e['seed'],**{k:a[k] for k in ('score','coins','crates','suicides','bombs','bombs_survived','bombs_resolved')}})
  times.sort()
  result[f'{version}_task{task}']={'completed':len(entries),'actions':len(times),'p95_ms':times[math.ceil(.95*len(times))-1] if times else None,'max_ms':max(times,default=None),'bombs_audited':bombs,'pending_actions':pending,'issues':issues,'episodes':entries}
path=ROOT/'runs/certified_placement/v8_partial_audit_5ff8d4c.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{x:y for x,y in v.items() if x!='episodes'} for k,v in result.items()},indent=2))

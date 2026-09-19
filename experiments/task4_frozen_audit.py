"""Audit frozen neural seats separately from official rule delegates."""
import json
from pathlib import Path
import numpy as np
from experiments.task4_worker import ACTIONS


def audit_opponents(directory):
    root=Path(directory)
    files=[root/'episodes.jsonl'] if (root/'episodes.jsonl').exists() else sorted(root.glob('*/episodes.jsonl'))
    groups={};failures=[]
    for ep in files:
        schedules={row['round']:row for row in map(json.loads,(ep.parent/'opponent_schedule.jsonl').read_text().splitlines())}
        records=list(map(json.loads,(ep.parent/'timing.jsonl').read_text().splitlines()))
        episodes=list(map(json.loads,ep.read_text().splitlines()))
        for episode in episodes:
            number=episode['round_index'];schedule=schedules[number]
            for seat in schedule['seats']:
                if seat['model_id']=='rule':continue
                name=seat['agent'];rows=[r for r in records if r['round_index']==number and r['agent_name']==name]
                if not rows:failures.append(name+':missing_actions');continue
                key=name+':'+seat['model_id'];group=groups.setdefault(key,dict(times=[],episodes=0,bombs=0,resolved=0,survived=0,suicides=0,invalid=0,score=0,kills=0,first=0))
                agent=next(a for a in episode['agents'] if a['name']==name)
                for field,source in [('bombs','bombs'),('resolved','bombs_resolved'),('survived','bombs_survived'),('suicides','suicides'),('invalid','invalid'),('score','score'),('kills','kills')]:group[field]+=agent[source]
                group['first']+=agent['score']==max(a['score'] for a in episode['agents'])
                group['episodes']+=1;origin=None
                if agent['suicides']:failures.append(key+':unexplained_self_death')
                for row in rows:
                    d=row.get('safety');step=row['step'];action=row['action']
                    group['times'].append(row['think_time'])
                    if row['timed_out'] or row['skipped']:failures.append(key+':timeout_or_skip')
                    if not d:failures.append(key+':missing_diagnostic');continue
                    index=ACTIONS.index(action)
                    if d['selected_action']!=action or not d['decision_mask'][index] or not d['physical_mask'][index]:failures.append(key+':mask_mismatch')
                    for field in ('robust_search_timed_out','robust_guarantee_loss','avoidable_escape_collapse'):
                        if d[field]:failures.append(key+':'+field)
                    if not d['opponent_to_v3_fallback'] and not d['v1_to_physical_fallback'] and not d['opponent_passing_counts'][index]:failures.append(key+':unproved_selection')
                    if d['own_bomb_pending']:
                        if origin is None or d['own_bomb_placed_step']!=origin or not 0<step-origin<7:failures.append(key+':responsibility_clock')
                    else:
                        if origin is not None and step-origin<7:failures.append(key+':premature_release')
                        origin=None
                    if action=='BOMB':
                        origin=step
                        if not d['opponent_passing_counts'][index]:failures.append(key+':uncertified_bomb')
    summaries={}
    for key,group in groups.items():
        times=group.pop('times');n=group['episodes']
        summary=dict(group,actions=len(times),act_p95_seconds=float(np.percentile(times,95)),act_max_seconds=max(times),
            suicide_rate=group['suicides']/n,bomb_survival_rate=group['survived']/group['resolved'] if group['resolved'] else None,
            invalid_action_rate=group['invalid']/len(times),mean_score=group['score']/n,mean_kills=group['kills']/n,first_place_rate=group['first']/n)
        summaries[key]=summary
    return dict(agents=summaries,failures=sorted(set(failures)))

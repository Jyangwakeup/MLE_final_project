"""Raw observation audit for frozen admission and 20-round diagnostic training."""
import json
from pathlib import Path
import numpy as np
AGENT='double_dqn_continuous_v2_agent';ACTIONS=('UP','RIGHT','DOWN','LEFT','WAIT','BOMB')

def audit_run(directory,task,policy=None):
    root=Path(directory);episodes=[];rows=[];issues=[];deaths=[]
    files=[root/'episodes.jsonl'] if (root/'episodes.jsonl').exists() else sorted(root.glob('*/episodes.jsonl'))
    for ep in files:
        current=[json.loads(l) for l in ep.read_text().splitlines()];episodes.extend(current)
        records=[json.loads(l) for l in (ep.parent/'timing.jsonl').read_text().splitlines()]
        own=[r for r in records if r['agent_name']==AGENT];rows.extend(own)
        origin=None;previous_round=None
        for r in own:
            d=r['safety'];step=r['step'];round_id=r['round_index'];i=ACTIONS.index(r['action'])
            if round_id!=previous_round:origin=None;previous_round=round_id
            if not d['decision_mask'][i]:issues.append('mask_mismatch')
            if not d['opponent_to_v3_fallback'] and not d['v1_to_physical_fallback'] and not d['opponent_passing_counts'][i]:issues.append('unproved_selection')
            if d['own_bomb_pending']:
                if origin is None or d['own_bomb_placed_step']!=origin or not 0<step-origin<7:issues.append('responsibility_clock')
            else:
                if origin is not None and step-origin<7:issues.append('premature_release')
                origin=None
            if r['action']=='BOMB':
                origin=step
                if not d['opponent_passing_counts'][i]:issues.append('uncertified_bomb')
                if task==1:issues.append('task1_bomb')
        for e in current:
            a=next(a for a in e['agents'] if a['name']==AGENT)
            if a['dead']:deaths.append({'seed':e['seed'],'round_index':e['round_index'],'death_step':a['death_step'],'self_death':a['killed_by_self'],'causes':a['death_causes']})
            if a['suicides']:issues.append('unexplained_self_death')
    agents=[next(a for a in e['agents'] if a['name']==AGENT) for e in episodes];n=len(agents)
    if not n or not rows:raise ValueError('Empty admission run')
    times=[r['think_time'] for r in rows];resolved=sum(a['bombs_resolved'] for a in agents)
    summary={'episodes':n,'mean_bombs':sum(a['bombs'] for a in agents)/n,'suicide_rate':sum(a['suicides'] for a in agents)/n,
             'bomb_survival_rate':sum(a['bombs_survived'] for a in agents)/resolved if resolved else 0.,'zero_bomb_round_rate':sum(a['bombs']==0 for a in agents)/n,
             'invalid_action_rate':sum(a['invalid'] for a in agents)/len(rows),'act_p95_seconds':float(np.percentile(times,95)),'act_max_seconds':max(times),
             'act_timeouts':sum(r['timed_out'] for r in rows),'act_skipped':sum(r['skipped'] for r in rows)}
    for plural,singular in [('robust_search_timeouts','robust_search_timed_out'),('robust_guarantee_losses','robust_guarantee_loss'),('avoidable_escape_collapses','avoidable_escape_collapse')]:summary[plural]=sum(r['safety'][singular] for r in rows)
    from experiments.compact_admission import gates
    failures=gates(task,summary) if policy is None else policy_gates(task,summary,policy)
    return {'run':str(root),'summary':summary,'actions':len(rows),'deaths':deaths,'failures':sorted(set(issues+failures))}


def policy_gates(task, summary, policy):
    from experiments.task4_campaign import engineering_checks
    bad=engineering_checks({f'task{task}':summary},policy)
    if summary['invalid_action_rate']>.01:bad.append('invalid_action_rate')
    if task>1:
        if summary['suicide_rate']>.05:bad.append('suicide_rate')
        if summary['bomb_survival_rate']<.95:bad.append('bomb_survival_rate')
    if task>2 and summary['zero_bomb_round_rate']>.1:bad.append('zero_bomb_round_rate')
    return bad

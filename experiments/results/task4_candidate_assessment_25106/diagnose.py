"""Reproduce world25106 and isolate placement proof duration without edits to runtime."""
import json,pickle,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()))
from agent_code.team_agent.safety import safety_decision
from experiments.task4_transfer import PARENT_SAFETY,TARGET_SAFETY
from agent_code.team_agent.controllable_survival import controllable_survival_actions
f=next(pathlib.Path('runs').glob('*confirmation_parent_t4*/*s25106/task4_failure_states.pkl'))
rows=pickle.loads(f.read_bytes());out=[]
for row in rows[-4:]:
 s=row['state'];d=row['safety'];own={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']}
 item={'step':s['step'],'observed_action':row['action']}
 for label,spec in [('v5',PARENT_SAFETY),('v9',TARGET_SAFETY)]:
  a=safety_decision(s,d['physical_mask'],allow_bomb=True,exploring=False,safety_spec=spec,own_bomb_pending=d['own_bomb_pending'],own_bomb_state=own)
  item[label]={'mask':a.mask.tolist(),'physical_fallback':bool(a.physical_fallback),'guarantee_loss':bool(a.robust_guarantee_loss)}
 if s['step']==71: assert item['v5']['mask'][5] and not item['v9']['mask'][5]
 if s['step']==74: assert item['v5']['physical_fallback']
 out.append(item)
ablation=[]
for duration in (6,7):
 for rearming in (False,True):
  a=controllable_survival_actions(rows[-4]['state'],('BOMB',),remaining_steps=duration,budget_ms=400,consider_opponent_rearming=rearming)
  assert not a.timed_out and bool(a.proven_actions)==(duration==6)
  ablation.append({'duration':duration,'rearming':rearming,'proven':a.proven_actions})
print(json.dumps({'fixture':str(f),'reproduction':out,'ablation':ablation},indent=2))

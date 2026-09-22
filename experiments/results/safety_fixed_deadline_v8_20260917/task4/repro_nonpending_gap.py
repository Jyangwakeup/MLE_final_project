"""Expected-red diagnostic: known unsafe movement is admitted outside own responsibility."""
import json,pickle
from pathlib import Path
import numpy as np
from agent_code.team_agent.safety import safety_decision
from agent_code.team_agent.controllable_survival import controllable_survival_actions
root=Path('/export/data/sfan/MLE_final_project_safety')
rows=pickle.loads((root/'runs/safety_stress_task4_0dc8dba/safety_stress_task4_0dc8dba_s24025/task4_death_states_round0001.pkl').read_bytes())
r=next(r for r in rows if r['state']['step']==189);s=r['safety']
spec=json.loads((root/'experiments/configs/safety_fixed_deadline_v8.json').read_text())['safety']
actual=safety_decision(r['state'],np.asarray(s['physical_mask']),spec,allow_bomb=True,exploring=False,own_bomb_pending=False,own_bomb_state={'pending':False,'timer':None,'placed_step':None})
proof=controllable_survival_actions(r['state'],('UP','RIGHT','DOWN','LEFT','WAIT'),remaining_steps=7,budget_ms=60000,consider_opponent_rearming=True)
print('v8_mask',actual.mask.tolist(),'actual_action',r['action'],'proven_actions',proof.proven_actions,flush=True)
assert not proof.timed_out
assert 'UP' in proof.proven_actions and 'RIGHT' not in proof.proven_actions
assert not actual.mask[1], 'RIGHT remains admitted despite available robustly proven alternatives'

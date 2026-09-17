from pathlib import Path
import pickle,sys,json,copy
sys.path.insert(0,'/export/data/sfan/MLE_final_project_safety')
from agent_code.team_agent.controllable_survival import controllable_survival_actions,danger_interval_steps
from agent_code.team_agent.opponent_transitions import enumerate_opponent_transition_scenarios,canonical_state_key
root=Path('/export/data/sfan/MLE_final_project_safety');rs=pickle.loads((root/'experiments/results/safety_round2_failure_20260917/failure_states.pkl').read_bytes());by={r['state']['step']:r for r in rs};results=[]
for step,actions in ((181,('BOMB',)),(182,('UP','DOWN','WAIT')),(183,('RIGHT','WAIT'))):
 state=by[step]['state'];d=by[step]['safety'];remaining=danger_interval_steps({'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer']},placing_bomb=step==181)
 for armed in (False,True):
  s=copy.deepcopy(state)
  if armed:s['others']=[(*a[:2],True,a[3]) for a in s['others']]
  for horizon in sorted({remaining,6,7,8}):
   r=controllable_survival_actions(s,actions,remaining_steps=horizon,budget_ms=60000)
   row=dict(step=step,force_armed=armed,horizon=horizon,proven=r.proven_actions,scenarios=r.scenarios_evaluated,failed_profile=r.first_failing_profile,failed_order=r.first_failing_order,timed_out=r.timed_out);results.append(row);print(row,flush=True)
for step in (181,182,183):
 current=by[step];actual=by[step+1]['state'];scenarios=enumerate_opponent_transition_scenarios(current['state'],current['action'],progress_world=True)
 matches=[dict(profile=s.action_profile,order=s.execution_order) for s in scenarios if canonical_state_key(s.game_state)==canonical_state_key(actual)]
 print('TRANSITION',step,'matched',matches,flush=True);results.append(dict(step=step,actual_transition_matches=matches))
(root/'experiments/results/safety_round2_failure_20260917/probes.json').write_text(json.dumps(results,indent=2)+'\n')

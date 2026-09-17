from pathlib import Path
import sys,pickle,importlib.util,json
root=Path('/export/data/sfan/MLE_final_project_safety');sys.path.insert(0,str(root))
from agent_code.team_agent.opponent_transitions import enumerate_opponent_transition_scenarios,canonical_state_key
r=pickle.loads((root/'experiments/results/safety_round2_failure_20260917/failure_states.pkl').read_bytes());by={x['state']['step']:x for x in r}
s=by[182]['state'];target=by[183]['state'];sc=enumerate_opponent_transition_scenarios(s,'DOWN',progress_world=True)
def normalize(s):return {**s,'others':[(a[0],a[1],False,a[3]) for a in s['others']]}
for c in sc:
 if canonical_state_key(normalize(c.game_state))==canonical_state_key(normalize(target)):
  print('MATCH except opponent bomb capacity',c.action_profile,'predicted',c.game_state['others'],'actual',target['others'])
source=(root/'agent_code/team_agent/controllable_survival.py').read_text();source=source.replace('armed_names = {str(other[0]) for other in state["others"] if bool(other[2])}', 'armed_names = {str(other[0]) for other in state["others"]}').replace('{tuple(other[3]) for other in state["others"] if bool(other[2])}', '{tuple(other[3]) for other in state["others"]}')
p=Path('/tmp/probe_future_rearming.py');p.write_text(source);name='agent_code.team_agent._probe_future_rearming';spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
for step,actions in ((181,('BOMB',)),(182,('UP','DOWN','WAIT')),(183,('RIGHT','WAIT'))):
 for n in (4,5,6,7):
  proof=mod.controllable_survival_actions(by[step]['state'],actions,remaining_steps=n,budget_ms=60000)
  print(step,n,proof)

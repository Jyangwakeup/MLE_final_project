import pickle,json,time,cProfile,pstats,io
from pathlib import Path
from dataclasses import asdict
from agent_code.team_agent.controllable_survival import controllable_survival_actions
out=Path('experiments/results/safety_proven_movement_v9_20260917/failure')
rows=pickle.loads((out/'task4_failure_states.pkl').read_bytes());state=rows[-1]['state'];results=[]
for repeat in range(3):
 for actions in (('BOMB',),('RIGHT','LEFT','WAIT'),('RIGHT','LEFT','WAIT','BOMB')):
  start=time.perf_counter();r=controllable_survival_actions(state,actions,remaining_steps=7,budget_ms=60000,consider_opponent_rearming=True);ms=1000*(time.perf_counter()-start)
  d={'repeat':repeat,'actions':actions,'ms':ms,'result':asdict(r)};results.append(d);print(d,flush=True)
p=cProfile.Profile();p.enable();controllable_survival_actions(state,('RIGHT','LEFT','WAIT','BOMB'),remaining_steps=7,budget_ms=60000,consider_opponent_rearming=True);p.disable();s=io.StringIO();pstats.Stats(p,stream=s).sort_stats('cumulative').print_stats(25);(out/'profile.txt').write_text(s.getvalue());(out/'split_search.json').write_text(json.dumps(results,indent=2)+'\n')

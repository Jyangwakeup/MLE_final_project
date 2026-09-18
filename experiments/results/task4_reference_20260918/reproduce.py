import sys,pickle,json,pathlib
sys.path.insert(0,str(pathlib.Path.cwd()))
from agent_code.team_agent.safety import safety_decision
from experiments.task4_transfer import PARENT_SAFETY,TARGET_SAFETY
root=pathlib.Path('experiments/results/task4_300ms_20260918/terminal/run_evidence')
f=next(root.rglob('*s24614/task4_failure_states.pkl'))
rows=pickle.loads(f.read_bytes())
for row in rows[-4:]:
 s=row['state'];d=row['safety'];print('OBS',s['step'],row['action'],'self',s['self'],'bombs',s['bombs'],'others',s['others']);print({k:d.get(k) for k in ('v1_to_physical_fallback','opponent_to_v3_fallback','own_bomb_pending','own_bomb_placed_step','opponent_passing_counts','robust_guarantee_loss','physical_mask','decision_mask')})
 if s['step']==107:
  own={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer'],'placed_step':d['own_bomb_placed_step']}
  for _ in range(2):
   actual=safety_decision(s,d['physical_mask'],allow_bomb=True,exploring=False,safety_spec=PARENT_SAFETY,own_bomb_pending=d['own_bomb_pending'],own_bomb_state=own)
   print('REPRO',actual.robust_guarantee_loss,actual.robust_search_timed_out,actual.mask.tolist());assert actual.robust_guarantee_loss

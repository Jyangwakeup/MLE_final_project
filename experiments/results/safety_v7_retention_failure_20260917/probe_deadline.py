from pathlib import Path
import sys,pickle
root=Path('/export/data/sfan/MLE_final_project_safety');sys.path.insert(0,str(root))
from agent_code.team_agent.controllable_survival import controllable_survival_actions
p=next((root/'runs/safety_retention_v7_task3_a5a4e29').rglob('task4_failure_states.pkl'));rs=pickle.loads(p.read_bytes());by={r['state']['step']:r for r in rs}
for step,actions in ((159,('BOMB',)),(160,('RIGHT','WAIT')),(161,('RIGHT','WAIT')),(162,('UP','RIGHT','DOWN','WAIT'))):
 for horizon in sorted({3,4,5,6,7,166-step}):
  proof=controllable_survival_actions(by[step]['state'],actions,remaining_steps=horizon,budget_ms=60000,consider_opponent_rearming=True)
  print(step,horizon,proof,flush=True)

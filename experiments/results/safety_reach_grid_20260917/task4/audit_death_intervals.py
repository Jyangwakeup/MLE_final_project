import json
from pathlib import Path
root=Path('/export/data/sfan/MLE_final_project_safety_movement');out=[]
for ep in (root/'runs/safety_stress_task4_3f67cbc').glob('*/episodes.jsonl'):
 text=ep.read_text()
 if not text.strip() or not text.endswith('\n'):continue
 e=json.loads(text.splitlines()[0]);a=next(a for a in e['agents'] if a['name']=='double_dqn_continuous_v2_agent')
 if not a['dead']:continue
 rows=[json.loads(l) for l in (ep.parent/'timing.jsonl').read_text().splitlines()];rows=[r for r in rows if r['agent_name']=='double_dqn_continuous_v2_agent'];proofs=[]
 for r in rows:
  s=r['safety'];i=('UP','RIGHT','DOWN','LEFT','WAIT','BOMB').index(r['action'])
  if not s['opponent_to_v3_fallback'] and not s['v1_to_physical_fallback'] and s['opponent_passing_counts'][i]:
   end=s['own_bomb_placed_step']+7 if s['own_bomb_pending'] else r['step']+7
   proofs.append({'step':r['step'],'action':r['action'],'own_pending':s['own_bomb_pending'],'proof_endpoint_exclusive':end})
 last=proofs[-1] if proofs else None
 out.append({'seed':e['seed'],'death_step':a['death_step'],'last_proof':last,'death_before_endpoint':last is not None and a['death_step']<last['proof_endpoint_exclusive']})
(root/'runs/safety_stress_3f67cbc/death_proof_interval_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

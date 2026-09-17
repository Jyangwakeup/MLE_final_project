from pathlib import Path
import json,sys
import numpy as np
root=Path('/export/data/sfan/MLE_final_project_safety');result={}
for d in sorted(list((root/'runs').glob('safety_retention_v6_*_125c514')) + list((root/'runs').glob('safety_v6_engineering_425f622')) + list((root/'runs').glob('safety_v6_regression_425f622'))):
 times=[]
 for p in d.rglob('timing.jsonl'):
  for l in p.open():
   r=json.loads(l)
   if r['agent_name']=='double_dqn_continuous_v2_agent' and r.get('think_time') is not None: times.append(r['think_time'])
 result[d.name]={'act_count':len(times),'pooled_p95':float(np.percentile(times,95)),'max':max(times)}
print(json.dumps(result,indent=2));Path('/tmp/pooled-timing-all-v6.json').write_text(json.dumps(result,indent=2)+'\n')

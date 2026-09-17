import sys,json
from pathlib import Path
root=Path('/export/data/sfan/MLE_final_project_safety_movement');sys.path.insert(0,str(root))
from experiments.compare_evaluations import compare_evaluations
assert json.loads((root/'runs/safety_stress_3f67cbc/result.json').read_text())['status']=='passed'
oldroot=Path('/export/data/sfan/MLE_final_project_safety');groups=[]
for base,commit in ((root,'3f67cbc'),(oldroot,'0dc8dba')):
 groups.append([base/f'runs/safety_stress_task4_{commit}/safety_stress_task4_{commit}_s{s}' for s in range(24000,24060)])
for p,q in zip(*groups):
 a,b=[json.loads((d/'metadata.json').read_text()) for d in (p,q)]
 assert a['seeds']==b['seeds'] and a['exploration_disabled'] and b['exploration_disabled']
r=compare_evaluations(*groups,root/'runs/safety_stress_3f67cbc/paired_v8_comparison',bootstrap_samples=10000)
print([(m['metric'],m['mean_difference'],m['bootstrap_ci95_low'],m['bootstrap_ci95_high']) for m in r['metrics']])

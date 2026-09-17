import sys,json
from pathlib import Path
root=Path('/export/data/sfan/MLE_final_project_safety');sys.path.insert(0,str(root))
from experiments.compare_evaluations import compare_evaluations
state=json.loads((root/'runs/certified_placement/retention_5ff8d4c.json').read_text())
if state['status']!='passed':raise RuntimeError('Require completed passing controller before final paired analysis')
for task in (1,2,3):
 groups=[]
 for version in ('v8','v5'):
  base=root/'runs'/f'safety_retention_{version}_task{task}_5ff8d4c'
  paths=[base/f'safety_retention_{version}_task{task}_5ff8d4c_s{s}' for s in range(24000,24060)]
  assert all((p/'episodes.jsonl').exists() for p in paths)
  groups.append(paths)
 output=root/'runs/certified_placement/v8_final_comparison_5ff8d4c'/f'task{task}'
 r=compare_evaluations(*groups,output,bootstrap_samples=10000)
 print(task,[(m['metric'],m['mean_difference'],m['bootstrap_ci95_low'],m['bootstrap_ci95_high']) for m in r['metrics']])

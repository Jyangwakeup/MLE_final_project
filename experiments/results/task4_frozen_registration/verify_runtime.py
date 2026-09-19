from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import json,torch,unittest,shutil
from experiments.resume import load_training_snapshot
from experiments.task4_transfer import sha256
from experiments.compact_audit import audit_run
from experiments.task4_protocol import CANDIDATE
from experiments.task4_frozen_audit import audit_opponents
from tests.test_task4_transfer import assert_equal
r=Path('runs');case=unittest.TestCase()
reports={}
for label,left,right,left_prefix in [('twelve_rounds',r/'frozen_resume_continuous',r/'frozen_resume_second',r/'frozen_resume_first'),('rule_to_historical_boundary',r/'frozen_resume_first',r/'frozen_resume_switch',None)]:
 a=torch.load(left/'checkpoints/final.pt',map_location='cpu',weights_only=True);b=torch.load(right/'checkpoints/final.pt',map_location='cpu',weights_only=True)
 assert_equal(case,a,b)
 ra,rb=load_training_snapshot(left).runner_state,load_training_snapshot(right).runner_state
 for key in ('world_rng_state','python_rng_state','numpy_rng_state'):assert_equal(case,ra[key],rb[key])
 rows=lambda p:list(map(json.loads,(p/'timing.jsonl').read_text().splitlines()))
 x,y=rows(left),rows(right)
 fields=('round_index','step','agent_name','action','requested_action','skipped','timed_out','safety')
 if left_prefix:y=rows(left_prefix)+y
 else:x=[v for v in x if v['round_index']==6]
 project=lambda rs:[{k:v.get(k) for k in fields} for v in rs]
 assert_equal(case,project(x),project(y))
 schedules=lambda p:list(map(json.loads,(p/'opponent_schedule.jsonl').read_text().splitlines()))
 expected=schedules(left) if left_prefix else schedules(left)[-1:]
 actual=schedules(left_prefix)+schedules(right) if left_prefix else schedules(right)
 assert_equal(case,expected,actual)
 reports[label]=dict(equal=True,actual_actions=a['stage_action_steps'],updates=a['updates'],replay_count=a['replay']['count'],decisions=len(x),left=str(left),right=str(right),checkpoint_sha256=sha256(right/'checkpoints/final.pt'))
for directory in ('frozen_preflight_engineering_1','frozen_resume_continuous','frozen_resume_first','frozen_resume_second','frozen_resume_switch'):
 own=audit_run(r/directory,4,policy=CANDIDATE);others=audit_opponents(r/directory)
 assert not [f for f in own['failures'] if f!='zero_bomb_round_rate'],own['failures']
 assert not others['failures'],others['failures']
 reports[directory]=dict(learner=own,opponents=others)
dest=Path('experiments/results/task4_frozen_registration');dest.mkdir(parents=True,exist_ok=True)
(dest/'runtime_verification.json').write_text(json.dumps(reports,indent=2,sort_keys=True)+'\n')
for source,name in [('/tmp/frozen-full-unit.log','full-unittest.txt'),('/tmp/frozen-controller-unit.log','frozen-unittest.txt')]:shutil.copy2(source,dest/name)
print(json.dumps({k:v for k,v in reports.items() if k in ('twelve_rounds','rule_to_historical_boundary')},indent=2))

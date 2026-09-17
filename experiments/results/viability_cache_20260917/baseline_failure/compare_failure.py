from pathlib import Path
import json,time
import numpy as np
from dataclasses import asdict
from unittest.mock import patch
from tests.test_order_equivalence import search_arguments,native
from tests import reference_controllable_survival as scalar
from tests import reference_vectorized_survival as vector
from tests import reference_opponent_transitions as original_orders
from agent_code.team_agent import controllable_survival as current
p=Path(__file__).with_name('failure_states.json')
records=json.loads(p.read_text());out=[]
for record in records:
 state=record['state']
 for name in ('field','explosion_map'):state[name]=np.asarray(state[name])
 state['self']=(*state['self'][:3],tuple(state['self'][3]))
 state['others']=[(*actor[:3],tuple(actor[3])) for actor in state['others']]
 state['bombs']=[(tuple(pos),timer) for pos,timer in state['bombs']]
for r in records:
 actions,remaining=search_arguments(r);results={}
 for name,module in [('original',scalar),('vectorized',vector),('cached',current)]:
  start=time.perf_counter()
  if name=='original':
   with patch.object(module,'enumerate_opponent_transition_scenarios',original_orders.enumerate_opponent_transition_scenarios):
    result=module.controllable_survival_actions(r['state'],actions,remaining_steps=remaining,budget_ms=60000)
  else:result=module.controllable_survival_actions(r['state'],actions,remaining_steps=remaining,budget_ms=60000)
  results[name]={'seconds':time.perf_counter()-start,'result':asdict(result)}
 assert all(not v['result']['timed_out'] for v in results.values())
 assert results['original']['result']==results['vectorized']['result']==results['cached']['result'],r['state']['step']
 out.append(dict(step=r['state']['step'],actions=actions,remaining=remaining,implementations=results))
print('20 states: original enumeration + scalar, vectorized, cached proofs equal')
# Do not overwrite frozen evidence when reproducing.
print(json.dumps(native(out[-1]),indent=2))

"""A new placement contract, keeping v5 behavior and failure counters intact."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from agent_code.team_agent.safety import safety_decision, resolve_safety_spec
from agent_code.team_agent.controllable_survival import ControllableSurvivalResult
from tests.test_danger import make_game_state
from tests.test_opponent_robust_safety import V5

V6 = {**V5, 'version':'survival-mask-v6', 'fallback':'non_bomb_physical_q'}
ROOT=Path(__file__).resolve().parents[1]
def failure_record(offset):
 r=json.loads((ROOT/'experiments/results/viability_cache_20260917/baseline_failure/failure_states.json').read_text())[offset]
 s=r['state']
 for k in ('field','explosion_map'):s[k]=np.asarray(s[k])
 s['self']=(*s['self'][:3],tuple(s['self'][3]))
 s['others']=[(*a[:3],tuple(a[3])) for a in s['others']]
 s['bombs']=[(tuple(p),t) for p,t in s['bombs']]
 s['coins']=[tuple(p) for p in s['coins']]
 return r

class CertifiedPlacementTests(unittest.TestCase):
 def decide(self,record,spec):
  d=record['safety']
  return safety_decision(record['state'],np.asarray(d['physical_mask']),spec,allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer']})
 def test_historical_no_safe_fallback_cannot_place_unproved_bomb(self):
  d=self.decide(failure_record(-2),V6)
  self.assertEqual(d.mask.tolist(),[False,False,False,False,True,False])
  self.assertTrue(d.physical_fallback)
 def test_legacy_contract_is_unchanged(self):
  d=self.decide(failure_record(-2),V5)
  self.assertTrue(d.mask[5]);self.assertTrue(d.physical_fallback)
 def test_existing_guarantee_loss_is_not_relabelled(self):
  d=self.decide(failure_record(-1),V6)
  self.assertTrue(d.robust_guarantee_loss);self.assertTrue(d.opponent_fallback)
  self.assertEqual(d.opponent_failing_profiles[0],('UP','WAIT','WAIT'))
 def test_unproved_bomb_as_only_base_action_never_returns_empty_mask(self):
  state=make_game_state(position=(7,7));physical=np.ones(6,dtype=bool)
  base=np.array([False]*5+[True])
  result=ControllableSurvivalResult((),1,1,False)
  with patch('agent_code.team_agent.safety.survival_mask',return_value=(base,False)),patch('agent_code.team_agent.controllable_survival.controllable_survival_actions',return_value=result):
   d=safety_decision(state,physical,V6,allow_bomb=True,exploring=False)
  self.assertEqual(d.mask.tolist(),[True]*5+[False]);self.assertTrue(d.physical_fallback)
 def test_proven_bomb_still_available(self):
  state=make_game_state(position=(7,7))
  d=safety_decision(state,np.ones(6,dtype=bool),V6,allow_bomb=True,exploring=False)
  self.assertTrue(d.mask[5]);self.assertGreater(d.opponent_passing_counts[5],0)
 def test_timeout_is_still_a_failure_and_not_a_placement_permit(self):
  state=make_game_state(position=(7,7));result=ControllableSurvivalResult((),0,0,True)
  with patch('agent_code.team_agent.controllable_survival.controllable_survival_actions',return_value=result):
   d=safety_decision(state,np.ones(6,dtype=bool),V6,allow_bomb=True,exploring=False)
  self.assertFalse(d.mask[5]);self.assertTrue(d.robust_search_timed_out)
 def test_versioned_fallback_cannot_be_silently_changed(self):
  self.assertEqual(resolve_safety_spec(V6),V6)
  with self.assertRaises(ValueError):resolve_safety_spec({**V6,'fallback':'physical_q'})
  with self.assertRaises(ValueError):resolve_safety_spec({**V5,'fallback':'non_bomb_physical_q'})

 def test_frozen_override_preserves_parent_weights_and_actual_act_veto(self):
  import os,logging,torch
  from types import SimpleNamespace
  from agent_code.double_dqn_continuous_v2_agent import callbacks
  parent=ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt'
  owner=SimpleNamespace(train=False,logger=logging.getLogger('certified-placement'))
  with patch.dict(os.environ,{'BOMBERMAN_CHECKPOINT':str(parent),'BOMBERMAN_SAFETY_SPEC':json.dumps(V6)}):
   callbacks.setup(owner)
  payload=torch.load(parent,map_location='cpu',weights_only=True)
  self.assertEqual(owner.safety_spec,V6)
  self.assertTrue(all(torch.equal(t,owner.model.policy.state_dict()[k].cpu()) for k,t in payload['policy'].items()))
  self.assertEqual(callbacks.act(owner,failure_record(-2)['state']),'WAIT')
  self.assertFalse(owner.last_safety_diagnostic['decision_mask'][5])
 def test_v5_training_resume_cannot_silently_adopt_v6(self):
  import os,logging
  from types import SimpleNamespace
  from agent_code.double_dqn_continuous_v2_agent import callbacks
  owner=SimpleNamespace(train=True,logger=logging.getLogger('certified-placement'))
  parent=ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt'
  with patch.dict(os.environ,{'BOMBERMAN_CHECKPOINT':str(parent),'BOMBERMAN_SAFETY_SPEC':json.dumps(V6),'BOMBERMAN_TRAINING_TASK':'weak_opponents'}):
   with self.assertRaisesRegex(ValueError,'incompatible safety specification'):callbacks.setup(owner)

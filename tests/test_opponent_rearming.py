"""Real world 24266: a disarmed opponent recovers and traps an own-bomb escape."""
import json
from pathlib import Path
import unittest
import numpy as np
from agent_code.team_agent.controllable_survival import controllable_survival_actions
from tests.test_certified_placement import V6

ROOT = Path(__file__).resolve().parents[1]


def record(step):
    records = json.loads((ROOT/'experiments/results/safety_round2_failure_20260917/failure_states.json').read_text())
    r = next(r for r in records if r['state']['step']==step)
    state = r['state']
    for key in ('field','explosion_map'):
        state[key] = np.asarray(state[key])
    state['self'] = (*state['self'][:3],tuple(state['self'][3]))
    state['others'] = [(*a[:3],tuple(a[3])) for a in state['others']]
    state['bombs'] = [(tuple(p),t) for p,t in state['bombs']]
    return r


class OpponentRearmingTest(unittest.TestCase):
    def test_recorded_escape_rejects_down_when_opponent_can_recover(self):
        state = record(182)['state']
        self.assertFalse(state['others'][0][2])
        result = controllable_survival_actions(state, ('UP','DOWN','WAIT'),
            remaining_steps=7, budget_ms=400, consider_opponent_rearming=True)
        self.assertFalse(result.timed_out)
        self.assertNotIn('DOWN',result.proven_actions)
        self.assertIn('UP',result.proven_actions)
        self.assertEqual(result.first_failing_profile,('DOWN','LEFT'))

    def test_legacy_short_interval_exposes_both_missing_conditions(self):
        state = record(182)['state']
        for horizon, rearming in ((5,True),(7,False)):
            with self.subTest(horizon=horizon,rearming=rearming):
                result = controllable_survival_actions(state, ('DOWN',),
                    remaining_steps=horizon,budget_ms=400,
                    consider_opponent_rearming=rearming)
                self.assertFalse(result.timed_out)
                self.assertIn('DOWN',result.proven_actions)


V7 = {**V6, 'version':'survival-mask-v7',
      'danger_interval':'own_bomb_and_horizon',
      'opponent_rearming':'possible_after_first_transition'}


class RearmingSafetyContractTest(unittest.TestCase):
    def decide(self, step, spec):
        from agent_code.team_agent.safety import safety_decision
        r = record(step);d = r['safety']
        return safety_decision(r['state'],np.asarray(d['physical_mask']),spec,
            allow_bomb=True,exploring=False,own_bomb_pending=d['own_bomb_pending'],
            own_bomb_state={'pending':d['own_bomb_pending'],'timer':d['own_bomb_timer']})

    def test_new_contract_blocks_the_recorded_bad_move_before_the_trap(self):
        old = self.decide(182,V6);new = self.decide(182,V7)
        self.assertTrue(old.mask[2])
        self.assertEqual(new.mask.tolist(),[True,False,False,False,True,False])
        self.assertFalse(new.robust_guarantee_loss)
        self.assertFalse(new.robust_search_timed_out)

    def test_too_late_state_is_still_reported_as_a_failure(self):
        for step in (183,184):
            with self.subTest(step=step):
                self.assertTrue(self.decide(step,V7).robust_guarantee_loss)

    def test_recorded_placement_keeps_a_proven_escape_available(self):
        result = self.decide(181,V7)
        self.assertTrue(result.mask[5])
        self.assertGreater(result.opponent_passing_counts[5],0)
        self.assertFalse(result.robust_search_timed_out)

    def test_version_and_rearming_identity_are_explicit(self):
        from agent_code.team_agent.safety import resolve_safety_spec
        self.assertEqual(resolve_safety_spec(V7),V7)
        with self.assertRaises(ValueError):
            resolve_safety_spec({**V7,'opponent_rearming':'never'})
        with self.assertRaises(ValueError):
            resolve_safety_spec({**V7,'danger_interval':'own_bomb_and_lingering'})

    def test_old_parent_can_only_use_new_contract_as_explicit_frozen_override(self):
        import os,logging,torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from agent_code.double_dqn_continuous_v2_agent import callbacks
        parent=ROOT/'agent_code/double_dqn_continuous_v2_agent/task3_validated.pt'
        env={'BOMBERMAN_CHECKPOINT':str(parent),'BOMBERMAN_SAFETY_SPEC':json.dumps(V7)}
        owner=SimpleNamespace(train=False,logger=logging.getLogger('rearming'))
        with patch.dict(os.environ,env):callbacks.setup(owner)
        payload=torch.load(parent,map_location='cpu',weights_only=True)
        self.assertEqual(owner.safety_spec,V7)
        self.assertTrue(all(torch.equal(t,owner.model.policy.state_dict()[k].cpu()) for k,t in payload['policy'].items()))
        training=SimpleNamespace(train=True,logger=logging.getLogger('rearming'))
        with patch.dict(os.environ,{**env,'BOMBERMAN_TRAINING_TASK':'weak_opponents'}):
            with self.assertRaisesRegex(ValueError,'incompatible safety specification'):
                callbacks.setup(training)

"""A recorded opponent trap must not admit an unproved move over proved alternatives."""
import json
from pathlib import Path
import unittest
import numpy as np
from agent_code.team_agent.safety import safety_decision
from tests.test_fixed_safety_deadline import V8


V9 = {**V8, "version": "survival-mask-v9", "nonpending_movement": "prefer_proven_actions"}

def recorded_trap():
    row = json.loads((Path(__file__).parent / 'fixtures/nonpending_movement_24025.json').read_text())
    state = row['state']
    for key in ('field', 'explosion_map'):
        state[key] = np.asarray(state[key])
    state['self'] = (*state['self'][:3], tuple(state['self'][3]))
    state['others'] = [(*a[:3], tuple(a[3])) for a in state['others']]
    state['bombs'] = [(tuple(p), t) for p, t in state['bombs']]
    return row


class ProvenMovementTests(unittest.TestCase):
    def test_recorded_trap_preserves_proven_alternatives(self):
        row = recorded_trap()
        decision = safety_decision(row['state'], np.asarray(row['safety']['physical_mask']),
                                   V9, allow_bomb=True, exploring=False,
                                   own_bomb_pending=False)
        self.assertFalse(decision.robust_search_timed_out)
        self.assertFalse(decision.mask[1], 'Unproved RIGHT must yield to proved movement')
        self.assertTrue(decision.mask[0])
        self.assertTrue(decision.mask[3])
        self.assertTrue(decision.mask[4])

    def test_legacy_v8_retains_original_mask(self):
        row = recorded_trap()
        decision = safety_decision(row['state'], np.asarray(row['safety']['physical_mask']),
                                   V8, allow_bomb=True, exploring=False,
                                   own_bomb_pending=False)
        self.assertTrue(decision.mask[1])

    def test_empty_proof_records_fallback_without_inventing_bomb_certificate(self):
        from unittest.mock import patch
        from agent_code.team_agent.controllable_survival import ControllableSurvivalResult
        row = recorded_trap()
        empty = ControllableSurvivalResult((), 1, 1, False, None, None)
        with patch('agent_code.team_agent.controllable_survival.controllable_survival_actions', return_value=empty):
            decision = safety_decision(row['state'], np.asarray(row['safety']['physical_mask']),
                                       V9, allow_bomb=True, exploring=False,
                                       own_bomb_pending=False)
        self.assertTrue(decision.opponent_fallback)
        self.assertFalse(decision.robust_guarantee_loss)
        self.assertFalse(decision.mask[5])
        self.assertTrue(decision.mask[:5].any())

    def test_own_responsibility_keeps_fixed_deadline(self):
        from tests.test_fixed_safety_deadline import deadline_record, decide
        decision = decide(deadline_record(162), 159, V9)
        self.assertFalse(decision.robust_guarantee_loss)
        self.assertEqual(decision.mask.tolist(), [False, True, False, False, False, False])

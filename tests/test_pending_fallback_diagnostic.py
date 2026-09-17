"""An exhausted own-bomb obligation must not disappear behind physical fallback."""
import unittest
import numpy as np
from agent_code.team_agent.safety import safety_decision
from tests.test_certified_placement import V5, V6
from tests.test_danger import make_game_state
from experiments.task4_worker import safety_failure


class PendingFallbackDiagnosticTest(unittest.TestCase):
    def trapped(self, flame=False):
        state = make_game_state(position=(7, 7), bombs_left=False)
        for position in ((6, 7), (8, 7), (7, 6), (7, 8)):
            state['field'][position] = 1
        if flame:
            state['explosion_map'][7, 7] = 2
        else:
            state['bombs'] = [((7, 7), 0)]
        return state

    def decide(self, state, pending=True, spec=V6):
        return safety_decision(state, np.array([False]*4+[True, False]), spec,
            allow_bomb=True, exploring=False, own_bomb_pending=pending,
            own_bomb_state={'pending': pending, 'timer': 0})

    def test_own_bomb_without_horizon_survivor_reports_loss(self):
        state = self.trapped()
        result = self.decide(state)
        self.assertTrue(result.physical_fallback)
        self.assertTrue(result.robust_guarantee_loss)
        self.assertTrue(result.opponent_fallback)
        self.assertEqual(result.mask.tolist(), [False]*4+[True, False])
        diagnostic = dict(diagnostic_version='escape-collapse-v2',
                          robust_guarantee_loss=result.robust_guarantee_loss)
        self.assertEqual(safety_failure(state, 'WAIT', diagnostic, elapsed=.001),
                         'robust_guarantee_loss')

    def test_visible_bomb_disappearance_during_flames_does_not_hide_loss(self):
        result = self.decide(self.trapped(flame=True))
        self.assertTrue(result.physical_fallback)
        self.assertTrue(result.robust_guarantee_loss)
        self.assertFalse(result.robust_search_timed_out)

    def test_opponent_only_danger_does_not_invent_own_bomb_obligation(self):
        result = self.decide(self.trapped(), pending=False)
        self.assertTrue(result.physical_fallback)
        self.assertFalse(result.robust_guarantee_loss)

    def test_legacy_v5_mask_and_diagnostics_remain_unchanged(self):
        result = self.decide(self.trapped(), spec=V5)
        self.assertTrue(result.physical_fallback)
        self.assertFalse(result.robust_guarantee_loss)
        self.assertEqual(result.mask.tolist(), [False]*4+[True, False])

    def test_pending_bomb_with_proven_escape_does_not_report_loss(self):
        state = make_game_state(position=(7, 7), bombs_left=False,
                                bombs=[((7, 7), 3)])
        result = safety_decision(state, np.array([True]*5+[False]), V6,
            allow_bomb=True, exploring=False, own_bomb_pending=True,
            own_bomb_state={'pending': True, 'timer': 3})
        self.assertFalse(result.physical_fallback)
        self.assertFalse(result.robust_guarantee_loss)
        self.assertTrue(result.mask.any())
        self.assertGreater(sum(result.opponent_passing_counts), 0)

from types import SimpleNamespace
import unittest

import numpy as np

from all_other_agent_code.rainbow_lite_v11_agent.features import (
    features_for_state, init_action_history,
)
from agent_code.team_agent.feature_system.continuous_v11 import (
    FEATURE_DIM, extract, quadrant_crate_objectives,
)
from tests.test_danger import make_game_state


class QuadrantV11Tests(unittest.TestCase):
    def test_runtime_appends_eight_crate_objective_features(self):
        state = make_game_state(position=(5, 5))
        state.update(round=1, step=1)
        owner = SimpleNamespace()
        init_action_history(owner)
        direct = extract(state)
        runtime = features_for_state(owner, state)
        self.assertEqual(FEATURE_DIM, 170)
        np.testing.assert_allclose(direct.vector[-8:], runtime.vector[-8:])

    def test_shares_count_only_reachable_crates_and_sum_to_one(self):
        state = make_game_state(position=(5, 5))
        state["field"][3, 3] = 1
        state["field"][7, 7] = 1
        values = quadrant_crate_objectives(state)
        shares, distances = values[:4], values[4:]
        self.assertAlmostEqual(float(shares.sum()), 1.0)
        self.assertGreater(shares[0], 0.0)
        self.assertGreater(shares[3], 0.0)
        self.assertLess(distances[0], 1.0)
        self.assertLess(distances[3], 1.0)
        self.assertEqual(distances[1], 1.0)
        self.assertEqual(distances[2], 1.0)

    def test_nearest_frontier_distance_distinguishes_near_and_far_groups(self):
        state = make_game_state(position=(8, 7))
        state["field"][6, 5] = 1
        state["field"][12, 11] = 1
        values = quadrant_crate_objectives(state)
        self.assertLess(values[4], values[7])


if __name__ == "__main__":
    unittest.main()

from types import SimpleNamespace
import unittest

import numpy as np

from all_other_agent_code.rainbow_lite_v10_agent.features import (
    features_for_state, init_action_history,
)
from agent_code.team_agent.feature_system.continuous_v10 import (
    FEATURE_DIM, QUADRANT_FIELDS, extract, quadrant_densities,
)
from tests.test_danger import make_game_state


class QuadrantV10Tests(unittest.TestCase):
    def test_schema_and_runtime_append_eight_density_features(self):
        state = make_game_state(position=(5, 5))
        state.update(round=1, step=1)
        owner = SimpleNamespace()
        init_action_history(owner)
        direct = extract(state)
        runtime = features_for_state(owner, state)
        self.assertEqual(len(QUADRANT_FIELDS), 8)
        self.assertEqual(FEATURE_DIM, 170)
        self.assertEqual(direct.vector.shape, (170,))
        self.assertEqual(runtime.vector.shape, (170,))
        np.testing.assert_allclose(direct.vector[-8:], runtime.vector[-8:])

    def test_objects_land_in_agent_centred_quadrants(self):
        state = make_game_state(position=(5, 5))
        state["field"][3, 3] = 1
        state["field"][7, 7] = 1
        state["others"] = [
            ("ne", 0, True, (7, 3)),
            ("sw", 0, True, (3, 7)),
        ]
        values = quadrant_densities(state)
        crate, opponent = values[:4], values[4:]
        self.assertGreater(crate[0], 0.0)
        self.assertEqual(crate[1], 0.0)
        self.assertEqual(crate[2], 0.0)
        self.assertGreater(crate[3], 0.0)
        self.assertEqual(opponent[0], 0.0)
        self.assertGreater(opponent[1], 0.0)
        self.assertGreater(opponent[2], 0.0)
        self.assertEqual(opponent[3], 0.0)

    def test_axis_objects_split_evenly_between_adjacent_quadrants(self):
        state = make_game_state(position=(8, 7))
        state["field"][8, 5] = 1
        values = quadrant_densities(state)[:4]
        self.assertGreater(values[0], 0.0)
        self.assertAlmostEqual(values[0], values[1])
        self.assertEqual(values[2], 0.0)
        self.assertEqual(values[3], 0.0)


if __name__ == "__main__":
    unittest.main()

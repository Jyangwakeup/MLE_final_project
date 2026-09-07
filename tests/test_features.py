import copy
import unittest

import numpy as np

from agent_code.team_agent.features import ACTIONS, safety_features
from tests.test_danger import CRATE, WALL, make_game_state


class SafetyFeaturesTestCase(unittest.TestCase):
    def test_movement_categories_cover_illegal_unsafe_and_escapable(self):
        illegal = make_game_state()
        illegal['field'][3, 2] = WALL

        unsafe = make_game_state(position=(2, 3), bombs=[((4, 3), 0)])
        escapable = make_game_state()

        self.assertEqual(safety_features(illegal).state_key[0], 0)
        self.assertEqual(safety_features(unsafe).state_key[1], 1)
        self.assertEqual(safety_features(escapable).state_key[0], 2)

    def test_current_danger_categories_cover_now_later_and_absent(self):
        safe = make_game_state()
        now = make_game_state(bombs=[((3, 3), 0)])
        later = make_game_state(bombs=[((3, 3), 1)])

        self.assertEqual(safety_features(safe).state_key[4], 0)
        self.assertEqual(safety_features(now).state_key[4], 1)
        self.assertEqual(safety_features(later).state_key[4], 2)

    def test_bomb_and_crate_categories_cover_all_values(self):
        unavailable = make_game_state(bombs_left=False)
        dead_end = make_game_state()
        for x, y in [(2, 3), (4, 3), (3, 2), (3, 4)]:
            dead_end['field'][x, y] = WALL
        one_crate = make_game_state()
        one_crate['field'][4, 3] = CRATE
        useful = make_game_state()
        useful['field'][4, 3] = CRATE
        useful['field'][5, 3] = CRATE

        self.assertEqual(safety_features(unavailable).state_key[5], 0)
        self.assertEqual(safety_features(dead_end).state_key[5], 1)
        self.assertEqual(safety_features(useful).state_key[5], 2)
        self.assertEqual(safety_features(unavailable).state_key[6], 0)
        self.assertEqual(safety_features(make_game_state()).state_key[6], 0)
        self.assertEqual(safety_features(one_crate).state_key[6], 1)
        self.assertEqual(safety_features(useful).state_key[6], 2)

    def test_vector_is_one_hot_float32_for_every_safety_field(self):
        features = safety_features(make_game_state())

        self.assertEqual(features.vector.shape, (21,))
        self.assertEqual(features.vector.dtype, np.float32)
        self.assertEqual(features.vector.sum(), 7.0)
        for index, value in enumerate(features.state_key):
            segment = features.vector[index * 3:(index + 1) * 3]
            self.assertEqual(segment.sum(), 1.0)
            self.assertEqual(segment[value], 1.0)

    def test_legal_mask_uses_physics_not_predicted_danger(self):
        state = make_game_state(position=(2, 3), bombs=[((4, 3), 0)])
        features = safety_features(state)

        self.assertEqual(tuple(ACTIONS), ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'))
        self.assertTrue(features.legal_mask[1])
        self.assertEqual(features.state_key[1], 1)
        self.assertTrue(features.legal_mask[4])
        self.assertTrue(features.legal_mask[5])

    def test_none_state_returns_none(self):
        self.assertIsNone(safety_features(None))

    def test_feature_extraction_does_not_modify_game_state(self):
        state = make_game_state(bombs=[((4, 3), 2)])
        state['field'][3, 2] = CRATE
        original = copy.deepcopy(state)

        safety_features(state)

        self.assertTrue(np.array_equal(state['field'], original['field']))
        self.assertTrue(np.array_equal(state['explosion_map'], original['explosion_map']))
        self.assertEqual(state['bombs'], original['bombs'])
        self.assertEqual(state['self'], original['self'])


if __name__ == '__main__':
    unittest.main()

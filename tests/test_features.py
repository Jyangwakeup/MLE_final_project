import copy
import unittest

import numpy as np

from agent_code.team_agent.features import (
    ACTIONS,
    coin_features,
    extract_features,
    opponent_features,
    safety_features,
)
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


class AggregatedFeaturesTestCase(unittest.TestCase):
    def test_aggregation_uses_safety_coin_opponent_group_order(self):
        state = make_game_state(position=(3, 3))
        state['field'][4, 3] = CRATE
        state['coins'] = [(3, 1)]
        state['others'] = [('other', 0, True, (3, 5))]

        safety = safety_features(state)
        coins = coin_features(state)
        opponents = opponent_features(state)
        features = extract_features(state)

        self.assertEqual(
            features.state_key,
            safety.state_key + coins.state_key + opponents.state_key,
        )
        np.testing.assert_array_equal(
            features.vector,
            np.concatenate((safety.vector, coins.vector, opponents.vector)),
        )
        np.testing.assert_array_equal(features.legal_mask, safety.legal_mask)

    def test_formal_interface_has_fixed_shapes_types_and_one_hot_segments(self):
        state = make_game_state(position=(3, 3), bombs_left=False)
        state['coins'] = [(3, 1)]
        state['others'] = [('other', 0, True, (6, 3))]

        features = extract_features(state)

        self.assertEqual(
            features.state_key,
            (2, 2, 2, 2, 0, 0, 0, 1, 0, 0, 0, 2, 2, 1),
        )
        self.assertEqual(len(features.state_key), 14)
        self.assertEqual(features.vector.shape, (40,))
        self.assertEqual(features.vector.dtype, np.float32)
        self.assertEqual(features.vector.sum(), 14.0)
        self.assertEqual(features.legal_mask.shape, (6,))
        self.assertEqual(features.legal_mask.dtype, np.bool_)
        self.assertEqual(tuple(ACTIONS), ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'))
        self.assertFalse(features.legal_mask[5])
        self.assertEqual(features.state_key[0:7], safety_features(state).state_key)
        self.assertEqual(features.state_key[7:11], coin_features(state).state_key)
        self.assertEqual(features.state_key[11:14], opponent_features(state).state_key)
        self.assertEqual(features.vector[0:21].sum(), 7.0)
        self.assertEqual(features.vector[21:29].sum(), 4.0)
        self.assertEqual(features.vector[29:40].sum(), 3.0)

    def test_none_state_returns_none(self):
        self.assertIsNone(extract_features(None))

    def test_aggregation_is_deterministic_and_does_not_modify_game_state(self):
        state = make_game_state(position=(3, 3), bombs=[((5, 3), 2)])
        state['field'][4, 3] = CRATE
        state['coins'] = [(3, 1), (5, 5)]
        state['others'] = [('other', 0, True, (3, 5))]
        original = copy.deepcopy(state)

        first = extract_features(state)
        second = extract_features(state)

        self.assertEqual(first.state_key, second.state_key)
        np.testing.assert_array_equal(first.vector, second.vector)
        np.testing.assert_array_equal(first.legal_mask, second.legal_mask)
        self.assertTrue(np.array_equal(state['field'], original['field']))
        self.assertTrue(np.array_equal(state['explosion_map'], original['explosion_map']))
        self.assertEqual(state['bombs'], original['bombs'])
        self.assertEqual(state['coins'], original['coins'])
        self.assertEqual(state['others'], original['others'])
        self.assertEqual(state['self'], original['self'])


if __name__ == '__main__':
    unittest.main()

import copy
import unittest

import numpy as np

from agent_code.team_agent.features import coin_features
from tests.test_danger import CRATE, WALL, make_game_state


class CoinFeaturesTestCase(unittest.TestCase):
    def test_direction_toward_fixed_coin_is_closer(self):
        state = make_game_state(position=(3, 3))
        state['coins'] = [(3, 1)]

        features = coin_features(state)

        self.assertEqual(features.state_key, (1, 0, 0, 0))

    def test_wall_crate_bomb_and_other_agent_block_navigation(self):
        states = []

        wall = make_game_state(position=(3, 3))
        wall['coins'] = [(3, 1)]
        wall['field'][3, 2] = WALL
        states.append(wall)

        crate = make_game_state(position=(3, 3))
        crate['coins'] = [(3, 1)]
        crate['field'][3, 2] = CRATE
        states.append(crate)

        bomb = make_game_state(position=(3, 3), bombs=[((3, 2), 3)])
        bomb['coins'] = [(3, 1)]
        states.append(bomb)

        other = make_game_state(position=(3, 3))
        other['coins'] = [(3, 1)]
        other['others'] = [('other', 0, True, (3, 2))]
        states.append(other)

        for state in states:
            self.assertEqual(coin_features(state).state_key[0], 0)

    def test_missing_or_unreachable_coins_produce_zero_fields(self):
        missing = make_game_state()
        unreachable = make_game_state(position=(3, 3))
        unreachable['coins'] = [(3, 1)]
        for x, y in [(3, 2), (2, 3), (4, 3), (3, 4)]:
            unreachable['field'][x, y] = WALL

        self.assertEqual(coin_features(missing).state_key, (0, 0, 0, 0))
        self.assertEqual(coin_features(unreachable).state_key, (0, 0, 0, 0))

    def test_equal_distance_coins_use_coordinate_tie_breaking(self):
        state = make_game_state(position=(3, 3))
        state['coins'] = [(5, 3), (1, 3)]

        features = coin_features(state)

        self.assertEqual(features.state_key, (0, 0, 0, 1))

    def test_vector_is_float32_one_hot_for_each_direction(self):
        state = make_game_state(position=(3, 3))
        state['coins'] = [(3, 1)]

        features = coin_features(state)

        self.assertEqual(features.vector.shape, (8,))
        self.assertEqual(features.vector.dtype, np.float32)
        self.assertEqual(features.vector.sum(), 4.0)
        for index, value in enumerate(features.state_key):
            segment = features.vector[index * 2:(index + 1) * 2]
            self.assertEqual(segment.sum(), 1.0)
            self.assertEqual(segment[value], 1.0)

    def test_none_state_returns_none(self):
        self.assertIsNone(coin_features(None))

    def test_coin_features_do_not_modify_game_state(self):
        state = make_game_state(position=(3, 3), bombs=[((4, 3), 2)])
        state['coins'] = [(3, 1), (5, 5)]
        original = copy.deepcopy(state)

        coin_features(state)

        self.assertTrue(np.array_equal(state['field'], original['field']))
        self.assertTrue(np.array_equal(state['explosion_map'], original['explosion_map']))
        self.assertEqual(state['bombs'], original['bombs'])
        self.assertEqual(state['coins'], original['coins'])
        self.assertEqual(state['self'], original['self'])


if __name__ == '__main__':
    unittest.main()

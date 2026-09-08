import copy
import unittest

import numpy as np

from agent_code.team_agent.features import _nearest_opponent, opponent_features
from tests.test_danger import CRATE, WALL, make_game_state


class OpponentFeaturesTestCase(unittest.TestCase):
    def test_no_opponents_produce_zero_fields(self):
        features = opponent_features(make_game_state())

        self.assertEqual(features.state_key, (0, 0, 0))
        self.assertFalse(hasattr(features, 'target'))
        self.assertFalse(hasattr(features, 'recommended_direction'))

    def test_direction_uses_dominant_axis_and_fixed_tie_breaking(self):
        cases = [
            ((3, 1), 1),
            ((5, 3), 2),
            ((3, 5), 3),
            ((1, 3), 4),
            ((2, 2), 1),
            ((4, 2), 1),
            ((4, 4), 2),
            ((2, 4), 3),
        ]
        for opponent_position, expected in cases:
            with self.subTest(opponent_position=opponent_position):
                state = make_game_state(position=(3, 3))
                state['others'] = [('other', 0, True, opponent_position)]

                self.assertEqual(opponent_features(state).state_key[0], expected)

    def test_distance_categories_cover_boundaries(self):
        cases = [((8, 7), 1), ((9, 7), 2), ((11, 7), 2), ((12, 7), 3)]
        for opponent_position, expected in cases:
            with self.subTest(opponent_position=opponent_position):
                state = make_game_state(position=(7, 7))
                state['others'] = [('other', 0, True, opponent_position)]

                self.assertEqual(opponent_features(state).state_key[1], expected)

    def test_nearest_opponent_order_is_distance_coordinate_then_name(self):
        distance_first = make_game_state(position=(5, 5))
        distance_first['others'] = [
            ('far', 0, True, (3, 5)),
            ('near', 0, True, (6, 5)),
        ]
        coordinate_second = make_game_state(position=(5, 5))
        coordinate_second['others'] = [
            ('up', 0, True, (5, 4)),
            ('left', 0, True, (4, 5)),
        ]
        name_third = make_game_state(position=(5, 5))
        name_third['others'] = [
            ('beta', 0, True, (5, 4)),
            ('alpha', 0, True, (5, 4)),
        ]

        self.assertEqual(_nearest_opponent(distance_first)[0], 'near')
        self.assertEqual(_nearest_opponent(coordinate_second)[0], 'left')
        self.assertEqual(_nearest_opponent(name_third)[0], 'alpha')

    def test_blast_coverage_checks_all_opponents(self):
        state = make_game_state(position=(3, 3))
        state['others'] = [
            ('nearest', 0, True, (4, 4)),
            ('covered', 0, True, (6, 3)),
        ]

        self.assertEqual(opponent_features(state).state_key[2], 1)

    def test_stone_blocks_blast_but_crate_does_not(self):
        stone = make_game_state(position=(3, 3))
        stone['field'][5, 3] = WALL
        stone['others'] = [('other', 0, True, (6, 3))]
        crate = make_game_state(position=(3, 3))
        crate['field'][5, 3] = CRATE
        crate['others'] = [('other', 0, True, (6, 3))]

        self.assertEqual(opponent_features(stone).state_key[2], 0)
        self.assertEqual(opponent_features(crate).state_key[2], 1)

    def test_blast_coverage_does_not_depend_on_bomb_availability(self):
        state = make_game_state(position=(3, 3), bombs_left=False)
        state['others'] = [('other', 0, True, (6, 3))]

        self.assertEqual(opponent_features(state).state_key[2], 1)

    def test_vector_is_float32_one_hot_for_each_opponent_field(self):
        state = make_game_state(position=(3, 3))
        state['others'] = [('other', 0, True, (6, 3))]

        features = opponent_features(state)

        self.assertEqual(features.vector.shape, (11,))
        self.assertEqual(features.vector.dtype, np.float32)
        self.assertEqual(features.vector.sum(), 3.0)
        offset = 0
        for value, size in zip(features.state_key, (5, 4, 2)):
            segment = features.vector[offset:offset + size]
            self.assertEqual(segment.sum(), 1.0)
            self.assertEqual(segment[value], 1.0)
            offset += size

    def test_none_state_returns_none(self):
        self.assertIsNone(opponent_features(None))

    def test_opponent_features_do_not_modify_game_state(self):
        state = make_game_state(position=(3, 3), bombs_left=False)
        state['field'][5, 3] = CRATE
        state['others'] = [('other', 0, True, (6, 3))]
        original = copy.deepcopy(state)

        opponent_features(state)

        self.assertTrue(np.array_equal(state['field'], original['field']))
        self.assertEqual(state['others'], original['others'])
        self.assertEqual(state['self'], original['self'])


if __name__ == '__main__':
    unittest.main()

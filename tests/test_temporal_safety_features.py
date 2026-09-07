import copy
import unittest

import numpy as np

import settings as s
from agent_code.team_agent.temporal_safety_features import ACTIONS, _temporal_maps, temporal_safety_features
from tests.test_danger import CRATE, WALL, make_game_state


class TemporalSafetyFeaturesTestCase(unittest.TestCase):
    def test_move_features_have_fixed_objective_fields(self):
        analysis = temporal_safety_features(make_game_state())

        self.assertEqual(tuple(analysis.move_features), ACTIONS)
        self.assertTrue(analysis.move_features['UP'].legal)
        self.assertFalse(hasattr(analysis, 'best_action'))
        self.assertFalse(hasattr(analysis.move_features['UP'], 'recommended_direction'))

    def test_legal_move_can_be_unsafe_on_the_next_step(self):
        state = make_game_state(position=(2, 3), bombs=[((4, 3), 0)])

        features = temporal_safety_features(state).move_features['RIGHT']

        self.assertTrue(features.legal)
        self.assertFalse(features.safe_next)
        self.assertEqual(features.safe_horizon, 0)

    def test_crate_becomes_enterable_after_its_explosion_step(self):
        state = make_game_state(bombs=[((2, 3), 0)])
        state['field'][3, 3] = CRATE

        _, blocked = _temporal_maps(state, s.BOMB_TIMER + s.EXPLOSION_TIMER + 1, False)

        self.assertTrue(blocked[1, 3, 3])
        self.assertFalse(blocked[2, 3, 3])

    def test_bomb_tile_blocks_entry_until_the_step_after_explosion(self):
        state = make_game_state(bombs=[((3, 3), 0)])

        _, blocked = _temporal_maps(state, s.BOMB_TIMER + s.EXPLOSION_TIMER + 1, False)

        self.assertTrue(blocked[1, 3, 3])
        self.assertFalse(blocked[2, 3, 3])

    def test_other_agents_are_fixed_obstacles(self):
        state = make_game_state(position=(3, 3))
        state['others'] = [('other', 0, True, (3, 2))]

        features = temporal_safety_features(state).move_features['UP']

        self.assertFalse(features.legal)
        self.assertFalse(features.safe_next)

    def test_multiple_bombs_keep_a_temporal_escape_path(self):
        state = make_game_state(position=(1, 1), bombs=[((3, 1), 0), ((5, 3), 3)])

        features = temporal_safety_features(state).move_features['DOWN']

        self.assertTrue(features.safe_next)
        self.assertTrue(features.escape_exists)
        self.assertEqual(features.safe_horizon, s.BOMB_TIMER + s.EXPLOSION_TIMER + 1)

    def test_hypothetical_bomb_reports_escape_and_crates_hit(self):
        state = make_game_state(position=(3, 3))
        state['field'][4, 3] = CRATE
        state['field'][5, 3] = CRATE

        features = temporal_safety_features(state).bomb_features

        self.assertTrue(features.can_drop_bomb)
        self.assertTrue(features.escape_after_bomb)
        self.assertEqual(features.crates_hit, 2)
        self.assertGreater(features.escape_area_after_bomb, 0)

    def test_hypothetical_bomb_can_be_a_dead_end(self):
        state = make_game_state(position=(3, 3))
        state['field'][2, 3] = WALL
        state['field'][4, 3] = WALL
        state['field'][3, 2] = WALL
        state['field'][3, 4] = WALL

        features = temporal_safety_features(state).bomb_features

        self.assertTrue(features.can_drop_bomb)
        self.assertFalse(features.escape_after_bomb)
        self.assertEqual(features.escape_area_after_bomb, 0)

    def test_analysis_does_not_modify_game_state(self):
        state = make_game_state(position=(3, 3), bombs=[((4, 3), 2)])
        state['field'][3, 2] = CRATE
        original = copy.deepcopy(state)

        temporal_safety_features(state)

        self.assertTrue(np.array_equal(state['field'], original['field']))
        self.assertTrue(np.array_equal(state['explosion_map'], original['explosion_map']))
        self.assertEqual(state['bombs'], original['bombs'])
        self.assertEqual(state['self'], original['self'])

    def test_none_state_returns_none(self):
        self.assertIsNone(temporal_safety_features(None))


if __name__ == '__main__':
    unittest.main()

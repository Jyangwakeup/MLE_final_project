import copy
import unittest

import numpy as np

import settings as s
from agent_code.team_agent.danger import HORIZON, predict_danger


WALL = -1
CRATE = 1


def make_game_state(field=None, bombs=None, explosion_map=None, bombs_left=True, position=(3, 3)):
    """Build a game state with the environment's public state format."""
    if field is None:
        field = np.zeros((s.COLS, s.ROWS), dtype=int)
        field[0, :] = WALL
        field[-1, :] = WALL
        field[:, 0] = WALL
        field[:, -1] = WALL
    if bombs is None:
        bombs = []
    if explosion_map is None:
        explosion_map = np.zeros(field.shape, dtype=int)

    return {
        'round': 1,
        'step': 1,
        'field': field,
        'self': ('team_agent', 0, bombs_left, position),
        'others': [],
        'bombs': bombs,
        'coins': [],
        'explosion_map': explosion_map,
        'user_input': 'WAIT',
    }


class DangerTestCase(unittest.TestCase):
    def test_stone_wall_blocks_blast(self):
        state = make_game_state(bombs=[((2, 3), 0)])
        state['field'][4, 3] = WALL

        danger = predict_danger(state).danger

        self.assertTrue(danger[1, 3, 3])
        self.assertFalse(danger[1, 4, 3])
        self.assertFalse(danger[1, 5, 3])

    def test_crate_does_not_block_blast(self):
        state = make_game_state(bombs=[((2, 3), 0)])
        state['field'][3, 3] = CRATE

        danger = predict_danger(state).danger

        self.assertTrue(danger[1, 3, 3])
        self.assertTrue(danger[1, 4, 3])

    def test_zero_timer_bomb_explodes_at_first_future_step(self):
        danger = predict_danger(make_game_state(bombs=[((3, 3), 0)])).danger

        self.assertFalse(danger[0].any())
        self.assertTrue(danger[1, 3, 3])
        self.assertTrue(danger[2, 3, 3])
        self.assertFalse(danger[3, 3, 3])

    def test_countdown_bomb_explodes_after_timer_plus_one_steps(self):
        danger = predict_danger(make_game_state(bombs=[((3, 3), 3)])).danger

        self.assertFalse(danger[3, 3, 3])
        self.assertTrue(danger[4, 3, 3])
        self.assertTrue(danger[5, 3, 3])
        self.assertFalse(danger[6, 3, 3])

    def test_existing_explosion_map_marks_remaining_danger(self):
        explosion_map = np.zeros((s.COLS, s.ROWS), dtype=int)
        explosion_map[2, 4] = 2

        danger = predict_danger(make_game_state(explosion_map=explosion_map)).danger

        self.assertTrue(danger[1, 2, 4])
        self.assertTrue(danger[2, 2, 4])
        self.assertFalse(danger[3, 2, 4])

    def test_multiple_bombs_keep_the_union_of_danger_periods(self):
        bombs = [((2, 3), 0), ((4, 3), 3)]
        danger = predict_danger(make_game_state(bombs=bombs)).danger

        self.assertTrue(danger[1, 2, 2])
        self.assertFalse(danger[4, 2, 2])
        self.assertTrue(danger[4, 4, 2])
        self.assertTrue(danger[5, 4, 2])

    def test_blast_does_not_trigger_other_bombs_early(self):
        bombs = [((2, 3), 0), ((3, 3), 4)]
        danger = predict_danger(make_game_state(bombs=bombs)).danger

        self.assertFalse(danger[1, 3, 2])
        self.assertTrue(danger[5, 3, 2])

    def test_hypothetical_bomb_explodes_at_fifth_future_step(self):
        danger = predict_danger(make_game_state(), hypothetical_bomb=True).danger

        self.assertEqual(danger.shape, (HORIZON + 1, s.COLS, s.ROWS))
        self.assertFalse(danger[4, 3, 3])
        self.assertTrue(danger[5, 3, 3])
        self.assertTrue(danger[6, 3, 3])
        self.assertFalse(danger[7, 3, 3])

    def test_hypothetical_bomb_requires_an_available_bomb(self):
        danger = predict_danger(make_game_state(bombs_left=False), hypothetical_bomb=True).danger

        self.assertFalse(danger.any())

    def test_prediction_does_not_modify_game_state(self):
        state = make_game_state(bombs=[((3, 3), 0)])
        state['explosion_map'][2, 3] = 1
        original = copy.deepcopy(state)

        predict_danger(state, hypothetical_bomb=True)

        self.assertTrue(np.array_equal(state['field'], original['field']))
        self.assertTrue(np.array_equal(state['explosion_map'], original['explosion_map']))
        self.assertEqual(state['bombs'], original['bombs'])
        self.assertEqual(state['self'], original['self'])


if __name__ == '__main__':
    unittest.main()

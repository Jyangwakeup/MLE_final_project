from types import SimpleNamespace
import unittest

import torch

from agent_code.rainbow_lite_v6_agent.features import features_for_state, init_action_history
from agent_code.team_agent.feature_system import ACTIONS, get_feature_schema
from agent_code.team_agent.rewards import reward_from_events
from experiments.migrate_rainbow_v5_to_v6 import migrate
from tests.test_danger import make_game_state


class ContinuousV6Tests(unittest.TestCase):
    def test_tracks_named_opponent_motion_and_action_distance(self):
        owner = SimpleNamespace()
        init_action_history(owner)
        state = make_game_state(position=(3, 3))
        state.update(round=1, step=1)
        state["others"] = [("enemy", 0, True, (7, 3))]
        first = features_for_state(owner, state)
        moved = make_game_state(position=(3, 3))
        moved.update(round=1, step=2)
        moved["others"] = [("enemy", 0, True, (6, 3))]
        second = features_for_state(owner, moved)
        self.assertEqual(get_feature_schema("continuous-v6-opponent-tracking").vector_shape, (160,))
        self.assertEqual(second.vector.shape, (160,))
        right_delta = 140 + ACTIONS.index("RIGHT") * 2
        self.assertGreater(second.vector[right_delta], 0.0)
        self.assertEqual(second.vector[-7], 1.0)
        self.assertLess(second.vector[-3], 0.0)

    def test_r19_only_rewards_survivable_opponent_threat(self):
        state = make_game_state(position=(3, 3))
        state["others"] = [("enemy", 0, True, (5, 3))]
        wait = reward_from_events([], "r19_opponent_pressure", old_game_state=state, new_game_state=state, action="WAIT")
        bomb = reward_from_events([], "r19_opponent_pressure", old_game_state=state, new_game_state=state, action="BOMB")
        self.assertGreater(bomb, wait)

    def test_r20_does_not_reward_predicted_opponent_threat(self):
        state = make_game_state(position=(3, 3))
        state["others"] = [("enemy", 0, True, (5, 3))]
        wait = reward_from_events(
            [], "r20_outcome_credit", old_game_state=state,
            new_game_state=state, action="WAIT")
        bomb = reward_from_events(
            [], "r20_outcome_credit", old_game_state=state,
            new_game_state=state, action="BOMB")
        self.assertAlmostEqual(bomb, wait)

    def test_r20_does_not_reward_predicted_crates(self):
        state = make_game_state(position=(3, 3))
        state["field"][5, 3] = 1
        wait = reward_from_events(
            [], "r20_outcome_credit", old_game_state=state,
            new_game_state=state, action="WAIT")
        bomb = reward_from_events(
            [], "r20_outcome_credit", old_game_state=state,
            new_game_state=state, action="BOMB")
        self.assertAlmostEqual(bomb, wait)


if __name__ == "__main__":
    unittest.main()

import unittest

import events as e

from agent_code.dqn_agent.train import reward_from_events


class DQNRewardTestCase(unittest.TestCase):
    def test_objective_outcomes_and_time_cost_are_combined(self):
        events = [
            e.COIN_COLLECTED,
            e.KILLED_OPPONENT,
            e.CRATE_DESTROYED,
            e.CRATE_DESTROYED,
            e.INVALID_ACTION,
        ]
        self.assertAlmostEqual(reward_from_events(events), 6.29)

    def test_death_is_penalized_only_once(self):
        self.assertAlmostEqual(
            reward_from_events([e.KILLED_SELF, e.GOT_KILLED]),
            -10.01,
        )

    def test_movement_directions_have_no_supervised_action_reward(self):
        for movement in (e.MOVED_UP, e.MOVED_RIGHT, e.MOVED_DOWN, e.MOVED_LEFT):
            with self.subTest(movement=movement):
                self.assertAlmostEqual(reward_from_events([movement]), -0.01)


if __name__ == "__main__":
    unittest.main()

import unittest

import events as e

from agent_code.team_agent.rewards import REWARD_VERSION, reward_from_events


class SharedRewardTestCase(unittest.TestCase):
    def test_base_reward_combines_repeated_objective_events(self):
        events = [
            e.COIN_COLLECTED,
            e.KILLED_OPPONENT,
            e.CRATE_DESTROYED,
            e.CRATE_DESTROYED,
            e.INVALID_ACTION,
            e.SURVIVED_ROUND,
        ]

        self.assertEqual(REWARD_VERSION, "base-v1")
        self.assertAlmostEqual(reward_from_events(events), 7.19)

    def test_death_is_penalized_only_once(self):
        self.assertAlmostEqual(
            reward_from_events([e.KILLED_SELF, e.GOT_KILLED]),
            -10.01,
        )

    def test_unmapped_events_only_receive_step_cost(self):
        self.assertAlmostEqual(reward_from_events([e.MOVED_UP]), -0.01)


if __name__ == "__main__":
    unittest.main()

import unittest

import events as e

from agent_code.team_agent.rewards import (
    REWARD_SPECS,
    REWARD_VERSION,
    resolve_reward_spec,
    reward_from_events,
)


class SharedRewardTestCase(unittest.TestCase):
    def test_r1_is_the_canonical_objective_reward(self):
        self.assertEqual(REWARD_VERSION, "r1")
        self.assertEqual(resolve_reward_spec("r1"), {
            "step": -0.01,
            "coin_collected": 1.0,
            "killed_opponent": 5.0,
            "crate_destroyed": 0.2,
            "death": -10.0,
            "invalid_action": -0.1,
        })
        self.assertIn("r1_no_crate", REWARD_SPECS)

    def test_r1_combines_repeated_objective_events(self):
        events = [
            e.COIN_COLLECTED,
            e.KILLED_OPPONENT,
            e.CRATE_DESTROYED,
            e.CRATE_DESTROYED,
            e.INVALID_ACTION,
            e.SURVIVED_ROUND,
        ]

        self.assertAlmostEqual(reward_from_events(events), 6.29)

    def test_death_is_penalized_only_once(self):
        self.assertAlmostEqual(
            reward_from_events([e.KILLED_SELF, e.GOT_KILLED]),
            -10.01,
        )

    def test_unmapped_events_only_receive_step_cost(self):
        self.assertAlmostEqual(reward_from_events([e.MOVED_UP]), -0.01)

    def test_survival_has_no_terminal_bonus(self):
        self.assertAlmostEqual(reward_from_events([e.SURVIVED_ROUND]), -0.01)


if __name__ == "__main__":
    unittest.main()

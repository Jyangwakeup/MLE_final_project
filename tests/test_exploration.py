import unittest

from agent_code.dqn_agent import callbacks as dqn_callbacks
from agent_code.q_learning_agent import callbacks as q_callbacks
from agent_code.team_agent.exploration import (
    EXPLORATION_VERSION,
    epsilon_at,
    resolve_exploration_spec,
)


class ExplorationScheduleTestCase(unittest.TestCase):
    def test_both_learning_agents_use_the_shared_schedule_function(self):
        self.assertIs(q_callbacks.epsilon_at, epsilon_at)
        self.assertIs(dqn_callbacks.epsilon_at, epsilon_at)

    def test_formal_linear_schedule_has_specified_boundaries_and_midpoint(self):
        schedule = resolve_exploration_spec({
            "version": "linear-v1",
            "start": 1.0,
            "end": 0.05,
            "decay_action_steps": 80_000,
        })

        self.assertEqual(schedule["version"], EXPLORATION_VERSION)
        self.assertAlmostEqual(epsilon_at(0, schedule), 1.0)
        self.assertAlmostEqual(epsilon_at(40_000, schedule), 0.525)
        self.assertAlmostEqual(epsilon_at(80_000, schedule), 0.05)
        self.assertAlmostEqual(epsilon_at(3_000_000, schedule), 0.05)


if __name__ == "__main__":
    unittest.main()

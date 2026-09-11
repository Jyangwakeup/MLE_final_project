import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as e
import torch

from agent_code.dqn_agent.callbacks import INPUT_SIZE, setup
from agent_code.dqn_agent.model import DQN
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


class DQNTrainingTaskTestCase(unittest.TestCase):
    def test_new_task_resets_progress_but_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "dqn-model.pt"
            model = DQN(INPUT_SIZE, 6)
            checkpoint = model.checkpoint()
            checkpoint.update({"action_steps": 123, "training_task": "task1"})
            torch.save(checkpoint, model_file)
            expected_parameter = next(model.policy.parameters()).detach().clone()
            agent = SimpleNamespace(train=True, logger=Mock())

            with (
                patch.dict(os.environ, {"BOMBERMAN_TRAINING_TASK": "task2"}),
                patch("agent_code.dqn_agent.callbacks.MODEL_FILE", model_file),
            ):
                setup(agent)

        self.assertEqual(agent.action_steps, 0)
        self.assertEqual(agent.training_task, "task2")
        self.assertTrue(torch.equal(
            next(agent.model.policy.parameters()).detach(),
            expected_parameter,
        ))


if __name__ == "__main__":
    unittest.main()

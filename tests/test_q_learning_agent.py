import os
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_code.q_learning_agent.callbacks import setup
from agent_code.q_learning_agent.features import legal_actions
from tests.test_danger import make_game_state


class QLearningAgentConfigurationTestCase(unittest.TestCase):
    def test_terminal_setting_disables_bomb_actions(self):
        agent = SimpleNamespace(train=True, logger=Mock())

        with patch.dict(os.environ, {"Q_LEARNING_ALLOW_BOMB": "false"}):
            setup(agent)

        self.assertFalse(agent.allow_bomb)
        self.assertFalse(legal_actions(
            make_game_state(bombs_left=True),
            allow_bomb=agent.allow_bomb,
        )[-1])

    def test_bombs_are_allowed_by_default(self):
        agent = SimpleNamespace(train=True, logger=Mock())

        with patch.dict(os.environ, {}, clear=True):
            setup(agent)

        self.assertTrue(agent.allow_bomb)
        self.assertTrue(legal_actions(
            make_game_state(bombs_left=True),
            allow_bomb=agent.allow_bomb,
        )[-1])

    def test_new_task_resets_progress_but_keeps_q_table(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "q-table.pkl"
            expected_table = {("known",): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
            with model_file.open("wb") as file:
                pickle.dump({
                    "q_table": expected_table,
                    "training_steps": 123,
                    "training_task": "task1",
                }, file)
            agent = SimpleNamespace(train=True, logger=Mock())

            with (
                patch.dict(os.environ, {"BOMBERMAN_TRAINING_TASK": "task2"}),
                patch("agent_code.q_learning_agent.callbacks.MODEL_FILE", model_file),
            ):
                setup(agent)

        self.assertEqual(agent.q_table, expected_table)
        self.assertEqual(agent.training_steps, 0)
        self.assertEqual(agent.training_task, "task2")


if __name__ == "__main__":
    unittest.main()

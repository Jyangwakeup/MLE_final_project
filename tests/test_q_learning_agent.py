import os
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_code.q_learning_agent.callbacks import FEATURE_VERSION, _features_for, setup
from agent_code.q_learning_agent.features import legal_actions
from tests.test_danger import make_game_state


class QLearningAgentConfigurationTestCase(unittest.TestCase):
    def test_shared_features_are_cached_per_round_step(self):
        agent = SimpleNamespace(
            _feature_cache_key=None,
            _feature_cache_value=None,
        )
        first_state = make_game_state()
        first_state.update(round=1, step=1)
        second_state = make_game_state(position=(3, 4))
        second_state.update(round=1, step=2)

        with patch(
            "agent_code.q_learning_agent.callbacks.features_for_state",
            wraps=__import__(
                "agent_code.q_learning_agent.callbacks",
                fromlist=["features_for_state"],
            ).features_for_state,
        ) as extractor:
            first = _features_for(agent, first_state)
            repeated = _features_for(agent, first_state)
            second = _features_for(agent, second_state)

        self.assertIs(first, repeated)
        self.assertIsNot(first, second)
        self.assertEqual(extractor.call_count, 2)

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

    def test_new_task_keeps_progress_and_q_table(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "q-table.pkl"
            expected_table = {("known",): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
            with model_file.open("wb") as file:
                pickle.dump({
                    "feature_version": FEATURE_VERSION,
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
        self.assertEqual(agent.training_steps, 123)
        self.assertEqual(agent.training_task, "task2")

    def test_explicit_evaluation_checkpoint_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            agent = SimpleNamespace(train=False, logger=Mock())

            with patch.dict(os.environ, {"BOMBERMAN_CHECKPOINT": str(missing)}):
                with self.assertRaises(FileNotFoundError):
                    setup(agent)


if __name__ == "__main__":
    unittest.main()

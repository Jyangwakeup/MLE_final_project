import os
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_code.q_learning_agent.callbacks import (
    FEATURE_ID,
    FEATURE_VERSION,
    HYPERPARAMETERS,
    ResumeCompatibilityError,
    _features_for,
    setup,
)
from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.rewards import resolve_reward_spec
from agent_code.q_learning_agent.features import legal_actions
from agent_code.q_learning_agent.train import (
    end_of_round, game_events_occurred, setup_training,
)
from agent_code.learning_common.temporal_reward import temporal_reward_context
from tests.test_danger import make_game_state


class QLearningAgentConfigurationTestCase(unittest.TestCase):
    def test_last_step_is_updated_once_as_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, logger=Mock())
            with (
                patch.dict(os.environ, {
                    "BOMBERMAN_CHECKPOINT": str(Path(directory) / "final.pkl"),
                    "BOMBERMAN_RUN_DIR": directory,
                }, clear=True),
            ):
                setup(agent)
                setup_training(agent)
                state = make_game_state()
                state.update(round=1, step=4)
                next_state = make_game_state(position=(3, 4))
                next_state.update(round=1, step=5)
                game_events_occurred(agent, state, "WAIT", next_state, [])
                end_of_round(agent, state, "WAIT", [])

            self.assertEqual(agent.training_steps, 0)
            self.assertEqual(len(agent.q_table), 1)
            values = next(iter(agent.q_table.values()))
            self.assertAlmostEqual(float(values[4]), -0.0015)
            with (Path(directory) / "training.csv").open() as file:
                row = next(__import__("csv").DictReader(file))
            self.assertAlmostEqual(float(row["reward"]), -0.01)

    def test_legacy_reward_checkpoint_is_frozen_only(self):
        legacy_spec = {
            "step": -0.01,
            "coin_collected": 1.0,
            "killed_opponent": 5.0,
            "crate_destroyed": 0.2,
            "death": -10.0,
            "invalid_action": -0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "legacy.pkl"
            with model_file.open("wb") as file:
                pickle.dump({
                    "checkpoint_schema": "training-resume-v3",
                    "feature_id": "discrete-v1",
                    "feature_version": "v1",
                    "feature_schema": feature_schema_contract("discrete-v1"),
                    "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
                    "reward_version": "r1",
                    "reward_spec": legacy_spec,
                    "q_table": {},
                    "training_steps": 1,
                    "agent_rng_state": random.Random(0).getstate(),
                }, file)

            frozen = SimpleNamespace(train=False, logger=Mock())
            with patch.dict(
                os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}, clear=True,
            ):
                setup(frozen)
            self.assertEqual(frozen.checkpoint_reward_version, "r1")

            resumed = SimpleNamespace(train=True, logger=Mock())
            with patch.dict(
                os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}, clear=True,
            ):
                with self.assertRaisesRegex(
                    ResumeCompatibilityError, "legacy checkpoint schema",
                ):
                    setup(resumed)

    def test_setup_uses_runner_agent_seed_and_exploration_schedule(self):
        specification = (
            '{"version":"linear-v1","start":1.0,"end":0.05,'
            '"decay_action_steps":1920000}'
        )
        agent = SimpleNamespace(train=True, logger=Mock())

        with patch.dict(os.environ, {
            "BOMBERMAN_AGENT_SEED": "17",
            "BOMBERMAN_EXPLORATION_SPEC": specification,
        }, clear=True):
            setup(agent)

        self.assertEqual(agent.agent_seed, 17)
        self.assertEqual(agent.rng.getstate(), random.Random(17).getstate())
        self.assertEqual(agent.exploration_spec["decay_action_steps"], 1_920_000)
    @staticmethod
    def _checkpoint(reward_id="r4_anti_oscillation"):
        return {
            "feature_id": FEATURE_ID,
            "feature_version": None,
            "feature_schema": feature_schema_contract(FEATURE_ID),
            "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
            "reward_id": reward_id,
            "reward_version": reward_id,
            "q_table": {},
            "training_steps": 0,
        }

    def test_reward_contract_is_inferred_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "q-table.pkl"
            with model_file.open("wb") as file:
                pickle.dump(self._checkpoint(), file)
            agent = SimpleNamespace(train=False, logger=Mock())
            with patch.dict(
                os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}, clear=True
            ):
                setup(agent)
        self.assertEqual(agent.feature_id, "discrete-q-v2")
        self.assertEqual(agent.reward_id, "r4_anti_oscillation")

    def test_explicit_reward_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "q-table.pkl"
            with model_file.open("wb") as file:
                pickle.dump(self._checkpoint(), file)
            agent = SimpleNamespace(train=False, logger=Mock())
            with patch.dict(os.environ, {
                "BOMBERMAN_CHECKPOINT": str(model_file),
                "BOMBERMAN_REWARD_ID": "r2_balanced",
            }, clear=True):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    setup(agent)

    def test_only_repeated_nonprogressing_reversal_is_flagged(self):
        agent = SimpleNamespace(move_history=["LEFT", "RIGHT"])
        old_state = make_game_state(position=(3, 3))
        new_state = make_game_state(position=(2, 3))
        old_state["coins"] = new_state["coins"] = [(8, 3)]
        agent.previous_action = "RIGHT"
        agent.stationary_streak = 0
        self.assertTrue(temporal_reward_context(
            agent, "LEFT", old_state, new_state, [])["repeated_oscillation"])

        agent.move_history = ["RIGHT"]
        self.assertFalse(temporal_reward_context(
            agent, "LEFT", old_state, new_state, [])["repeated_oscillation"])

    def test_idle_streak_ignores_first_wait_and_counts_repetition(self):
        agent = SimpleNamespace(stationary_streak=0)
        old_state = make_game_state(position=(3, 3))
        new_state = make_game_state(position=(3, 3))
        old_state["coins"] = new_state["coins"] = [(8, 3)]
        agent.previous_action = None
        agent.move_history = []
        first = temporal_reward_context(
            agent, "WAIT", old_state, new_state, [])["idle_streak"]
        agent.stationary_streak = first
        second = temporal_reward_context(
            agent, "WAIT", old_state, new_state, [])["idle_streak"]
        self.assertEqual((first, second), (1, 2))
        self.assertEqual(temporal_reward_context(
            agent, "RIGHT", old_state, new_state, [])["idle_streak"], 0)

    def test_shared_features_are_cached_per_round_step(self):
        agent = SimpleNamespace(
            _feature_cache_key=None,
            _feature_cache_value=None,
            feature_id=FEATURE_ID,
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

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "new.pkl"
            with (
                patch.dict(os.environ, {"Q_LEARNING_ALLOW_BOMB": "false"}),
                patch("agent_code.q_learning_agent.callbacks.MODEL_FILE", missing),
            ):
                setup(agent)

        self.assertFalse(agent.allow_bomb)
        self.assertFalse(legal_actions(
            make_game_state(bombs_left=True),
            allow_bomb=agent.allow_bomb,
        )[-1])

    def test_bombs_are_allowed_by_default(self):
        agent = SimpleNamespace(train=True, logger=Mock())

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "new.pkl"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("agent_code.q_learning_agent.callbacks.MODEL_FILE", missing),
            ):
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
                    "feature_id": FEATURE_ID,
                    "feature_version": None,
                    "feature_schema": feature_schema_contract(FEATURE_ID),
                    "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
                    "reward_version": "r1",
                    "reward_id": "r1",
                    "reward_spec": resolve_reward_spec("r1"),
                    "checkpoint_schema": "training-resume-v5",
                    "network_spec": None,
                    "hyperparameters": HYPERPARAMETERS,
                    "agent_rng_state": random.Random(0).getstate(),
                    "q_table": expected_table,
                    "training_steps": 123,
                    "total_action_steps": 123,
                    "stage_action_steps": 123,
                    "safe_exploration_decisions": 17,
                    "safe_exploration_fallbacks": 2,
                    "n_step_state": {"n_step": 1, "gamma": 0.95, "pending": []},
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
        self.assertEqual(agent.total_action_steps, 123)
        self.assertEqual(agent.stage_action_steps, 0)
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

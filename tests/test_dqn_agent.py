import os
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as e
import numpy as np
import torch

from agent_code.dqn_agent.callbacks import (
    FEATURE_VERSION,
    HYPERPARAMETERS,
    INPUT_SIZE,
    _features_for,
    act,
    legal_actions,
    setup,
    state_to_features,
    network_spec,
)
from agent_code.dqn_agent.model import DQN, Transition
from agent_code.dqn_agent.train import (
    end_of_round,
    game_events_occurred,
    reward_from_events,
    setup_training,
)
from agent_code.q_learning_agent.features import features_for_state as q_features_for_state
from agent_code.dqn_agent.features import FEATURE_DIM
from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.rewards import (
    resolve_reward_spec,
    reward_from_events as shared_reward_from_events,
)
from tests.test_danger import make_game_state


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
            model_file = Path(directory) / "legacy.pt"
            checkpoint = DQN(INPUT_SIZE, 6).checkpoint()
            checkpoint.update({
                "checkpoint_schema": "training-resume-v3",
                "feature_id": "discrete-q-v2",
                "feature_version": None,
                "feature_schema": feature_schema_contract("discrete-q-v2"),
                "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
                "reward_id": "r1",
                "reward_version": "r1",
                "reward_spec": legacy_spec,
            })
            torch.save(checkpoint, model_file)

            frozen = SimpleNamespace(train=False, logger=Mock())
            with patch.dict(
                os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}, clear=True,
            ):
                setup(frozen)
            self.assertEqual(frozen.reward_id, "r1")

            resumed = SimpleNamespace(train=True, logger=Mock())
            with patch.dict(
                os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}, clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "legacy checkpoint schema"):
                    setup(resumed)

    def test_setup_passes_runner_seed_and_exploration_schedule_to_model(self):
        agent = SimpleNamespace(train=True, logger=Mock())
        specification = (
            '{"version":"linear-v1","start":1.0,"end":0.05,'
            '"decay_action_steps":80000}'
        )

        with (
            patch.dict(os.environ, {
                "BOMBERMAN_AGENT_SEED": "17",
                "BOMBERMAN_EXPLORATION_SPEC": specification,
            }, clear=True),
            patch("agent_code.dqn_agent.callbacks.DQN") as model_type,
        ):
            setup(agent)

        model_type.assert_called_once_with(
            INPUT_SIZE, 6, seed=17, device="cpu", training_task=None,
            retention_spec={
                "parent_fraction": 0.5, "distillation_weight": 1.0,
                "temperature": 1.0, "per_task_capacity": 20_000,
                "current_warmup": 2_000,
            },
        )
        self.assertEqual(agent.agent_seed, 17)
        self.assertEqual(agent.rng.getstate(), random.Random(17).getstate())
        self.assertEqual(agent.exploration_spec["decay_action_steps"], 80_000)

    def test_shared_flag_disables_bombs_during_training_and_evaluation(self):
        for training in (True, False):
            with self.subTest(training=training):
                agent = SimpleNamespace(
                    train=training,
                    logger=Mock(),
                    rng=Mock(),
                    action_steps=0,
                    allow_bomb=False,
                    _feature_cache_key=None,
                    _feature_cache_value=None,
                    model=Mock(),
                    feature_id="discrete-q-v2",
                )
                agent.model.q_values.return_value = np.array(
                    [0, 0, 0, 0, 0, 100], dtype=np.float32
                )
                agent.rng.random.return_value = 1.0
                agent.rng.choice.side_effect = lambda choices: choices[0]

                action = act(agent, make_game_state(bombs_left=True))

                self.assertNotEqual(action, "BOMB")

    def test_new_task_keeps_progress_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "dqn-model.pt"
            model = DQN(INPUT_SIZE, 6)
            checkpoint = model.checkpoint()
            checkpoint.update({
                "feature_id": "discrete-q-v2",
                "feature_version": FEATURE_VERSION,
                "feature_schema": feature_schema_contract("discrete-q-v2"),
                "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
                "reward_id": "r1",
                "reward_version": "r1",
                "checkpoint_schema": "training-resume-v5",
                "network_spec": network_spec(INPUT_SIZE),
                "hyperparameters": HYPERPARAMETERS,
                "feature_version": FEATURE_VERSION,
                "reward_version": "r1",
                "reward_spec": resolve_reward_spec("r1"),
                "action_steps": 123,
                "total_action_steps": 123,
                "stage_action_steps": 123,
                "safe_exploration_decisions": 17,
                "safe_exploration_fallbacks": 2,
                "n_step_state": {"n_step": 1, "gamma": 0.95, "pending": []},
                "training_task": "task1",
                "agent_rng_state": random.Random(0).getstate(),
            })
            torch.save(checkpoint, model_file)
            expected_parameter = next(model.policy.parameters()).detach().clone()
            agent = SimpleNamespace(train=True, logger=Mock())

            with (
                patch.dict(os.environ, {"BOMBERMAN_TRAINING_TASK": "task2"}),
                patch("agent_code.dqn_agent.callbacks.MODEL_FILE", model_file),
            ):
                setup(agent)

        self.assertEqual(agent.action_steps, 123)
        self.assertEqual(agent.total_action_steps, 123)
        self.assertEqual(agent.stage_action_steps, 0)
        self.assertEqual(agent.training_task, "task2")
        self.assertTrue(torch.equal(
            next(agent.model.policy.parameters()).detach(),
            expected_parameter,
        ))

    def test_incompatible_checkpoint_is_rejected_during_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "old-model.pt"
            checkpoint = DQN(INPUT_SIZE, 6).checkpoint()
            checkpoint["feature_version"] = "old-features"
            torch.save(checkpoint, model_file)
            agent = SimpleNamespace(train=False, logger=Mock())

            with patch.dict(os.environ, {"BOMBERMAN_CHECKPOINT": str(model_file)}):
                with self.assertRaisesRegex(ValueError, "feature ID"):
                    setup(agent)


class DQNSharedFeatureTestCase(unittest.TestCase):
    def test_vector_and_legal_mask_come_from_team_features(self):
        state = make_game_state(bombs_left=True)
        vector = state_to_features(state)
        legal = legal_actions(state)

        self.assertEqual(INPUT_SIZE, FEATURE_DIM)
        self.assertEqual(vector.shape, (50,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(legal.shape, (6,))
        self.assertTrue(legal[-1])

        dqn_features = _features_for(
            SimpleNamespace(_feature_cache_key=None, _feature_cache_value=None),
            state,
        )
        q_features = q_features_for_state(state)
        np.testing.assert_array_equal(dqn_features.vector, q_features.vector)
        np.testing.assert_array_equal(dqn_features.legal_mask, q_features.legal_mask)

    def test_dqn_reward_is_the_shared_reward_function(self):
        self.assertIs(reward_from_events, shared_reward_from_events)

    def test_shared_features_are_cached_per_round_step(self):
        agent = SimpleNamespace(
            _feature_cache_key=None, _feature_cache_value=None,
            feature_id="discrete-q-v2",
        )
        state = make_game_state()
        state.update(round=1, step=1)
        self.assertIs(_features_for(agent, state), _features_for(agent, state))


class DQNTransitionTestCase(unittest.TestCase):
    def _agent(self):
        model = Mock()
        model.checkpoint.return_value = {}
        return SimpleNamespace(
            model=model,
            model_file=Path(self.directory) / "checkpoint.pt",
            action_steps=1,
            rng=random.Random(0),
            training_task="task1",
            logger=Mock(),
            _feature_cache_key=None,
            _feature_cache_value=None,
            feature_id="discrete-q-v2",
            reward_id="r1",
        )

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = self.temp_directory.name

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_last_step_is_observed_once_as_terminal(self):
        agent = self._agent()
        setup_training(agent)
        old_state = make_game_state()
        old_state.update(round=1, step=4)
        new_state = make_game_state(position=(3, 4))
        new_state.update(round=1, step=5)

        recorded_rewards = []
        with patch(
            "agent_code.dqn_agent.train._append_training_metrics",
            side_effect=lambda owner, _state: recorded_rewards.append(
                owner.round_reward),
        ):
            game_events_occurred(agent, old_state, "WAIT", new_state, [e.WAITED])
            end_of_round(agent, old_state, "WAIT", [e.WAITED, e.SURVIVED_ROUND])

        agent.model.observe.assert_called_once()
        transition = agent.model.observe.call_args.args[0]
        self.assertIsInstance(transition, Transition)
        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_state)
        self.assertAlmostEqual(transition.reward, -0.01)
        self.assertEqual(recorded_rewards, [-0.01])


class DQNCheckpointTestCase(unittest.TestCase):
    def test_checkpoint_restores_replay_sampler_and_torch_rng(self):
        model = DQN(INPUT_SIZE, 6, seed=17, batch_size=2, warmup=2)
        transition = Transition(
            np.zeros(INPUT_SIZE, dtype=np.float32), 0, 1.0,
            np.ones(INPUT_SIZE, dtype=np.float32), False,
            np.ones(6, dtype=bool),
        )
        model.replay.append(transition)
        checkpoint = model.checkpoint()
        expected_torch_rng = checkpoint["torch_rng_state"].clone()
        expected_sampler_state = checkpoint["replay"]["rng_state"]

        restored = DQN(INPUT_SIZE, 6, seed=999, batch_size=2, warmup=2)
        restored.load_checkpoint(checkpoint, training=True)

        self.assertEqual(len(restored.replay), 1)
        self.assertEqual(restored.replay.random.getstate(), expected_sampler_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_rng))

        next_transition = Transition(
            np.ones(INPUT_SIZE, dtype=np.float32), 1, -0.5,
            np.zeros(INPUT_SIZE, dtype=np.float32), False,
            np.ones(6, dtype=bool),
        )
        original_loss = model.observe(next_transition)
        restored_loss = restored.observe(next_transition)
        self.assertAlmostEqual(original_loss, restored_loss)
        original_parameters = list(model.policy.parameters())
        restored_parameters = list(restored.policy.parameters())
        self.assertEqual(len(original_parameters), len(restored_parameters))
        for original, resumed in zip(original_parameters, restored_parameters):
            self.assertTrue(torch.equal(original, resumed))
        for name, value in model.target.state_dict().items():
            self.assertTrue(torch.equal(value, restored.target.state_dict()[name]))
        self.assertEqual(model.updates, restored.updates)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_checkpoint_resumes_the_next_update_and_loads_on_cpu(self):
        model = DQN(INPUT_SIZE, 6, seed=17, batch_size=2, warmup=2, device="cuda")
        self.assertTrue(next(model.policy.parameters()).is_cuda)
        first = Transition(
            np.zeros(INPUT_SIZE, dtype=np.float32), 0, 1.0,
            np.ones(INPUT_SIZE, dtype=np.float32), False,
            np.ones(6, dtype=bool),
        )
        model.replay.append(first)
        checkpoint = model.checkpoint()
        self.assertEqual(checkpoint["training_device_type"], "cuda")
        self.assertIn("cuda_rng_state_all", checkpoint)

        restored = DQN(INPUT_SIZE, 6, seed=999, batch_size=2, warmup=2, device="cuda")
        restored.load_checkpoint(checkpoint, training=True)
        second = Transition(
            np.ones(INPUT_SIZE, dtype=np.float32), 1, -0.5,
            np.zeros(INPUT_SIZE, dtype=np.float32), False,
            np.ones(6, dtype=bool),
        )
        self.assertAlmostEqual(model.observe(second), restored.observe(second))
        for original, resumed in zip(
            model.policy.parameters(), restored.policy.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(original, resumed))

        cpu_model = DQN(INPUT_SIZE, 6, seed=0, device="cpu")
        cpu_model.load_checkpoint(checkpoint, training=False)
        self.assertFalse(next(cpu_model.policy.parameters()).is_cuda)
        with self.assertRaisesRegex(ValueError, "training device"):
            cpu_model.load_checkpoint(checkpoint, training=True)


if __name__ == "__main__":
    unittest.main()

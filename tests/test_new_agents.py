import logging
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agent_code.learning_common.neural import double_dqn_next_values
from agent_code.learning_common.replay import (
    CHECKPOINT_FORMAT as REPLAY_CHECKPOINT_FORMAT,
    ReplayBuffer,
    Transition as ReplayTransition,
    decode_board,
    encode_board,
)
from agent_code.learning_common.runtime import effective_legal_mask, epsilon_at
from agent_code.learning_common.neural_agent import act_neural
from agent_code.team_agent.rewards import resolve_reward_spec
from agent_code.double_q_compact_agent import callbacks as dq_callbacks
from agent_code.double_q_compact_agent import train as dq_train
from agent_code.double_q_compact_agent.features import canonical_legal_mask
from agent_code.double_dqn_continuous_agent import features as continuous_features
from agent_code.double_dqn_continuous_agent import callbacks as continuous_callbacks
from agent_code.double_dqn_continuous_agent import train as continuous_train
from agent_code.cnn_double_dqn_agent import features as board_features
from agent_code.cnn_double_dqn_agent import callbacks as cnn_callbacks
from agent_code.cnn_double_dqn_agent import train as cnn_train
from agent_code.cnn_double_dqn_agent.model import BoardQNetwork, build_learner as build_cnn
from agent_code.hybrid_dueling_double_dqn_agent import features as hybrid_features
from agent_code.hybrid_dueling_double_dqn_agent import callbacks as hybrid_callbacks
from agent_code.hybrid_dueling_double_dqn_agent import train as hybrid_train
from agent_code.hybrid_dueling_double_dqn_agent.model import HybridDuelingQNetwork
from experiments.resume import (
    commit_training_snapshot, load_training_snapshot, materialize_learner_checkpoint,
)


def game_state():
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    field[2::2, 2::2] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "coins": [(3, 1)], "bombs": [],
        "explosion_map": np.zeros_like(field),
        "self": ("me", 0, True, (1, 1)), "others": [],
    }


class FeatureAdapterTests(unittest.TestCase):
    def test_new_feature_adapters(self):
        state = game_state()
        compact = dq_callbacks.features_for_state(state)
        continuous = continuous_features.features_for_state(state)
        board = board_features.features_for_state(state)
        hybrid = hybrid_features.features_for_state(state)
        self.assertEqual(len(compact.state_key), 12)
        self.assertEqual(continuous.vector.shape, (84,))
        self.assertEqual(board.board.shape, (12, 17, 17))
        self.assertEqual(hybrid.board.shape, (12, 17, 17))
        self.assertEqual(hybrid.vector.shape, (70,))
        np.testing.assert_array_equal(board.board, hybrid.board)
        np.testing.assert_array_equal(continuous.vector[:70], hybrid.vector)

    def test_canonical_action_mapping_is_a_bijection(self):
        features = dq_callbacks.features_for_state(game_state())
        canonical = canonical_legal_mask(features)
        for world, mapped in enumerate(features.action_transform.world_to_canonical):
            self.assertEqual(
                features.action_transform.canonical_to_world[mapped], world)
            self.assertEqual(canonical[mapped], features.legal_mask[world])

    def test_board_codec_is_exact(self):
        board = board_features.features_for_state(game_state()).board
        encoded = encode_board(board)
        np.testing.assert_array_equal(decode_board(encoded), board)
        self.assertEqual(encoded["packed"].nbytes + encoded["timer"].nbytes, 867)

    def test_columnar_replay_checkpoint_round_trip_is_exact(self):
        features = hybrid_features.features_for_state(game_state())
        states = {
            "vector": features.vector,
            "board": features.board,
            "hybrid": (features.board, features.vector),
        }
        for kind, state in states.items():
            with self.subTest(kind=kind):
                source = ReplayBuffer(8, 17, kind)
                source.append(ReplayTransition(
                    state, 2, 0.123456789, state, False,
                    np.array([True, False, True, False, True, False])))
                source.append(ReplayTransition(
                    state, 4, -7.01, None, True, None))
                payload = source.state_dict()
                self.assertEqual(payload["format"], REPLAY_CHECKPOINT_FORMAT)
                self.assertNotIn("transitions", payload)
                restored = ReplayBuffer(8, 99, kind)
                restored.load_state_dict(payload)
                first = source.sample(2)
                second = restored.sample(2)
                self.assertEqual(
                    [(item.action, item.reward, item.done) for item in first],
                    [(item.action, item.reward, item.done) for item in second])
                for expected, actual in zip(first, second):
                    if kind == "hybrid":
                        np.testing.assert_array_equal(expected.state[0], actual.state[0])
                        np.testing.assert_array_equal(expected.state[1], actual.state[1])
                    else:
                        np.testing.assert_array_equal(expected.state, actual.state)

    def test_curriculum_mask_is_separate_from_physical_mask(self):
        physical = np.ones(6, dtype=bool)
        effective = effective_legal_mask(physical, dq_callbacks.ACTIONS, False)
        self.assertTrue(physical[-1])
        self.assertFalse(effective[-1])
        self.assertTrue(effective[-2])

    def test_wait_can_be_the_only_physical_action(self):
        state = game_state()
        x, y = state["self"][3]
        for xx, yy in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            state["field"][xx, yy] = -1
        state["self"] = ("me", 0, False, (x, y))
        features = continuous_features.features_for_state(state)
        np.testing.assert_array_equal(
            features.legal_mask, np.array([False, False, False, False, True, False]))


class AlgorithmTests(unittest.TestCase):
    def test_epsilon_schedule_endpoints(self):
        parameters = dq_callbacks.HYPERPARAMETERS
        self.assertEqual(epsilon_at(0, parameters), 1.0)
        self.assertEqual(epsilon_at(80_000, parameters), 0.05)
        self.assertEqual(epsilon_at(160_000, parameters), 0.05)

    def test_neural_action_uses_runner_exploration_spec(self):
        class RNG:
            def random(self): return 0.5
            def choice(self, values): return values[0]

        owner = SimpleNamespace(
            train=True, action_steps=80_000, rng=RNG(),
            curriculum_allows_bomb=True,
            exploration_spec={
                "version": "linear-v1", "start": 1.0, "end": 0.05,
                "decay_action_steps": 160_000,
            },
            model=SimpleNamespace(
                q_values=lambda state: np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)),
            _feature_cache_key=None, _feature_cache_value=None,
        )
        features = SimpleNamespace(
            legal_mask=np.ones(6, dtype=bool), vector=np.zeros(1, dtype=np.float32))
        action = act_neural(
            owner, game_state(), actions=continuous_callbacks.ACTIONS,
            hyperparameters=continuous_callbacks.HYPERPARAMETERS,
            extractor=lambda state: features, state_value=lambda value: value.vector,
        )
        self.assertEqual(action, "UP")
        self.assertEqual(owner.action_steps, 80_001)

    def test_continuous_frozen_act_advances_and_resets_history(self):
        owner = SimpleNamespace(
            train=False, previous_action=None, reward_previous_position=None,
            reward_previous_coin_target=None,
        )
        first = game_state()
        second = game_state()
        second["step"] = 2
        next_round = game_state()
        next_round["round"] = 2
        observed = []

        def choose(owner, state, **kwargs):
            observed.append(kwargs["extractor"](state).vector.copy())
            return ("RIGHT", "DOWN", "LEFT")[len(observed) - 1]

        with patch(
            "agent_code.double_dqn_continuous_agent.callbacks.act_neural",
            side_effect=choose,
        ):
            continuous_callbacks.act(owner, first)
            self.assertEqual(owner.previous_action, "RIGHT")
            self.assertEqual(owner.reward_previous_position, (1, 1))
            self.assertEqual(owner.reward_previous_coin_target, (3, 1))

            continuous_callbacks.act(owner, second)
            self.assertEqual(observed[1][70 + 1], 1.0)

            continuous_callbacks.act(owner, next_round)
            self.assertEqual(observed[2][70 + 6], 1.0)

    def test_double_dqn_selects_with_policy_and_evaluates_with_target(self):
        policy = torch.tensor([[0.0, 9.0, 4.0], [8.0, 7.0, 6.0]])
        target = torch.tensor([[100.0, 2.0, 50.0], [3.0, 20.0, 30.0]])
        legal = torch.tensor([[True, True, True], [False, True, False]])
        values = double_dqn_next_values(policy, target, legal)
        torch.testing.assert_close(values, torch.tensor([2.0, 20.0]))

    def test_double_q_cross_evaluation_and_terminal(self):
        class RNG:
            def random(self): return 0.0
            def choice(self, values): return values[0]
        owner = SimpleNamespace(
            rng=RNG(),
            q_table_a={(2,): np.zeros(6, dtype=np.float32)},
            q_table_b={(2,): np.array([7, 8, 9, 10, 11, 12], dtype=np.float32)},
        )
        owner.q_table_a[(2,)][1] = 5.0
        transition = dq_train.Transition(
            (1,), 0, 1.0, (2,), False,
            np.array([False, True, False, False, False, False]),
        )
        dq_train.update_double_q(owner, transition)
        expected_target = 1.0 + 0.95 * 8.0
        self.assertAlmostEqual(owner.q_table_a[(1,)][0], 0.1 * expected_target, places=6)
        terminal = dq_train.Transition((3,), 2, -2.0, None, True, None)
        dq_train.update_double_q(owner, terminal)
        self.assertAlmostEqual(owner.q_table_a[(3,)][2], -0.2)

    def test_cnn_and_hybrid_batch_shapes(self):
        self.assertEqual(BoardQNetwork()(torch.zeros(3, 12, 17, 17)).shape, (3, 6))
        network = HybridDuelingQNetwork()
        output = network(torch.zeros(3, 12, 17, 17), torch.zeros(3, 70))
        self.assertEqual(output.shape, (3, 6))
        build_cnn(0, {
            "gamma": .95, "learning_rate": 2e-4, "batch_size": 2,
            "replay_capacity": 2, "warmup": 2, "target_sync_interval": 2,
            "gradient_clip": 10, "epsilon_start": 1, "epsilon_end": .05,
            "epsilon_decay_action_steps": 80_000,
        })
        self.assertEqual(torch.get_num_threads(), 1)


class CheckpointTests(unittest.TestCase):
    def _self(self, train):
        return SimpleNamespace(train=train, logger=logging.getLogger("new-agent-test"))

    def test_double_q_checkpoint_round_trip_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl"
            env = {
                "BOMBERMAN_CHECKPOINT": str(checkpoint),
                "BOMBERMAN_FEATURE_ID": "discrete-compact-v1",
                "BOMBERMAN_REWARD_ID": "r1",
                "BOMBERMAN_TRAINING_TASK": "coin_navigation",
            }
            with patch.dict(os.environ, env, clear=False):
                learner = self._self(True)
                dq_callbacks.setup(learner)
                dq_train.setup_training(learner)
                dq_train.end_of_round(learner, game_state(), "WAIT", [])
                loaded = self._self(False)
                dq_callbacks.setup(loaded)
            with checkpoint.open("rb") as file:
                payload = pickle.load(file)
            payload["feature_id"] = "discrete-v1"
            with checkpoint.open("wb") as file:
                pickle.dump(payload, file)
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaisesRegex(ValueError, "feature"):
                    dq_callbacks.setup(self._self(False))

    def test_neural_checkpoints_round_trip(self):
        cases = (
            (continuous_callbacks, continuous_train, "continuous-v2"),
            (cnn_callbacks, cnn_train, "board-v1"),
            (hybrid_callbacks, hybrid_train, "hybrid-v1"),
        )
        for callbacks, training, feature_id in cases:
            with self.subTest(feature_id=feature_id), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "checkpoint.pt"
                env = {
                    "BOMBERMAN_CHECKPOINT": str(checkpoint),
                    "BOMBERMAN_FEATURE_ID": feature_id,
                    "BOMBERMAN_REWARD_ID": "r1",
                    "BOMBERMAN_TRAINING_TASK": "coin_navigation",
                }
                with patch.dict(os.environ, env, clear=False):
                    learner = self._self(True)
                    callbacks.setup(learner)
                    training.setup_training(learner)
                    training.end_of_round(learner, game_state(), "WAIT", [])
                    frozen = self._self(False)
                    callbacks.setup(frozen)
                    first = callbacks.act(frozen, game_state())
                    second = callbacks.act(frozen, game_state())
                    self.assertEqual(first, second)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                for field in (
                    "policy", "target", "optimizer", "replay", "torch_rng_state",
                    "agent_rng_state", "feature_schema", "network_spec", "hyperparameters",
                ):
                    self.assertIn(field, payload)

    def test_neural_warm_start_loads_only_policy_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            destination = root / "destination.pt"
            source_env = {
                "BOMBERMAN_CHECKPOINT": str(source),
                "BOMBERMAN_FEATURE_ID": "continuous-v2",
                "BOMBERMAN_REWARD_ID": "r1",
                "BOMBERMAN_TRAINING_TASK": "coin_navigation",
            }
            with patch.dict(os.environ, source_env, clear=True):
                original = self._self(True)
                continuous_callbacks.setup(original)
                continuous_train.setup_training(original)
                continuous_train.end_of_round(original, game_state(), "WAIT", [])

            warm_env = {
                "BOMBERMAN_CHECKPOINT": str(destination),
                "BOMBERMAN_INIT_CHECKPOINT": str(source),
                "BOMBERMAN_FEATURE_ID": "continuous-v2",
                "BOMBERMAN_REWARD_ID": "r5_conditional_loop",
                "BOMBERMAN_TRAINING_TASK": "crate_navigation",
            }
            with patch.dict(os.environ, warm_env, clear=True):
                warmed = self._self(True)
                continuous_callbacks.setup(warmed)

            source_payload = torch.load(source, map_location="cpu", weights_only=True)
            for name, value in warmed.model.policy.state_dict().items():
                torch.testing.assert_close(value.cpu(), source_payload["policy"][name])
                torch.testing.assert_close(
                    warmed.model.target.state_dict()[name].cpu(),
                    source_payload["policy"][name],
                )
            self.assertEqual(warmed.reward_id, "r5_conditional_loop")
            self.assertEqual(warmed.action_steps, 0)
            self.assertEqual(warmed.model.updates, 0)
            self.assertEqual(len(warmed.model.replay), 0)
            self.assertFalse(warmed.model.optimizer.state)

    def test_terminal_transition_replaces_pending_transition(self):
        owner = self._self(True)
        owner.reward_id = "r1"
        owner.curriculum_allows_bomb = True
        owner._feature_cache_key = None
        owner._feature_cache_value = None
        owner.model = SimpleNamespace(observe=lambda transition: observed.append(transition))
        owner.model_file = Path("unused.pt")
        observed = []
        continuous_train.setup_training(owner)
        old = game_state()
        new = game_state()
        new["step"] = 2
        continuous_train.game_events_occurred(owner, old, "WAIT", new, [])
        with patch("agent_code.learning_common.neural_agent.save_checkpoint_atomic"), \
             patch("agent_code.learning_common.neural_agent._append_metrics"):
            owner.model.checkpoint = lambda: {}
            owner.model.updates = 0
            owner.model.replay = []
            owner.action_steps = 1
            owner.rng = __import__("random").Random(0)
            owner.reward_spec = {}
            owner.training_task = "coin_navigation"
            continuous_train.end_of_round(owner, old, "WAIT", [])
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0].done)
        self.assertIsNone(observed[0].next_state)

    def test_double_q_resume_snapshot_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pkl"
            payload = {
                "checkpoint_schema": "training-resume-v7",
                "algorithm": "double_q_learning", "actions": list(dq_callbacks.ACTIONS),
                "feature_id": dq_callbacks.FEATURE_ID,
                "feature_schema": dq_callbacks.FEATURE_SCHEMA,
                "reward_id": "r1", "reward_version": "r1",
                "reward_spec": resolve_reward_spec("r1"),
                "training_task": "coin_navigation", "network_spec": None,
                "hyperparameters": dq_callbacks.HYPERPARAMETERS,
                "agent_seed": 11, "agent_rng_state": __import__("random").Random(11).getstate(),
                "exploration_spec": {},
                "safe_exploration": True,
                "safe_exploration_decisions": 1,
                "safe_exploration_fallbacks": 0,
                "safety_spec": {"version": "survival-mask-v1", "mode": "exploration", "horizon": 7, "fallback": "physical_q"},
                "safety_decisions": 1, "safety_interventions": 0,
                "safety_fallbacks": 0,
                "action_history_state": {
                    "previous_action": None, "wait_streak": 0, "round": None,
                },
                "n_step": 1,
                "n_step_state": {"n_step": 1, "gamma": 0.95, "pending": []},
                "retention_spec": {},
                "training_budget": {
                    "target_stage_action_steps": None, "min_rounds": 1,
                },
                "total_action_steps": 1,
                "stage_action_steps": 1,
                "training_device_name": None, "training_device_type": "cpu",
                "q_table_a": {(1,) * 12: np.ones(6, dtype=np.float32)},
                "q_table_b": {(2,) * 12: np.full(6, 2, dtype=np.float32)},
            }
            with checkpoint.open("wb") as file:
                pickle.dump(payload, file)
            commit_training_snapshot(
                root, checkpoint, algorithm="double_q_learning",
                task="coin_navigation", seed=11, round_index=1,
                world_rng_state={}, python_rng_state=(), numpy_rng_state=(),
                early_stopping_rewards=[], source_commit=None,
            )
            loaded = load_training_snapshot(root)
            materialized = root / "restored.pkl"
            materialize_learner_checkpoint(loaded, materialized)
            with materialized.open("rb") as file:
                restored = pickle.load(file)
            np.testing.assert_array_equal(
                restored["q_table_a"][(1,) * 12], payload["q_table_a"][(1,) * 12])
            np.testing.assert_array_equal(
                restored["q_table_b"][(2,) * 12], payload["q_table_b"][(2,) * 12])


if __name__ == "__main__":
    unittest.main()

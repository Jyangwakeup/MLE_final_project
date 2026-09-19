"""Tests for the isolated Q-learning demonstration experiment."""

from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import events as e
from agent_code.learning_common.linear_agent import LinearTransition
from agent_code.optimized_double_q_lambda_demo_agent import callbacks
from experiments.agent_contracts import resolve_agent_contract
from experiments.pretrain_q_learning_demonstrations import pretrain
from experiments.q_learning_demo_data import append_raw_transition, encode_raw_dataset
from tests.test_danger import make_game_state


def state(step, position):
    value = make_game_state(position=position)
    value.update(round=1, step=step, coins=[(8, 3)])
    return value


class QLearningDemonstrationTests(unittest.TestCase):
    def test_contract_matches_r20_agent(self):
        old = resolve_agent_contract("optimized_double_q_lambda_agent")
        new = resolve_agent_contract("optimized_double_q_lambda_demo_agent")
        self.assertEqual(old.algorithm, new.algorithm)
        self.assertEqual(old.feature_schema, new.feature_schema)
        self.assertEqual(old.hyperparameters, new.hyperparameters)

    def test_demonstration_update_does_not_touch_online_state(self):
        model = callbacks.make_model(11)
        model.traces[0, 0, 0] = 0.5
        traces = model.traces.copy()
        updates = model.updates
        rng_state = pickle.dumps(model.rng.bit_generator.state)
        transition = LinearTransition(
            np.zeros(84, dtype=np.float32), 0, 1.0,
            np.ones(84, dtype=np.float32), False,
            np.ones(6, dtype=bool),
        )
        loss = model.observe_demonstration(
            transition, rng=np.random.default_rng(3), learning_rate=0.02)
        self.assertGreater(loss, 0)
        np.testing.assert_array_equal(model.traces, traces)
        self.assertEqual(model.updates, updates)
        self.assertEqual(pickle.dumps(model.rng.bit_generator.state), rng_state)

    def test_raw_capture_encodes_history_and_terminal_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, dataset = root / "raw.pkl", root / "dataset.npz"
            first = state(1, (3, 3))
            second = state(2, (4, 3))
            append_raw_transition(
                raw, environment_seed=6000, round_index=1,
                old_state=first, action="RIGHT", new_state=second,
                events=[e.MOVED_RIGHT], terminal=False)
            append_raw_transition(
                raw, environment_seed=6000, round_index=1,
                old_state=second, action="WAIT", new_state=second,
                events=[e.WAITED], terminal=False)
            append_raw_transition(
                raw, environment_seed=6000, round_index=1,
                old_state=second, action="WAIT", new_state=None,
                events=[e.WAITED, e.SURVIVED_ROUND], terminal=True)
            result = encode_raw_dataset(raw, dataset)
            encoded = np.load(dataset, allow_pickle=False)
            self.assertEqual(result["transitions"], 3)
            self.assertEqual(encoded["states"].shape, (3, 84))
            self.assertEqual(encoded["legal_masks"].shape, (3, 6))
            self.assertEqual(int(encoded["actions"][0]), callbacks.ACTIONS.index("RIGHT"))
            self.assertTrue(bool(encoded["dones"][2]))
            self.assertFalse(encoded["next_legal"][2].any())
            # The unchanged legacy adapter records the previous action.
            self.assertGreater(float(np.abs(encoded["states"][1] - encoded["states"][0]).sum()), 0)
            # end_of_round repeats the last callback without recording the
            # selected action a second time.
            np.testing.assert_array_equal(encoded["states"][1], encoded["states"][2])

    def test_pretraining_preserves_online_counter_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.npz"
            np.savez_compressed(
                dataset,
                states=np.zeros((2, 84), dtype=np.float32),
                actions=np.asarray([0, 1], dtype=np.int8),
                rewards=np.asarray([1.0, 0.5], dtype=np.float32),
                next_states=np.ones((2, 84), dtype=np.float32),
                next_legal=np.ones((2, 6), dtype=bool),
                dones=np.asarray([False, True]),
            )
            model = callbacks.make_model(11)
            payload = model.checkpoint()
            payload.update({
                "algorithm": callbacks.ALGORITHM,
                "feature_id": callbacks.FEATURE_ID,
                "feature_schema": callbacks.FEATURE_SCHEMA,
                "hyperparameters": callbacks.HYPERPARAMETERS,
                "reward_id": "r20_safe_credit_targeted_wait",
                "agent_seed": 11,
            })
            payload["updates"] = 123
            source, output = root / "source.pkl", root / "pretrained.pkl"
            with source.open("wb") as file:
                pickle.dump(payload, file)
            pretrain(dataset, source, output, passes=2, seed=7)
            with output.open("rb") as file:
                trained = pickle.load(file)
            self.assertEqual(trained["updates"], 123)
            self.assertEqual(trained["learner_rng_state"], payload["learner_rng_state"])
            self.assertTrue(np.all(trained["traces"] == 0))
            self.assertEqual(
                trained["demonstration"]["total_demonstration_updates"], 4)


if __name__ == "__main__":
    unittest.main()

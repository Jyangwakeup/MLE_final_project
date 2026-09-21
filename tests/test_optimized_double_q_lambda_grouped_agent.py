"""Model and contract tests for grouped tile-coded Double Q(lambda)."""

from types import SimpleNamespace
import unittest

import numpy as np

from experiments.agent_variants.optimized_double_q_lambda_grouped_agent.callbacks import (
    BASE_GROUPS, GROUP_ACTION_FEATURE_INDICES, HISTORY_GROUPS, HYPERPARAMETERS,
    make_model,
)
from experiments.agent_contracts import resolve_agent_contract


class GroupedDoubleQLambdaTests(unittest.TestCase):
    def test_contract_has_two_four_tiling_groups(self):
        contract = resolve_agent_contract("optimized_double_q_lambda_grouped_agent")
        self.assertEqual(contract.feature_schema["vector_shape"], [84])
        self.assertEqual(HYPERPARAMETERS["group_tilings"], [4, 4])
        self.assertEqual(HYPERPARAMETERS["group_memory_sizes"], [16384, 16384])
        self.assertEqual(HYPERPARAMETERS["grouped_tile_encoding_version"], 1)
        self.assertEqual(len(BASE_GROUPS[0]), 27)
        self.assertEqual(HISTORY_GROUPS[0], [60, 61, 63, 77, 78])
        self.assertEqual(len(GROUP_ACTION_FEATURE_INDICES), 6)

    def test_history_change_does_not_change_base_tiles(self):
        model = make_model(11)
        before = np.zeros(84, dtype=np.float32)
        after = before.copy()
        after[77] = 1.0
        after[78] = 1.0
        base_before, history_before = model.active_tiles(before, 0)
        base_after, history_after = model.active_tiles(after, 0)
        np.testing.assert_array_equal(base_before, base_after)
        self.assertFalse(np.array_equal(history_before, history_after))

    def test_q_value_is_sum_of_both_groups_and_estimators(self):
        model = make_model(3)
        state = np.zeros(84, dtype=np.float32)
        active = model.active_tiles(state, 2)
        for estimator in range(2):
            for group in range(2):
                model.weights[group][estimator, 2, active[group]] = estimator + group + 1
        expected = sum(float(model.weights[group][estimator, 2, active[group]].sum())
                       for estimator in range(2) for group in range(2)) / 2
        self.assertAlmostEqual(float(model.q_values(state)[2]), expected)

    def test_watkins_cut_updates_current_tiles_and_terminal_clears(self):
        model = make_model(7)
        model.traces[0][0, 1, 3] = 1.0
        model.traces[1][0, 1, 3] = 1.0
        model.set_behavior_greedy(False)
        transition = SimpleNamespace(
            state=np.zeros(84, dtype=np.float32), action=0, reward=1.0,
            next_state=np.zeros(84, dtype=np.float32), done=False,
            next_legal=np.ones(6, dtype=bool), steps=1)
        model.observe(transition)
        self.assertEqual(float(model.traces[0][:, 1, 3].sum()), 0.0)
        self.assertEqual(float(model.traces[1][:, 1, 3].sum()), 0.0)
        self.assertGreater(sum(float(trace.sum()) for trace in model.traces), 0.0)
        model.observe(SimpleNamespace(
            state=transition.state, action=0, reward=0.0,
            next_state=None, done=True, next_legal=None, steps=1))
        self.assertEqual(sum(float(trace.sum()) for trace in model.traces), 0.0)

    def test_checkpoint_round_trip_preserves_q_and_training_state(self):
        model = make_model(13)
        transition = SimpleNamespace(
            state=np.zeros(84, dtype=np.float32), action=3, reward=2.0,
            next_state=np.ones(84, dtype=np.float32), done=False,
            next_legal=np.ones(6, dtype=bool), steps=1)
        model.observe(transition)
        payload = model.checkpoint()
        restored = make_model(13)
        restored.load_checkpoint(payload, training=True)
        np.testing.assert_allclose(
            restored.q_values(transition.state), model.q_values(transition.state))
        self.assertEqual(restored.updates, model.updates)
        for left, right in zip(restored.traces, model.traces):
            np.testing.assert_array_equal(left, right)

    def test_double_q_target_uses_other_estimator(self):
        model = make_model(19)
        state = np.zeros(84, dtype=np.float32)
        next_state = np.ones(84, dtype=np.float32)
        estimator = int(np.random.default_rng(19).integers(2))
        other = 1 - estimator
        selected = 4
        for group, tiles in enumerate(model.active_tiles(next_state, selected)):
            model.weights[group][estimator, selected, tiles] = 2.0
            model.weights[group][other, selected, tiles] = 0.5
        before = model.q_values(state, estimator=estimator)[0]
        expected_target = model.gamma * model.q_values(next_state, estimator=other)[selected]
        delta = model.observe(SimpleNamespace(
            state=state, action=0, reward=0.0, next_state=next_state,
            done=False, next_legal=np.asarray([False, False, False, False, True, False]),
            steps=1))
        self.assertAlmostEqual(delta, abs(float(expected_target - before)), places=5)


if __name__ == "__main__":
    unittest.main()

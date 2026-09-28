"""Regression tests for the continuous-v2 history-input repair."""

import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from agent_code.learning_common import linear_agent
from agent_code.learning_common.action_history import (
    advance_observation,
    init_action_history,
    record_selected_action,
)
from all_other_agent_code.optimized_double_q_lambda_agent.callbacks import (
    HYPERPARAMETERS as LEGACY_HYPERPARAMETERS,
)
from experiments.agent_variants.optimized_double_q_lambda_history_agent import callbacks
from experiments.agent_variants.optimized_double_q_lambda_history_agent.features import features_for_state
from experiments.agent_contracts import resolve_agent_contract
from experiments.package_agent import build_submission
from tests.test_danger import make_game_state


def state(*, round_index=1, step=1, position=(3, 3), coins=((8, 3),)):
    value = make_game_state(position=position)
    value.update(round=round_index, step=step, coins=list(coins))
    return value


class HistoryInputAgentTests(unittest.TestCase):
    def test_contract_marks_repaired_history_without_changing_dimension(self):
        contract = resolve_agent_contract("optimized_double_q_lambda_history_agent")
        self.assertEqual(contract.feature_id, "continuous-v2")
        self.assertEqual(contract.feature_schema["vector_shape"], [84])
        self.assertEqual(contract.hyperparameters["history_input_version"], 1)
        self.assertNotIn("history_input_version", LEGACY_HYPERPARAMETERS)

    def test_move_populates_target_continuity_and_reverse_action_only(self):
        owner = SimpleNamespace()
        init_action_history(owner)
        old = state(position=(3, 3), step=1)
        advance_observation(owner, old)
        before = features_for_state(owner, old)
        record_selected_action(owner, old, "RIGHT")
        self.assertIs(features_for_state(owner, old), before)

        new = state(position=(4, 3), step=2)
        after = features_for_state(owner, new)
        self.assertEqual(float(after.vector[77]), 1.0)
        np.testing.assert_array_equal(
            after.vector[78:84], np.asarray([0, 0, 0, 1, 0, 0], dtype=np.float32))

    def test_explicit_empty_history_and_round_reset_are_zero(self):
        empty = features_for_state(SimpleNamespace(), state())
        self.assertEqual(float(empty.vector[77]), 0.0)
        np.testing.assert_array_equal(empty.vector[78:84], np.zeros(6))

        owner = SimpleNamespace()
        init_action_history(owner)
        old = state(round_index=1, step=1)
        advance_observation(owner, old)
        record_selected_action(owner, old, "RIGHT")
        next_round = state(round_index=2, step=1, position=(4, 3))
        advance_observation(owner, next_round)
        value = features_for_state(owner, next_round)
        self.assertEqual(float(value.vector[77]), 0.0)
        np.testing.assert_array_equal(value.vector[78:84], np.zeros(6))
        self.assertEqual(list(owner.history_input_cache), [(2, 1)])

    def test_two_state_cache_preserves_decision_and_transition_views(self):
        owner = SimpleNamespace(
            linear_config=callbacks, _feature_cache_key=None,
            _feature_cache_value=None)
        init_action_history(owner)
        old = state(step=1, position=(3, 3))
        advance_observation(owner, old)
        decision_old = linear_agent.features(owner, old)
        record_selected_action(owner, old, "RIGHT")
        transition_old = linear_agent.features(owner, old)
        self.assertIs(decision_old, transition_old)

        new = state(step=2, position=(4, 3))
        transition_new = linear_agent.features(owner, new)
        advance_observation(owner, new)
        next_decision = linear_agent.features(owner, new)
        self.assertIs(transition_new, next_decision)
        self.assertEqual(float(next_decision.vector[77]), 1.0)
        self.assertEqual(list(owner.history_input_cache), [(1, 1), (1, 2)])

        third = state(step=3, position=(5, 3))
        linear_agent.features(owner, third)
        self.assertEqual(list(owner.history_input_cache), [(1, 2), (1, 3)])

    def test_legacy_checkpoint_is_rejected_by_package_contract(self):
        contract = resolve_agent_contract("optimized_double_q_lambda_history_agent")
        legacy = {
            "algorithm": contract.algorithm,
            "feature_id": contract.feature_id,
            "feature_schema": contract.feature_schema,
            "actions": contract.feature_schema["action_order"],
            "network_spec": contract.network_spec,
            "hyperparameters": LEGACY_HYPERPARAMETERS,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "legacy.pkl"
            with checkpoint.open("wb") as file:
                pickle.dump(legacy, file)
            with self.assertRaisesRegex(ValueError, "restore"):
                build_submission(
                    "optimized_double_q_lambda_history_agent", checkpoint,
                    root / "submission.zip")


if __name__ == "__main__":
    unittest.main()

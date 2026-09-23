"""Single-factor contract and tile separation regression tests."""
import unittest
import numpy as np
from agent_code.optimized_double_q_lambda_agent import callbacks as old
from experiments.agent_variants.optimized_double_q_lambda_crate_agent import callbacks as new
from agent_code.team_agent.feature_system.continuous_v2 import VECTOR_FIELDS
from experiments.agent_contracts import resolve_agent_contract


class CrateQuantizationTests(unittest.TestCase):
    def test_only_ternary_contract_changes(self):
        before = dict(old.HYPERPARAMETERS)
        after = dict(new.HYPERPARAMETERS)
        before.pop("ternary_indices")
        after.pop("ternary_indices")
        self.assertEqual(before, after)
        added = set(new.HYPERPARAMETERS["ternary_indices"]) - set(
            old.HYPERPARAMETERS["ternary_indices"])
        self.assertEqual(len(added), 6)
        self.assertTrue(all(VECTOR_FIELDS[i].endswith(
            "crate_frontier_distance_delta") for i in added))
        self.assertEqual(resolve_agent_contract(
            "optimized_double_q_lambda_crate_agent").feature_id, "continuous-v2")

    def test_all_actions_separate_three_crate_directions(self):
        for seed in (11, 22, 33):
            model = new.make_model(seed)
            for projection in model.action_feature_indices:
                index = next(i for i in projection if VECTOR_FIELDS[i].endswith(
                    "crate_frontier_distance_delta"))
                encodings = []
                for value in (-1 / 288, 0, 1 / 288):
                    state = np.zeros(84, dtype=np.float32)
                    state[index] = value
                    encodings.append(model.coder.encode(state[projection]))
                for first, second in ((0, 1), (1, 2), (0, 2)):
                    self.assertTrue(np.all(encodings[first] != encodings[second]))

    def test_determinism_and_no_input_mutation(self):
        first, second = new.make_model(11), new.make_model(11)
        state = np.linspace(-0.1, 0.5, 84, dtype=np.float32)
        saved = state.copy()
        np.testing.assert_array_equal(first.q_values(state), second.q_values(state))
        np.testing.assert_array_equal(state, saved)


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path

from agent_code.rainbow_lite_agent import callbacks as rainbow_callbacks
from experiments.agent_contracts import resolve_agent_contract
from experiments.performance_stopping import resolve_performance_stopping


CONFIG_DIR = Path(__file__).parents[1] / "experiments" / "configs"


class RecommendedTrainingConfigTests(unittest.TestCase):
    def _config(self, stem, task):
        return json.loads(
            (CONFIG_DIR / f"{stem}_task{task}.json").read_text(encoding="utf-8")
        )

    def test_rainbow_r13_no_safety_contract(self):
        config = json.loads((
            CONFIG_DIR / "rainbow_lite_r13_no_safety_task2.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(config["reward_id"], "r13_no_safety_survival_credit")
        self.assertEqual(config["safety"]["mode"], "off")
        self.assertEqual(config["training"]["target_stage_action_steps"], 150_000)
        self.assertGreaterEqual(config["training"]["n_rounds"], 4_000)

    def test_rainbow_r14_quick_evaluation_and_no_safety_contract(self):
        config = json.loads((
            CONFIG_DIR / "rainbow_lite_r14_no_safety_task2.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(config["reward_id"], "r14_no_safety_useful_bomb_credit")
        self.assertEqual(config["safety"]["mode"], "off")
        self.assertEqual(len(config["evaluation"]["seeds"]), 5)

    def test_rainbow_r15_quick_evaluation_and_no_safety_contract(self):
        config = json.loads((
            CONFIG_DIR / "rainbow_lite_r15_no_safety_task2.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(config["reward_id"], "r15_no_safety_objective_credit")
        self.assertEqual(config["safety"]["mode"], "off")
        self.assertEqual(len(config["evaluation"]["seeds"]), 5)

    def test_recommended_feature_and_reward_pairs(self):
        expected = {
            "double_q_lambda": ("continuous-v2", "r7_safe_credit_potential"),
            "expected_sarsa_lambda": (
                "continuous-v4", "r10_bounded_history_anti_loop"
            ),
            "rainbow_lite": (
                "continuous-v4", "r10_bounded_history_anti_loop"
            ),
            "rainbow_lite_fallback": (
                "continuous-v4", "r7_safe_credit_sparse"
            ),
        }
        for stem, (feature_id, reward_id) in expected.items():
            with self.subTest(stem=stem):
                for task in (1, 2):
                    config = self._config(stem, task)
                    self.assertEqual(config["feature_id"], feature_id)
                    self.assertEqual(config["reward_id"], reward_id)

    def test_task1_uses_frozen_score_stopping(self):
        for stem in (
            "double_q_lambda", "expected_sarsa_lambda", "rainbow_lite",
            "rainbow_lite_fallback",
        ):
            with self.subTest(stem=stem):
                training = self._config(stem, 1)["training"]
                self.assertFalse(training["early_stopping"]["enabled"])
                spec = resolve_performance_stopping(
                    training["performance_stopping"], task="coin_navigation"
                )
                self.assertEqual(spec["threshold"], 48.0)
                self.assertEqual(spec["consecutive_passes"], 3)

    def test_task2_preserves_learning_budget(self):
        for stem in (
            "double_q_lambda", "expected_sarsa_lambda", "rainbow_lite",
            "rainbow_lite_fallback",
        ):
            with self.subTest(stem=stem):
                training = self._config(stem, 2)["training"]
                self.assertEqual(training["target_stage_action_steps"], 150_000)
                self.assertEqual(training["min_rounds"], 500)
                self.assertFalse(training["early_stopping"]["enabled"])

    def test_r9_task2_configs_are_paired_with_existing_candidates(self):
        for stem, feature_id, n_step in (
            ("expected_sarsa_lambda", "continuous-v4", 1),
            ("rainbow_lite", "continuous-v4", 4),
        ):
            baseline = self._config(stem, 2)
            candidate = json.loads((
                CONFIG_DIR / f"{stem}_r9_task2.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(candidate["reward_id"], "r9_safe_credit_anti_loop")
            self.assertEqual(candidate["feature_id"], feature_id)
            self.assertEqual(candidate["training"]["n_step"], n_step)
            baseline["reward_id"] = candidate["reward_id"]
            self.assertEqual(candidate, baseline)

    def test_rainbow_runtime_matches_continuous_v4_contract(self):
        contract = resolve_agent_contract("rainbow_lite_agent", "continuous-v4")
        self.assertEqual(contract.feature_schema["vector_shape"], [126])
        self.assertEqual(rainbow_callbacks.NETWORK_SPEC["input_shape"], [126])

    def test_archived_rainbow_contracts_resolve_from_historical_tree(self):
        code = """
from experiments.agent_contracts import resolve_agent_contract
names = (
    'rainbow_lite_v6_agent', 'rainbow_lite_v6_stable_agent',
    'rainbow_lite_v7_agent', 'rainbow_lite_v8_agent',
    'rainbow_lite_v9_agent', 'rainbow_lite_v10_agent',
    'rainbow_lite_v11_agent',
)
for name in names:
    assert resolve_agent_contract(name).agent == name
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()

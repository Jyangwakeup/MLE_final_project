"""Regression coverage for the isolated Watkins Double Q(lambda) candidate."""

import json
from pathlib import Path
import unittest

import numpy as np


class OptimizedDoubleQLambdaTests(unittest.TestCase):
    def test_contract_uses_the_isolated_agent_and_watkins_cut(self):
        from experiments.agent_contracts import resolve_agent_contract
        from agent_code.optimized_double_q_lambda_agent.callbacks import (
            HYPERPARAMETERS,
        )

        contract = resolve_agent_contract("optimized_double_q_lambda_agent")
        self.assertEqual(contract.algorithm, "double_q_lambda")
        self.assertEqual(contract.feature_id, "continuous-v2")
        self.assertTrue(HYPERPARAMETERS["watkins_trace_cut"])

    def test_non_greedy_behavior_cuts_old_traces_before_current_update(self):
        from agent_code.learning_common.tile_coding import TraceControl

        hyperparameters = {
            "gamma": 0.95, "learning_rate": 0.1, "lambda": 0.8,
            "tilings": 2, "bins": 4, "memory_size": 64,
            "watkins_trace_cut": True,
        }
        learner = TraceControl(
            2, 2, seed=7, hyperparameters=hyperparameters,
            algorithm="double_q_lambda")
        learner.traces[0, 1, 3] = 1.0
        learner.set_behavior_greedy(False)
        transition = type("Transition", (), {
            "state": np.zeros(2, dtype=np.float32), "action": 0,
            "reward": 1.0, "next_state": np.zeros(2, dtype=np.float32),
            "done": False, "next_legal": np.array([True, True]), "steps": 1,
        })()

        learner.observe(transition)

        self.assertEqual(float(learner.traces[0, 1, 3]), 0.0)
        self.assertGreater(float(learner.traces.sum()), 0.0)

    def test_task1_config_has_action_budget_and_snapshot_interval(self):
        root = Path(__file__).resolve().parents[1] / "experiments" / "configs"
        config = json.loads((root / "optimized_double_q_lambda_task1.json").read_text())

        self.assertEqual(config["training"]["target_stage_action_steps"], 100000)
        self.assertEqual(
            config["training"]["checkpoint_snapshot_interval_action_steps"],
            25000)
        self.assertEqual(config["safety"]["mode"], "off")


if __name__ == "__main__":
    unittest.main()

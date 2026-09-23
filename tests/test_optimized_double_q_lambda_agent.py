"""Regression coverage for the isolated Watkins Double Q(lambda) candidate."""

import json
from pathlib import Path
import unittest

import numpy as np

from experiments.performance_stopping import resolve_performance_stopping
from experiments.resume import validate_resume_transition


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

    def test_formal_task1_and_task2_are_an_exact_model_contract(self):
        root = Path(__file__).resolve().parents[1] / "experiments" / "configs"
        task1 = json.loads((
            root / "optimized_double_q_lambda_formal_task1.json").read_text())
        task2 = json.loads((
            root / "optimized_double_q_lambda_task2.json").read_text())

        for field in ("algorithm", "feature_id", "reward_id", "safety"):
            self.assertEqual(task1[field], task2[field])
        self.assertEqual(task1["safety"]["mode"], "all")
        stopping = resolve_performance_stopping(
            task1["training"]["performance_stopping"], task="coin_navigation")
        self.assertEqual(stopping["consecutive_passes"], 3)
        self.assertEqual(stopping["evaluation_seeds"], list(range(9000, 9020)))
        self.assertEqual(task2["training"]["target_stage_action_steps"], 150000)
        self.assertEqual(task2["training"]["min_rounds"], 500)
        self.assertNotIn("performance_stopping", task2["training"])

    def test_formal_contract_can_promote_directly_to_task2(self):
        from experiments.agent_contracts import resolve_agent_contract
        from agent_code.team_agent.feature_system import ACTIONS
        from agent_code.team_agent.rewards import resolve_reward_spec

        contract = resolve_agent_contract("optimized_double_q_lambda_agent")
        safety = {
            "version": "survival-mask-v1", "mode": "all",
            "horizon": 7, "fallback": "physical_q",
        }
        base = {
            "algorithm": contract.algorithm, "seed": 11,
            "task": "coin_navigation", "checkpoint_schema": "training-resume-v8",
            "feature_id": contract.feature_id,
            "feature_schema": contract.feature_schema,
            "reward_id": "r7_safe_credit_potential",
            "reward_spec": resolve_reward_spec("r7_safe_credit_potential"),
            "training_device_type": "cpu", "training_device_name": None,
            "agent_seed": 11, "source_commit": "same", "source_hash": "same",
            "safe_exploration": True, "safety_spec": safety,
            "actions": list(ACTIONS), "network_spec": None,
            "hyperparameters": contract.hyperparameters,
        }
        child = {**base, "task": "crate_navigation"}
        self.assertEqual(
            validate_resume_transition(base, child, parent_status="early_stopped"),
            "next_task",
        )


if __name__ == "__main__":
    unittest.main()

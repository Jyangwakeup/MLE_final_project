"""Contract tests for audited runner-only curriculum migration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.exploration import resolve_exploration_spec
from agent_code.team_agent.rewards import resolve_reward_spec
from agent_code.team_agent.safety import resolve_safety_spec
from agent_code.learning_common.training_spec import (
    resolve_retention_spec, resolve_safety_replay_spec,
)
from experiments.resume import CHECKPOINT_SCHEMA_VERSION
from experiments.runtime_migration import (
    build_runtime_migration_audit, validate_changed_paths,
)


def _contract(task: str, source_hash: str) -> dict:
    reward = "r7_safe_credit_potential"
    return {
        "algorithm": "double_q_lambda", "seed": 11, "task": task,
        "feature_id": "continuous-v4",
        "feature_schema": feature_schema_contract("continuous-v4"),
        "feature_version": None, "reward_id": reward,
        "reward_version": reward, "reward_spec": resolve_reward_spec(reward),
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
        "training_device_name": None, "training_device_type": "cpu",
        "agent_seed": 11, "exploration_spec": resolve_exploration_spec(),
        "safe_exploration": True, "safety_spec": resolve_safety_spec({
            "version": "survival-mask-v1", "mode": "all",
            "horizon": 7, "fallback": "physical_q"}),
        "n_step": 1, "retention_spec": resolve_retention_spec(),
        "safety_replay_spec": resolve_safety_replay_spec(),
        "training_budget": {"target_stage_action_steps": None, "min_rounds": 1},
        "performance_stopping": None, "network_spec": None,
        "hyperparameters": {}, "transfer_contract": None,
        "source_commit": "parent-commit", "source_hash": source_hash,
        "source_hash_scope": "agent-runtime-v2",
    }


class RuntimeMigrationTests(unittest.TestCase):
    def test_strict_path_allowlist(self):
        self.assertEqual(validate_changed_paths([
            "experiments/run.py", "scripts/x.sh", "tests/test_x.py",
        ]), ["experiments/run.py", "scripts/x.sh", "tests/test_x.py"])
        for path in (
            "agent_code/optimized_double_q_lambda_v4_agent/train.py",
            "agent_code/team_agent/rewards.py", "settings.py",
        ):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, path):
                validate_changed_paths([path])

    def test_audit_accepts_runner_only_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            parent_run = root / "parent"
            parent_run.mkdir()
            (parent_run / "promotion_audit.json").write_text(
                json.dumps({"passed": True, "checkpoint_sha256": "abc"}),
                encoding="utf-8")
            calls = iter(("current-commit", "experiments/run.py\nexperiments/training.py", ""))
            with patch("experiments.runtime_migration._git", side_effect=lambda *_: next(calls)):
                audit = build_runtime_migration_audit(
                    project_root=root, parent_run=parent_run,
                    parent_contract=_contract("coin_navigation", "old"),
                    child_contract=_contract("crate_navigation", "new"),
                    parent_status="completed")
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["current_source_hash"], "new")

    def test_audit_rejects_unqualified_parent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            parent_run = root / "parent"
            parent_run.mkdir()
            (parent_run / "promotion_audit.json").write_text(
                json.dumps({"passed": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "passed Task 1"):
                build_runtime_migration_audit(
                    project_root=root, parent_run=parent_run,
                    parent_contract=_contract("coin_navigation", "old"),
                    child_contract=_contract("crate_navigation", "new"),
                    parent_status="completed")


if __name__ == "__main__":
    unittest.main()

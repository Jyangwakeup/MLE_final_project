import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from experiments.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    LoadedSnapshot,
    commit_training_snapshot,
    load_training_snapshot,
    materialize_migrated_checkpoint,
    materialize_learner_checkpoint,
    validate_v6_migration,
    validate_resume_transition,
)
from agent_code.team_agent.feature_system import feature_schema_contract


EXPLORATION_SPEC = {
    "version": "linear-v1",
    "start": 1.0,
    "end": 0.05,
    "decay_action_steps": 80_000,
}
REWARD_SPEC = {
    "step": -0.01,
    "coin_collected": 1.0,
    "killed_opponent": 5.0,
    "crate_destroyed": 0.2,
    "death": -10.0,
    "invalid_action": -0.1,
}
RETENTION_SPEC = {
    "parent_fraction": 0.5,
    "distillation_weight": 1.0,
    "temperature": 1.0,
    "per_task_capacity": 20_000,
    "current_warmup": 2_000,
}
TRAINING_BUDGET = {"target_stage_action_steps": 100_000, "min_rounds": 1}
SAFETY_SPEC = {
    "version": "survival-mask-v1", "mode": "exploration", "horizon": 7,
    "fallback": "physical_q",
}


class ResumeProtocolTestCase(unittest.TestCase):
    def _q_checkpoint(self, path: Path, value: float, round_index: int) -> None:
        with path.open("wb") as file:
            pickle.dump({
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                "algorithm": "q_learning",
                "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
                "feature_version": "v1",
                "feature_id": "discrete-v1",
                "feature_schema": feature_schema_contract("discrete-v1"),
                "network_spec": None,
                "hyperparameters": {},
                "reward_id": "r1",
                "reward_version": "r1",
                "reward_spec": REWARD_SPEC,
                "training_device_type": "cpu",
                "training_device_name": None,
                "agent_seed": 11,
                "exploration_spec": EXPLORATION_SPEC,
                "safe_exploration": True,
                "safe_exploration_decisions": round_index,
                "safe_exploration_fallbacks": 0,
                "safety_spec": SAFETY_SPEC,
                "safety_decisions": round_index,
                "safety_interventions": 0,
                "safety_fallbacks": 0,
                "action_history_state": {
                    "previous_action": None, "wait_streak": 0, "round": None,
                },
                "n_step": 1,
                "n_step_state": {"n_step": 1, "gamma": 0.95, "pending": []},
                "retention_spec": RETENTION_SPEC,
                "training_budget": TRAINING_BUDGET,
                "q_table": {(0,) * 14: np.full(6, value, dtype=np.float32)},
                "training_steps": round_index,
                "total_action_steps": round_index,
                "stage_action_steps": round_index,
                "training_task": "coin_navigation",
                "agent_rng_state": random.Random(7).getstate(),
            }, file)

    def test_q_snapshot_round_trip_uses_npz_and_falls_back_one_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "final.pkl"
            self._q_checkpoint(checkpoint, 1.0, 1)
            commit_training_snapshot(
                run, checkpoint, algorithm="q_learning", task="coin_navigation",
                seed=11, round_index=1, world_rng_state={"state": 1},
                python_rng_state=random.Random(1).getstate(),
                numpy_rng_state=np.random.RandomState(1).get_state(),
                early_stopping_rewards=[1.0], source_commit="abc",
            )
            self._q_checkpoint(checkpoint, 2.0, 2)
            commit_training_snapshot(
                run, checkpoint, algorithm="q_learning", task="coin_navigation",
                seed=11, round_index=2, world_rng_state={"state": 2},
                python_rng_state=random.Random(2).getstate(),
                numpy_rng_state=np.random.RandomState(2).get_state(),
                early_stopping_rewards=[1.0, 2.0], source_commit="abc",
            )

            latest = json.loads((run / "resume" / "latest.json").read_text())
            self.assertEqual(len(latest["generations"]), 2)
            newest = run / "resume" / latest["generations"][0]
            manifest = json.loads((newest / "manifest.json").read_text())
            self.assertEqual(manifest["agent_seed"], 11)
            self.assertEqual(manifest["exploration_spec"], EXPLORATION_SPEC)
            self.assertTrue((newest / "q_table.npz").is_file())
            (newest / "q_table.npz").write_bytes(b"corrupt")

            loaded = load_training_snapshot(run)
            self.assertEqual(loaded.round_index, 1)
            self.assertEqual(loaded.lost_rounds, 1)
            self.assertIn("integrity validation", loaded.fallback_reason)
            destination = run / "restored.pkl"
            materialize_learner_checkpoint(loaded, destination)
            with destination.open("rb") as file:
                restored = pickle.load(file)
            np.testing.assert_array_equal(restored["q_table"][(0,) * 14], np.ones(6))

    def test_resume_transition_allows_same_or_direct_next_task_only(self):
        parent = {
            "algorithm": "q_learning", "seed": 11, "task": "crate_navigation",
            "feature_version": "v1", "reward_version": "r1",
            "reward_spec": REWARD_SPEC,
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "training_device_type": "cpu", "training_device_name": None,
            "agent_seed": 11, "exploration_spec": EXPLORATION_SPEC,
            "safe_exploration": True, "n_step": 1,
            "safety_spec": SAFETY_SPEC,
            "retention_spec": RETENTION_SPEC,
            "training_budget": TRAINING_BUDGET,
            "source_commit": "abc123", "source_hash": "source-hash",
        }
        validate_resume_transition(parent, {
            **parent, "task": "weak_opponents",
        }, parent_status="completed")
        with self.assertRaisesRegex(ValueError, "direct next Task"):
            validate_resume_transition(parent, {
                **parent, "task": "full_match",
            }, parent_status="completed")
        with self.assertRaisesRegex(ValueError, "reward_version"):
            validate_resume_transition(parent, {
                **parent, "reward_version": "r1_no_crate",
            }, parent_status="completed")
        with self.assertRaisesRegex(ValueError, "reward_spec"):
            validate_resume_transition(parent, {
                **parent,
                "reward_spec": {**REWARD_SPEC, "coin_collected": 3.0},
            }, parent_status="completed")
        self.assertEqual(
            validate_resume_transition(parent, dict(parent), parent_status="running"),
            "same_task",
        )
        with self.assertRaisesRegex(ValueError, "completed parent"):
            validate_resume_transition(
                parent, {**parent, "task": "weak_opponents"}, parent_status="running"
            )
        for field, value in (
            ("algorithm", "dqn"), ("seed", 22),
            ("feature_version", "v2"), ("checkpoint_schema", "old"),
            ("training_device_type", "cuda"),
            ("agent_seed", 22),
            ("exploration_spec", {**EXPLORATION_SPEC, "end": 0.1}),
            ("source_commit", "different-commit"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_resume_transition(
                        parent, {**parent, field: value}, parent_status="completed"
                    )

        with self.assertRaisesRegex(ValueError, "source_hash"):
            validate_resume_transition(
                parent, {**parent, "source_hash": "different-source"},
                parent_status="completed",
            )

    def test_resume_rejects_new_feature_schema_mismatch(self):
        parent = {
            "algorithm": "q_learning", "seed": 11, "task": "coin_navigation",
            "feature_id": "discrete-v1", "feature_version": "v1",
            "feature_schema": feature_schema_contract("discrete-v1"),
            "reward_id": "r1", "reward_version": "r1",
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
        }
        child = dict(parent)
        child["feature_schema"] = {
            **parent["feature_schema"], "vector_shape": [41],
        }
        with self.assertRaisesRegex(ValueError, "schema/shape"):
            validate_resume_transition(parent, child, parent_status="running")

    def test_legacy_checkpoint_directory_is_not_a_resume_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self._q_checkpoint(run / "final.pkl", 1.0, 1)
            with self.assertRaisesRegex(ValueError, "no complete resume snapshot"):
                load_training_snapshot(run)

    def test_explicit_v6_migration_rewrites_only_the_child_checkpoint(self):
        parent_contract = {
            "algorithm": "q_learning", "seed": 11, "task": "coin_navigation",
            "feature_id": "discrete-v1", "feature_version": "v1",
            "feature_schema": feature_schema_contract("discrete-v1"),
            "reward_id": "r1", "reward_version": "r1", "reward_spec": REWARD_SPEC,
            "checkpoint_schema": "training-resume-v6",
            "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
            "training_device_type": "cpu", "training_device_name": None,
            "agent_seed": 11, "exploration_spec": EXPLORATION_SPEC,
            "safe_exploration": True, "safety_spec": SAFETY_SPEC,
            "n_step": 1, "retention_spec": RETENTION_SPEC,
            "network_spec": None, "hyperparameters": {},
        }
        child_contract = {
            **parent_contract,
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "training_budget": {"target_stage_action_steps": None, "min_rounds": 1},
            "performance_stopping": {"version": "task1-frozen-score-v1"},
        }
        validate_v6_migration(
            parent_contract, child_contract, parent_status="completed")
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "checkpoint_schema": "training-resume-v6",
                "training_budget": TRAINING_BUDGET,
                "q_table": {(0,) * 14: np.ones(6, dtype=np.float32)},
            }
            snapshot = LoadedSnapshot(
                run_directory=Path(directory), generation="generation-00000750",
                generation_hash="hash", algorithm="q_learning",
                task="coin_navigation", seed=11, round_index=750,
                source_commit="old", source_hash="old-hash",
                runner_state={"contract": parent_contract}, learner_payload=payload,
                learner_path=None,
            )
            destination = Path(directory) / "child.pkl"
            materialize_migrated_checkpoint(
                snapshot, destination,
                training_budget={"target_stage_action_steps": None, "min_rounds": 1})
            migrated = pickle.loads(destination.read_bytes())
            self.assertEqual(migrated["checkpoint_schema"], CHECKPOINT_SCHEMA_VERSION)
            self.assertIsNone(migrated["training_budget"]["target_stage_action_steps"])
            self.assertEqual(payload["checkpoint_schema"], "training-resume-v6")
        with self.assertRaisesRegex(ValueError, "v6 parent"):
            validate_v6_migration(
                {**parent_contract, "checkpoint_schema": "training-resume-v5"},
                child_contract, parent_status="completed")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_dqn_snapshot_materializes_a_weights_only_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "final.pt"
            torch.save({
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                "algorithm": "dqn",
                "actions": ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
                "feature_version": "v1",
                "feature_id": "discrete-v1",
                "feature_schema": feature_schema_contract("discrete-v1"),
                "network_spec": None,
                "hyperparameters": {},
                "reward_id": "r1",
                "reward_version": "r1",
                "reward_spec": REWARD_SPEC,
                "training_task": "coin_navigation",
                "training_device_type": "cuda",
                "training_device_name": "NVIDIA A100-PCIE-40GB",
                "agent_seed": 11,
                "exploration_spec": EXPLORATION_SPEC,
                "safe_exploration": True,
                "safe_exploration_decisions": 1,
                "safe_exploration_fallbacks": 0,
                "safety_spec": SAFETY_SPEC,
                "safety_decisions": 1,
                "safety_interventions": 0,
                "safety_fallbacks": 0,
                "action_history_state": {
                    "previous_action": None, "wait_streak": 0, "round": None,
                },
                "n_step": 1,
                "n_step_state": {"n_step": 1, "gamma": 0.95, "pending": []},
                "retention_spec": RETENTION_SPEC,
                "training_budget": TRAINING_BUDGET,
                "total_action_steps": 1,
                "stage_action_steps": 1,
                "policy": {"weight": torch.ones(1)},
                "target": {"weight": torch.ones(1)},
                "optimizer": {},
                "replay": {"transitions": [], "rng_state": random.Random(1).getstate()},
                "torch_rng_state": torch.get_rng_state(),
                "agent_rng_state": random.Random(2).getstate(),
                "updates": 0,
                "teacher": None,
            }, checkpoint)
            commit_training_snapshot(
                run, checkpoint, algorithm="dqn", task="coin_navigation", seed=11,
                round_index=1, world_rng_state={"state": 1},
                python_rng_state=random.Random(1).getstate(),
                numpy_rng_state=np.random.RandomState(1).get_state(),
                early_stopping_rewards=[1.0], source_commit="abc",
            )
            loaded = load_training_snapshot(run)
            destination = run / "restored.pt"
            materialize_learner_checkpoint(loaded, destination)
            restored = torch.load(destination, map_location="cpu", weights_only=True)
            self.assertEqual(restored["checkpoint_schema"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(CHECKPOINT_SCHEMA_VERSION, "training-resume-v11")
            self.assertIn("replay", restored)


if __name__ == "__main__":
    unittest.main()

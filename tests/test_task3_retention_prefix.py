import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.task3_retention_prefix import (
    GATE_FAILURE,
    INFRASTRUCTURE_ERROR,
    PASS,
    _summary_rows,
    evaluate_gates,
    main,
    verify_deterministic_prefix,
)


def metrics(**overrides):
    value = {
        "mean_score": 10.0,
        "mean_coins": 10.0,
        "mean_crates": 10.0,
        "mean_kills": 1.0,
        "first_place_rate": 0.5,
        "suicide_rate": 0.0,
        "bomb_survival_rate": 1.0,
        "zero_bomb_round_rate": 0.0,
        "invalid_action_rate": 0.0,
        "act_p95_seconds": 0.05,
        "act_max_seconds": 0.2,
        "act_timeouts": 0.0,
        "act_skipped": 0.0,
        "avoidable_escape_collapses": 0.0,
        "robust_guarantee_losses": 0.0,
        "robust_search_timeouts": 0.0,
    }
    value.update(overrides)
    return value


GATES = {
    "minimum_retention": 0.9,
    "task3_score_gain": 0.5,
    "task3_kill_gain": 0.1,
    "task3_first_place_gain": 0.05,
    "maximum_suicide_rate": 0.05,
    "minimum_bomb_survival_rate": 0.95,
    "maximum_zero_bomb_round_rate": 0.1,
    "maximum_invalid_rate": 0.01,
    "maximum_act_p95_seconds": 0.25,
    "maximum_act_seconds": 0.48,
}


class Task3RetentionPrefixTests(unittest.TestCase):
    def test_documented_script_invocation_imports_repository_modules(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "experiments/task3_retention_prefix.py", "--help"],
            cwd=root, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gate_passes_exact_boundaries_and_uses_combat_or(self):
        parent = {task: metrics() for task in ("task1", "task2", "task3")}
        child = {
            "task1": metrics(mean_score=9.0),
            "task2": metrics(mean_coins=9.0, mean_crates=9.0,
                             suicide_rate=0.05, bomb_survival_rate=0.95),
            "task3": metrics(
                mean_score=10.5, mean_coins=9.0, mean_crates=9.0,
                mean_kills=1.0, first_place_rate=0.55,
                suicide_rate=0.05, bomb_survival_rate=0.95,
                zero_bomb_round_rate=0.1, invalid_action_rate=0.01,
                act_p95_seconds=0.25, act_max_seconds=0.48,
            ),
        }
        passed, checks = evaluate_gates(parent, child, GATES)
        self.assertTrue(passed)
        self.assertTrue(checks["task3_combat_gain"]["passed"])

    def test_combat_boundary_tolerates_binary_float_representation(self):
        parent = {task: metrics() for task in ("task1", "task2", "task3")}
        parent["task3"]["first_place_rate"] = 0.45
        child = {
            "task1": metrics(),
            "task2": metrics(),
            "task3": metrics(mean_score=10.5, mean_kills=1.0,
                             first_place_rate=0.5),
        }
        passed, checks = evaluate_gates(parent, child, GATES)
        self.assertTrue(passed)
        self.assertEqual(checks["task3_combat_gain"]["first_place_gain"],
                         0.04999999999999999)
        self.assertTrue(checks["task3_combat_gain"]["passed"])

    def test_retention_failure_is_reported_without_rounding(self):
        parent = {task: metrics() for task in ("task1", "task2", "task3")}
        child = {
            "task1": metrics(),
            "task2": metrics(mean_coins=8.999999, mean_crates=10.0),
            "task3": metrics(mean_score=10.5, mean_kills=1.1),
        }
        passed, checks = evaluate_gates(parent, child, GATES)
        self.assertFalse(passed)
        self.assertFalse(checks["task2_mean_coins_retention"]["passed"])

    def test_prefix_verification_checks_episode_training_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"; reference = root / "reference"
            for run, run_id in ((candidate, "candidate"), (reference, "reference")):
                (run / "replays").mkdir(parents=True)
                episodes = [
                    {"run_id": run_id, "round_index": index, "agents": [{"score": index}]}
                    for index in (1, 2)
                ]
                (run / "episodes.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in episodes),
                    encoding="utf-8")
                with (run / "training.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("round", "loss", "checkpoint"))
                    writer.writeheader()
                    writer.writerow({"round": 1, "loss": 1.0, "checkpoint": run_id})
                    writer.writerow({"round": 2, "loss": 0.5, "checkpoint": run_id})
                with (run / "replays" / "round_00002.pt").open("wb") as handle:
                    pickle.dump({"actions": ["UP"], "field": np.arange(4)}, handle)
            result = verify_deterministic_prefix(candidate, reference, rounds=2)
            self.assertTrue(result["sampled_replay_equal"])
            with (candidate / "replays" / "round_00002.pt").open("wb") as handle:
                pickle.dump({"actions": ["WAIT"], "field": np.arange(4)}, handle)
            with self.assertRaisesRegex(ValueError, "reference replay"):
                verify_deterministic_prefix(candidate, reference, rounds=2)

    def test_incomplete_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            summary = root / "evaluation_summary"; summary.mkdir(parents=True)
            with (summary / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("run_id", "agent_name", "exploration_disabled",
                                "episode_count"),
                )
                writer.writeheader()
                writer.writerow({
                    "run_id": "evaluation_s1", "agent_name": "agent",
                    "exploration_disabled": True, "episode_count": 1,
                })
                writer.writerow({
                    "run_id": "AVERAGE", "agent_name": "agent",
                    "exploration_disabled": True, "episode_count": 1,
                })
            with self.assertRaisesRegex(ValueError, "unexpected seeds"):
                _summary_rows(root, "agent", (1, 2))

    def test_cli_exit_codes_distinguish_gate_and_infrastructure_failures(self):
        with mock.patch(
            "experiments.task3_retention_prefix.run_pipeline",
            return_value=(GATE_FAILURE, {}),
        ):
            self.assertEqual(
                main(("--manifest", "manifest.json")), GATE_FAILURE)
        with mock.patch(
            "experiments.task3_retention_prefix.run_pipeline",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(
                main(("--manifest", "manifest.json")), INFRASTRUCTURE_ERROR)
        self.assertNotEqual(PASS, GATE_FAILURE)


if __name__ == "__main__":
    unittest.main()

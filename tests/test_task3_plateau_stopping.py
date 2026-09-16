import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


from experiments.task3_plateau_stopping import (
    GATE_FAILURE,
    INFRASTRUCTURE_ERROR,
    PASS,
    PlateauTracker,
    resolve_plateau_spec,
    validate_plateau_training_run,
)


def assessment(round_index, score, *, eligible=True, failures=0, kills=0.5,
               first_place=0.5, suicide=0.0, retention=1.0, p95=0.04):
    return {
        "cumulative_round": round_index,
        "eligible": eligible,
        "failed_gate_count": failures,
        "checkpoint_run": f"checkpoint-{round_index}",
        "checkpoint_sha256": str(round_index) * 8,
        "task3_score": score,
        "task3_kills": kills,
        "task3_first_place_rate": first_place,
        "task3_suicide_rate": suicide,
        "task3_bomb_survival_rate": 1.0,
        "minimum_retention": retention,
        "act_p95_seconds": p95,
    }


class PlateauTrackerTests(unittest.TestCase):
    def setUp(self):
        self.spec = resolve_plateau_spec({
            "version": "task3-plateau-v1",
            "interval_rounds": 50,
            "max_total_rounds": 300,
            "minimum_score_improvement": 0.25,
            "qualified_patience": 2,
            "unqualified_patience": 3,
        })

    def test_qualified_plateau_rolls_back_to_best_checkpoint(self):
        tracker = PlateauTracker(self.spec)
        tracker.record(assessment(50, 7.0, kills=0.5))
        tracker.record(assessment(100, 7.1, kills=0.6))
        result = tracker.record(assessment(150, 6.8, kills=0.4))
        self.assertTrue(result["stop"])
        self.assertEqual(result["reason"], "qualified_plateau")
        self.assertEqual(result["selected_checkpoint_run"], "checkpoint-100")
        self.assertEqual(result["qualified_patience"], 2)

    def test_meaningful_improvement_resets_patience(self):
        tracker = PlateauTracker(self.spec)
        tracker.record(assessment(50, 7.0))
        tracker.record(assessment(100, 7.1))
        result = tracker.record(assessment(150, 7.3))
        self.assertFalse(result["stop"])
        self.assertEqual(result["qualified_patience"], 0)
        self.assertEqual(result["plateau_anchor_score"], 7.3)

    def test_tie_band_can_update_best_without_resetting_patience(self):
        tracker = PlateauTracker(self.spec)
        tracker.record(assessment(50, 7.0, kills=0.5))
        result = tracker.record(assessment(100, 7.1, kills=0.7))
        self.assertEqual(result["selected_checkpoint_run"], "checkpoint-100")
        self.assertEqual(result["qualified_patience"], 1)
        self.assertEqual(result["plateau_anchor_score"], 7.0)

    def test_unqualified_gate_progress_resets_failure_patience(self):
        tracker = PlateauTracker(self.spec)
        tracker.record(assessment(50, 4.0, eligible=False, failures=3))
        progressed = tracker.record(
            assessment(100, 4.1, eligible=False, failures=2))
        self.assertEqual(progressed["unqualified_patience"], 0)
        tracker.record(assessment(150, 4.2, eligible=False, failures=2))
        tracker.record(assessment(200, 4.15, eligible=False, failures=2))
        result = tracker.record(
            assessment(250, 4.0, eligible=False, failures=2))
        self.assertTrue(result["stop"])
        self.assertEqual(result["reason"], "unqualified_plateau")
        self.assertIsNone(result["selected_checkpoint_run"])

    def test_round_cap_returns_best_qualified_checkpoint_without_claiming_plateau(self):
        tracker = PlateauTracker(self.spec)
        result = None
        for round_index, score in zip(range(50, 301, 50), (
            5.0, 5.3, 5.6, 5.9, 6.2, 6.5,
        )):
            result = tracker.record(assessment(round_index, score))
        self.assertTrue(result["stop"])
        self.assertEqual(result["reason"], "budget_truncated_qualified")
        self.assertFalse(result["plateau_converged"])
        self.assertEqual(result["selected_checkpoint_run"], "checkpoint-300")

    def test_round_cap_without_qualified_checkpoint_fails(self):
        tracker = PlateauTracker(self.spec)
        result = None
        for round_index, score in zip(range(50, 301, 50), (
            4.0, 4.3, 4.6, 4.9, 5.2, 5.5,
        )):
            result = tracker.record(assessment(
                round_index, score, eligible=False, failures=1))
        self.assertTrue(result["stop"])
        self.assertEqual(result["reason"], "round_cap_unqualified")

    def test_history_replay_is_resumable_and_rejects_duplicate_rounds(self):
        history = [assessment(50, 7.0), assessment(100, 7.1)]
        resumed = PlateauTracker(self.spec, history=history)
        self.assertEqual(resumed.state["qualified_patience"], 1)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            resumed.record(assessment(100, 8.0))


class PlateauInterfaceTests(unittest.TestCase):
    def test_segment_validation_uses_cumulative_generation_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "resume").mkdir()
            (run / "metadata.json").write_text(json.dumps({
                "status": "completed",
                "termination": {
                    "local_completed_rounds": 50,
                    "cumulative_completed_rounds": 100,
                },
            }))
            (run / "resume" / "latest.json").write_text(json.dumps({
                "generation_hash": "latest-hash",
                "generations": ["generation-00000100", "generation-00000099"],
            }))
            with (run / "training.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("loss", "updates"))
                writer.writeheader()
                for round_index in range(50):
                    writer.writerow({"loss": 1 / (round_index + 1), "updates": round_index + 1})
            snapshot = type("Snapshot", (), {
                "generation": "unused", "generation_hash": "hash", "round_index": 0,
            })()
            with patch(
                "experiments.task3_plateau_stopping._load_generation",
                return_value=snapshot,
            ) as loader:
                result = validate_plateau_training_run(
                    run, local_rounds=50, cumulative_round=100)
            self.assertEqual(loader.call_args_list[0].args[1], "generation-00000100")
            self.assertEqual(loader.call_args_list[1].args[1], "generation-00000099")
            self.assertEqual(result["final_updates"], 50)

    def test_direct_script_help_works(self):
        result = subprocess.run(
            [sys.executable, "experiments/task3_plateau_stopping.py", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--resume", result.stdout)

    def test_manifest_uses_separated_seed_sets_and_fifty_round_boundaries(self):
        manifest = json.loads(Path(
            "experiments/task3_plateau_stopping.json").read_text())
        self.assertEqual(manifest["schema_version"], "task3-plateau-v1")
        self.assertEqual(manifest["training_checkpoints"], [50, 100, 150, 200, 250, 300])
        self.assertEqual(manifest["development_seeds"], list(range(19200, 19220)))
        self.assertEqual(manifest["confirmation_seeds"], list(range(19300, 19320)))
        self.assertEqual(manifest["main_validation_seeds"], list(range(19400, 19500)))
        self.assertTrue(set(manifest["development_seeds"]).isdisjoint(
            manifest["confirmation_seeds"]))
        self.assertTrue(set(manifest["confirmation_seeds"]).isdisjoint(
            manifest["main_validation_seeds"]))

    def test_cli_exit_codes_distinguish_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}")
            for expected in (PASS, GATE_FAILURE, INFRASTRUCTURE_ERROR):
                with self.subTest(expected=expected), patch(
                    "experiments.task3_plateau_stopping.run_pipeline",
                    return_value=(expected, {}),
                ):
                    from experiments.task3_plateau_stopping import main
                    self.assertEqual(main([
                        "--manifest", str(manifest),
                        "--project-root", directory,
                    ]), expected)


if __name__ == "__main__":
    unittest.main()

"""Public-contract tests for multi-run Task 1 training analysis."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from experiments.analyze_training import analyze_training, analyze_training_runs


class TrainingAnalysisTestCase(unittest.TestCase):
    def _make_run(
        self,
        root: Path,
        *,
        algorithm: str,
        seed: int,
        coins: list[int],
        task: str = "coin_navigation",
        feature_version: str = "v1",
        reward_version: str = "r1",
        missing_episode_round: int | None = None,
        non_finite_reward: bool = False,
        run_name: str | None = None,
        epsilon_offset: float = 0.0,
    ) -> Path:
        run = root / (run_name or f"{algorithm}-{seed}")
        run.mkdir()
        metadata = {
            "run_id": run.name,
            "algorithm": algorithm,
            "seed": seed,
            "task": task,
            "feature_version": feature_version,
            "reward_version": reward_version,
            "termination": {"requested_rounds": len(coins)},
        }
        (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with (run / "episodes.jsonl").open("w", encoding="utf-8") as file:
            for round_index, value in enumerate(coins, start=1):
                if round_index == missing_episode_round:
                    continue
                file.write(json.dumps({
                    "schema_version": "episode-v1",
                    "run_id": run.name,
                    "round_index": round_index,
                    "environment_seed": 1000 + seed,
                    "scenario": "coin-heaven",
                    "round_steps": 400,
                    "agents": [{
                        "name": f"{algorithm}_agent",
                        "score": value,
                        "coins": value,
                        "kills": 0,
                        "suicides": 0,
                        "crates": 0,
                        "bombs": 0,
                        "invalid": 0,
                        "survived": True,
                        "dead": False,
                    }],
                }) + "\n")
        with (run / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=(
                "schema_version", "algorithm", "round", "reward", "action_steps",
                "epsilon", "q_states", "loss", "updates", "checkpoint",
            ))
            writer.writeheader()
            for round_index, value in enumerate(coins, start=1):
                writer.writerow({
                    "schema_version": "training-v1",
                    "algorithm": algorithm,
                    "round": round_index,
                    "reward": "nan" if non_finite_reward else value / 2,
                    "action_steps": round_index * 400,
                    "epsilon": 1 - round_index / (2 * len(coins)) + epsilon_offset,
                    "q_states": round_index * 10 if algorithm == "q_learning" else "",
                    "loss": round_index / 100 if algorithm == "dqn" else "",
                    "updates": round_index * 5 if algorithm == "dqn" else "",
                    "checkpoint": "checkpoints/final.bin",
                })
        return run

    def test_multi_run_report_aligns_rounds_and_aggregates_by_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q_first = self._make_run(root, algorithm="q_learning", seed=11, coins=[1, 3, 5, 7])
            q_second = self._make_run(root, algorithm="q_learning", seed=22, coins=[3, 5, 7, 9])
            dqn_first = self._make_run(root, algorithm="dqn", seed=11, coins=[2, 4, 6, 8])
            dqn_second = self._make_run(root, algorithm="dqn", seed=22, coins=[4, 6, 8, 10])
            before = hashlib.sha256((q_first / "episodes.jsonl").read_bytes()).hexdigest()

            report = analyze_training_runs(
                [q_first, q_second, dqn_first, dqn_second], root / "analysis", rolling_window=2
            )

            self.assertEqual(len(report["round_metrics"]), 16)
            self.assertEqual(report["round_metrics"][2]["coins_rolling_mean"], 4.0)
            self.assertEqual(report["round_metrics"][3]["cumulative_coins"], 16)
            dqn_row = next(row for row in report["round_metrics"] if row["algorithm"] == "dqn")
            self.assertEqual(dqn_row["updates"], 5)
            self.assertEqual(dqn_row["dqn_last_update_loss"], 0.01)
            self.assertEqual(
                next(row for row in report["algorithm_summary"] if row["algorithm"] == "q_learning")
                ["mean_coins"],
                5.0,
            )
            self.assertAlmostEqual(
                next(row for row in report["algorithm_summary"] if row["algorithm"] == "q_learning")
                ["mean_coins_sample_sd"],
                1.4142135623730951,
            )
            self.assertEqual(hashlib.sha256((q_first / "episodes.jsonl").read_bytes()).hexdigest(), before)
            for filename in (
                "round_metrics.csv", "run_summary.csv", "algorithm_summary.csv",
                "analysis_metadata.json", "coins_per_round_q_learning.png",
                "coins_per_round_dqn.png", "coins_algorithm_comparison.png",
                "coins_first_vs_last100.png", "cumulative_coins.png",
                "reward_epsilon_progress.png", "learner_diagnostics.png",
            ):
                self.assertTrue((root / "analysis" / filename).is_file(), filename)

    def test_rejects_incompatible_or_invalid_inputs_before_reports_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._make_run(root, algorithm="q_learning", seed=11, coins=[1, 2])
            wrong_task = self._make_run(root, algorithm="dqn", seed=11, coins=[1, 2], task="classic")
            output = root / "analysis"
            with self.assertRaisesRegex(ValueError, "Task"):
                analyze_training_runs([first, wrong_task], output)
            self.assertFalse(output.exists())

            duplicate = self._make_run(
                root, algorithm="q_learning", seed=11, coins=[1, 2], run_name="q-learning-11-copy"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                analyze_training_runs([first, duplicate], output)

            invalid = self._make_run(
                root, algorithm="dqn", seed=22, coins=[1, 2], missing_episode_round=2
            )
            with self.assertRaisesRegex(ValueError, "round"):
                analyze_training_runs([first, invalid], output)

            non_finite = self._make_run(
                root, algorithm="dqn", seed=33, coins=[1, 2], non_finite_reward=True
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                analyze_training_runs([first, non_finite], output)

            different_schedule = self._make_run(
                root, algorithm="dqn", seed=44, coins=[1, 2], epsilon_offset=0.1
            )
            with self.assertRaisesRegex(ValueError, "Exploration schedules"):
                analyze_training_runs([first, different_schedule], output)

    def test_existing_single_run_analysis_remains_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._make_run(Path(temporary), algorithm="q_learning", seed=11, coins=[1, 3, 5])
            summary = analyze_training(run)
            self.assertEqual(summary["rounds"], 3)
            self.assertEqual(summary["final_q_states"], 30)
            self.assertTrue((run / "training_summary.json").is_file())

    @unittest.skipUnless(importlib.util.find_spec("nbformat"), "nbformat is installed with JupyterLab")
    def test_task1_notebook_has_a_valid_portable_structure(self):
        import nbformat

        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "task1_results.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        self.assertEqual(notebook.nbformat, 4)
        source = "\n".join("".join(cell["source"]) for cell in notebook.cells)
        self.assertIn("analyze_training_runs", source)
        self.assertNotIn("/Users/", source)


if __name__ == "__main__":
    unittest.main()

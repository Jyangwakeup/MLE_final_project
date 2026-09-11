import csv
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import settings as s
from experiments.run import (
    TASKS,
    _task_settings,
    run_agent_evaluation,
    run_agent_session,
)
from experiments.training import TrainingEarlyStopping, early_stopping_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "experiments" / "run.py"
BASE_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "base.json"
RUNS_ROOT = PROJECT_ROOT / "runs"


class ExperimentRunTest(unittest.TestCase):
    def setUp(self):
        self.outputs = []

    def tearDown(self):
        for output in self.outputs:
            if output.exists():
                shutil.rmtree(output)

    def output(self, suffix):
        path = RUNS_ROOT / f"test-experiment-{uuid.uuid4().hex}-{suffix}"
        self.outputs.append(path)
        return path

    def test_four_tasks_select_expected_scenario_and_agents(self):
        self.assertEqual(set(TASKS), {1, 2, 3, 4})
        self.assertEqual(_task_settings(1, None), ("coin_navigation", "coin-heaven", ()))
        self.assertEqual(_task_settings(2, None), ("crate_navigation", "classic", ()))
        self.assertEqual(
            _task_settings(3, None),
            ("weak_opponents", "classic", ("peaceful_agent", "coin_collector_agent")),
        )
        self.assertEqual(
            _task_settings(4, ["random_agent", "rule_based_agent"]),
            ("full_match", "classic", ("random_agent", "rule_based_agent")),
        )

    def test_task_constraints_reject_wrong_opponents(self):
        with self.assertRaises(ValueError):
            _task_settings(1, ["random_agent"])
        with self.assertRaises(ValueError):
            _task_settings(3, ["random_agent"])
        with self.assertRaises(ValueError):
            _task_settings(4, [])

    def test_training_writes_checkpoint_table_summary_and_chart(self):
        output = self.output("train")
        checkpoint = output / "checkpoints" / "final.pkl"
        with (
            patch.object(s, "MAX_STEPS", 3),
            patch.dict(os.environ, {"Q_LEARNING_ALLOW_BOMB": "false"}),
        ):
            run_agent_session(
                BASE_CONFIG, "train", 11, output, "q_learning_agent", (),
                "coin-heaven", 1, checkpoint, "coin_navigation", "sampled", 500,
            )

        self.assertTrue(checkpoint.is_file())
        self.assertTrue((output / "training_summary.json").is_file())
        self.assertTrue((output / "training_progress.png").is_file())
        self.assertTrue((output / "replays" / "round_00001.pt").is_file())
        replay_manifest = (output / "replays" / "manifest.jsonl").read_text()
        self.assertIn('"reason": "periodic_sample"', replay_manifest)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["expanded_config"]["execution"]["replay_policy"], "sampled")
        self.assertEqual(metadata["expanded_config"]["execution"]["replay_interval"], 500)
        with (output / "training.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["algorithm"], "q_learning")
        self.assertEqual(rows[0]["round"], "1")

    def test_training_early_stopping_detects_reward_plateau(self):
        output = self.output("early-stop-unit")
        output.mkdir()
        training_path = output / "training.csv"
        training_path.write_text("round,reward\n", encoding="utf-8")
        stopper = TrainingEarlyStopping(training_path, {
            "window": 2,
            "patience": 2,
            "min_rounds": 4,
            "min_delta": 0.1,
            "target_reward": 9.0,
        })

        decisions = []
        for round_index, reward in enumerate((10, 10, 10, 10), start=1):
            with training_path.open("a", encoding="utf-8") as file:
                file.write(f"{round_index},{reward}\n")
            decisions.append(stopper(round_index))

        self.assertEqual(decisions, [False, False, False, True])
        self.assertEqual(stopper.result["reason"], "rolling_mean_plateau")
        self.assertEqual(stopper.result["completed_rounds"], 4)

    def test_early_stopping_is_opt_in(self):
        self.assertIsNone(early_stopping_config({}))
        config = {"early_stopping": {"enabled": True, "window": 5}}
        self.assertEqual(early_stopping_config(config), config["early_stopping"])

    def test_evaluation_checks_checkpoint_before_creating_output(self):
        output = self.output("missing")
        result = subprocess.run(
            [
                sys.executable, str(RUNNER), "--config", str(BASE_CONFIG),
                "--mode", "evaluate", "--task", "1", "--agent",
                "q_learning_agent", "--checkpoint", str(output / "missing.pkl"),
                "--seed", "10001", "--n-rounds", "1", "--run-id", output.name,
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint does not exist", result.stderr)
        self.assertFalse(output.exists())

    def test_evaluation_is_frozen_and_writes_episode_results(self):
        training_output = self.output("source")
        evaluation_output = self.output("evaluation")
        checkpoint = training_output / "checkpoints" / "final.pkl"
        with (
            patch.object(s, "MAX_STEPS", 3),
            patch.dict(os.environ, {"Q_LEARNING_ALLOW_BOMB": "false"}),
        ):
            run_agent_session(
                BASE_CONFIG, "train", 11, training_output, "q_learning_agent", (),
                "coin-heaven", 1, checkpoint, "coin_navigation",
            )
            before = checkpoint.read_bytes()
            run_agent_session(
                BASE_CONFIG, "evaluate", 10001, evaluation_output,
                "q_learning_agent", (), "coin-heaven", 1, checkpoint,
                "coin_navigation",
            )

        self.assertEqual(checkpoint.read_bytes(), before)
        metadata = json.loads((evaluation_output / "metadata.json").read_text())
        self.assertEqual(metadata["mode"], "evaluate")
        self.assertEqual(metadata["expanded_config"]["execution"]["task"], "coin_navigation")
        self.assertTrue((evaluation_output / "episodes.jsonl").is_file())
        self.assertFalse((evaluation_output / "training.csv").exists())

    def test_multi_seed_evaluation_groups_runs_under_run_id_directory(self):
        output = self.output("multi-seed")
        checkpoint = output.parent / f"{output.name}-checkpoint.pkl"
        checkpoint.write_bytes(b"checkpoint")
        self.addCleanup(checkpoint.unlink, missing_ok=True)

        def fake_session(*args):
            run_output = args[3]
            run_output.mkdir()
            return run_output

        def fake_analysis(run_directories, summary):
            summary.mkdir()

        with (
            patch("experiments.run.run_agent_session", side_effect=fake_session),
            patch("experiments.run.analyze_runs", side_effect=fake_analysis),
        ):
            summary = run_agent_evaluation(
                BASE_CONFIG, (10001, 10002), 1, output.name,
                "q_learning_agent", (), "coin-heaven", checkpoint,
                "coin_navigation",
            )

        self.assertEqual(summary, output / f"{output.name}_summary")
        self.assertTrue((output / f"{output.name}_s10001").is_dir())
        self.assertTrue((output / f"{output.name}_s10002").is_dir())
        self.assertTrue((summary / "fixed_evaluation.json").is_file())


if __name__ == "__main__":
    unittest.main()

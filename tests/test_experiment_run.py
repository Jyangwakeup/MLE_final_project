import csv
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import settings as s
from experiments.run import (
    TASKS,
    _checkpoint_feature_contract,
    _local_python_dependencies,
    _source_hash,
    _task_settings,
    main as experiment_main,
    run_agent_evaluation,
    run_agent_session,
)
from experiments.training import (
    TrainingEarlyStopping,
    early_stopping_config,
    replay_progress_interval,
)
from experiments.devices import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "experiments" / "run.py"
BASE_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "reward_r2_balanced.json"
WARM_START_CONFIG = (
    PROJECT_ROOT / "experiments" / "configs" / "reward_r5_conditional_loop.json"
)
BASE_EARLY_STOPPING = json.loads(BASE_CONFIG.read_text())["training"]["early_stopping"]
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

    def test_source_dependencies_exclude_unrelated_agents(self):
        dependencies = _local_python_dependencies([
            PROJECT_ROOT / "experiments" / "run.py",
            *(PROJECT_ROOT / "agent_code" / "rainbow_lite_agent").glob("*.py"),
        ])
        self.assertIn(
            (PROJECT_ROOT / "agent_code/rainbow_lite_agent/callbacks.py").resolve(),
            dependencies,
        )
        self.assertIn(
            (PROJECT_ROOT / "agent_code/learning_common/runtime.py").resolve(),
            dependencies,
        )
        self.assertNotIn(
            (PROJECT_ROOT / "agent_code/double_q_lambda_agent/callbacks.py").resolve(),
            dependencies,
        )
        self.assertNotIn(
            (PROJECT_ROOT / "agent_code/learning_common/tile_coding.py").resolve(),
            dependencies,
        )

    def test_source_dependencies_include_relative_feature_imports(self):
        dependencies = _local_python_dependencies([
            PROJECT_ROOT / "agent_code/double_dqn_continuous_v2_agent/callbacks.py",
        ])
        self.assertIn(
            (PROJECT_ROOT / "agent_code/double_dqn_continuous_v2_agent/features.py").resolve(),
            dependencies,
        )
        self.assertIn(
            (PROJECT_ROOT / "agent_code/team_agent/feature_system/continuous_v2.py").resolve(),
            dependencies,
        )

    def test_agent_source_hash_is_independent_of_task_config(self):
        task1 = PROJECT_ROOT / "experiments/configs/expected_sarsa_lambda_task1.json"
        task2 = PROJECT_ROOT / "experiments/configs/expected_sarsa_lambda_task2.json"
        self.assertEqual(
            _source_hash("expected_sarsa_lambda_agent", task1),
            _source_hash("expected_sarsa_lambda_agent", task2),
        )

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
        self.assertEqual(
            _task_settings(4, None),
            ("full_match", "classic", ("rule_based_agent",) * 3),
        )

    def test_board_checkpoint_feature_contract_accepts_null_vector_shape(self):
        directory = self.output("board-contract")
        directory.mkdir()
        checkpoint = directory / "checkpoint.pkl"
        with checkpoint.open("wb") as file:
            pickle.dump({
                "feature_id": "board-v1",
                "feature_schema": {"vector_shape": None, "board_shape": [12, 17, 17]},
            }, file)

        contract = _checkpoint_feature_contract(checkpoint, "cnn_double_dqn")

        self.assertEqual(contract["feature_id"], "board-v1")
        self.assertIsNone(contract["runtime_adapter"])

    def test_task_constraints_reject_wrong_opponents(self):
        with self.assertRaises(ValueError):
            _task_settings(1, ["random_agent"])
        with self.assertRaises(ValueError):
            _task_settings(3, ["random_agent"])
        with self.assertRaises(ValueError):
            _task_settings(4, [])

    def test_cli_separates_training_resume_from_evaluation_checkpoint(self):
        train_output = self.output("invalid-train-checkpoint")
        eval_output = self.output("invalid-eval-resume")
        common = ["--config", str(BASE_CONFIG), "--task", "1", "--agent", "q_learning_agent"]
        with redirect_stderr(io.StringIO()):
            train_result = experiment_main([
                *common, "--mode", "train", "--n-rounds", "1", "--seed", "11",
                "--checkpoint", str(BASE_CONFIG), "--output", str(train_output),
            ])
            eval_result = experiment_main([
                *common, "--mode", "evaluate", "--checkpoint", str(BASE_CONFIG),
                "--resume-from", str(PROJECT_ROOT), "--output", str(eval_output),
            ])
        self.assertEqual(train_result, 2)
        self.assertEqual(eval_result, 2)
        self.assertFalse(train_output.exists())
        self.assertFalse(eval_output.exists())

    def test_cli_rejects_cuda_for_q_learning_before_creating_output(self):
        output = self.output("q-cuda")
        with redirect_stderr(io.StringIO()):
            result = experiment_main([
                "--config", str(BASE_CONFIG), "--mode", "train", "--task", "1",
                "--agent", "q_learning_agent", "--n-rounds", "1", "--seed", "11",
                "--device", "cuda", "--output", str(output),
            ])
        self.assertEqual(result, 2)
        self.assertFalse(output.exists())

    def test_cnn_training_uses_requested_cuda_device_when_available(self):
        with patch("experiments.devices.torch.cuda.is_available", return_value=True), patch(
            "experiments.devices.torch.cuda.get_device_name", return_value="Test GPU"
        ):
            device = resolve_device("cnn_double_dqn", "train", "cuda")
        self.assertEqual(device["actual"], "cuda:0")
        self.assertEqual(device["type"], "cuda")
        self.assertEqual(device["name"], "Test GPU")

    def test_cnn_evaluation_remains_cpu_when_cuda_is_requested(self):
        with self.assertRaisesRegex(ValueError, "evaluation is forced to CPU"):
            resolve_device("cnn_double_dqn", "evaluate", "cuda")

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
        with checkpoint.open("rb") as file:
            checkpoint_payload = pickle.load(file)
        self.assertEqual(checkpoint_payload["feature_id"], "discrete-q-v2")
        self.assertEqual(checkpoint_payload["feature_schema"]["state_shape"], [16])
        self.assertEqual(checkpoint_payload["actions"], [
            "UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB",
        ])
        self.assertEqual(checkpoint_payload["reward_id"], "r2_balanced")
        self.assertTrue((output / "training_summary.json").is_file())
        self.assertTrue((output / "training_progress.png").is_file())
        self.assertTrue((output / "replays" / "round_00001.pt").is_file())
        replay_manifest = (output / "replays" / "manifest.jsonl").read_text()
        self.assertIn('"reason": "periodic_sample"', replay_manifest)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["expanded_config"]["execution"]["replay_policy"], "sampled")
        self.assertEqual(metadata["expanded_config"]["execution"]["replay_interval"], 500)
        self.assertFalse(metadata["expanded_config"]["execution"]["allow_bomb"])
        self.assertEqual(metadata["agent_seed"], 11)
        self.assertEqual(metadata["exploration_spec"], {
            "version": "linear-v1",
            "start": 1.0,
            "end": 0.05,
            "decay_action_steps": 80_000,
        })
        self.assertEqual(
            metadata["expanded_config"]["training"]["exploration"],
            metadata["exploration_spec"],
        )
        self.assertEqual(metadata["reward_version"], "r2_balanced")
        self.assertEqual(metadata["feature_id"], "discrete-q-v2")
        self.assertEqual(metadata["feature_schema"]["state_shape"], [16])
        self.assertEqual(metadata["feature_schema"]["vector_shape"], [50])
        self.assertEqual(
            metadata["action_order"],
            ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"],
        )
        self.assertEqual(metadata["reward_id"], "r2_balanced")
        self.assertEqual(metadata["checkpoint"], str(checkpoint.resolve()))
        self.assertEqual(metadata["rewards"]["invalid_action"], -0.2)
        self.assertEqual(len(metadata["source_hash"]), 64)
        self.assertEqual(
            metadata["config_source_sha256"],
            hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest(),
        )
        with checkpoint.open("rb") as file:
            payload = pickle.load(file)
        self.assertEqual(payload["agent_seed"], 11)
        self.assertEqual(payload["exploration_spec"], metadata["exploration_spec"])
        with (output / "training.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["algorithm"], "q_learning")
        self.assertEqual(rows[0]["round"], "1")

    def test_cli_overrides_feature_reward_and_stops_at_stage_action_budget(self):
        output = self.output("objective-budget")
        with patch.object(s, "MAX_STEPS", 3):
            result = experiment_main([
                "--config", str(PROJECT_ROOT / "experiments/configs/pre_task3_task1.json"),
                "--mode", "train", "--task", "1",
                "--agent", "q_learning_agent", "--feature-id", "discrete-objective-v1",
                "--reward-id", "r5_coin_potential", "--n-rounds", "5",
                "--target-stage-action-steps", "2", "--min-rounds", "1",
                "--seed", "11", "--device", "cpu", "--output", str(output),
                "--replay-policy", "none",
            ])
        self.assertEqual(result, 0)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["feature_id"], "discrete-objective-v1")
        self.assertEqual(metadata["feature_schema"]["vector_shape"], [60])
        self.assertEqual(metadata["reward_id"], "r5_coin_potential")
        self.assertTrue(metadata["termination"]["action_budget_reached"])
        self.assertEqual(
            metadata["termination"]["completion_reason"],
            "stage_action_target_reached",
        )
        self.assertLess(metadata["termination"]["completed_rounds"], 5)
        with (output / "checkpoints" / "final.pkl").open("rb") as file:
            checkpoint = pickle.load(file)
        self.assertEqual(checkpoint["checkpoint_schema"], "training-resume-v11")
        self.assertTrue(checkpoint["safe_exploration"])

    def test_objective_dqn_runner_writes_a_v11_60_dimensional_checkpoint(self):
        output = self.output("objective-dqn")
        checkpoint = output / "checkpoints" / "final.pt"
        with patch.object(s, "MAX_STEPS", 3):
            run_agent_session(
                PROJECT_ROOT / "experiments/configs/pre_task3_task1.json",
                "train", 11, output, "dqn_agent", (), "coin-heaven", 1,
                checkpoint, "coin_navigation", "none", 1,
                action_budget_config={
                    "target_stage_action_steps": 2, "min_rounds": 1},
                safe_exploration=True,
                feature_id_override="discrete-objective-v1",
                reward_id_override="r5_coin_potential",
            )

        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["checkpoint_schema"], "training-resume-v11")
        self.assertEqual(payload["feature_schema"]["vector_shape"], [60])
        self.assertEqual(payload["policy"]["layers.0.weight"].shape[1], 60)
        self.assertEqual(payload["network_spec"]["input_shape"], [60])
        self.assertEqual(payload["hyperparameters"]["batch_size"], 64)
        self.assertEqual(payload["reward_id"], "r5_coin_potential")
        self.assertTrue(payload["safe_exploration"])
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["feature_id"], "discrete-objective-v1")

    def test_training_can_resume_into_an_immutable_child_run(self):
        parent = self.output("resume-parent")
        child = self.output("resume-child")
        self.outputs.remove(child)
        checkpoint = parent / "checkpoints" / "final.pkl"
        with patch.object(s, "MAX_STEPS", 3):
            run_agent_session(
                BASE_CONFIG, "train", 11, parent, "q_learning_agent", (),
                "coin-heaven", 1, checkpoint, "coin_navigation", "none", 1,
                BASE_EARLY_STOPPING,
            )
            result = experiment_main([
                "--config", str(BASE_CONFIG), "--mode", "train", "--task", "1",
                "--agent", "q_learning_agent", "--n-rounds", "1", "--seed", "11",
                "--resume-from", str(parent), "--output", str(child),
                "--replay-policy", "none",
            ])

        self.assertEqual(result, 0)
        self.outputs.append(child)
        metadata = json.loads((child / "metadata.json").read_text())
        self.assertEqual(metadata["lineage"]["resume_kind"], "same_task")
        self.assertEqual(metadata["lineage"]["parent_run"], str(parent.resolve()))
        self.assertEqual(metadata["termination"]["local_completed_rounds"], 1)
        self.assertEqual(metadata["termination"]["cumulative_completed_rounds"], 2)
        episode = json.loads((child / "episodes.jsonl").read_text().splitlines()[0])
        self.assertEqual(episode["round_index"], 2)

    def test_training_can_warm_start_policy_with_a_new_reward(self):
        import torch

        parent = self.output("warm-parent")
        child = self.output("warm-child")
        self.outputs.remove(child)
        source_checkpoint = parent / "checkpoints" / "final.pt"
        with patch.object(s, "MAX_STEPS", 3):
            run_agent_session(
                BASE_CONFIG, "train", 11, parent,
                "double_dqn_continuous_agent", (), "coin-heaven", 1,
                source_checkpoint, "coin_navigation", "none", 1,
            )
            result = experiment_main([
                "--config", str(WARM_START_CONFIG), "--mode", "train",
                "--task", "2", "--agent", "double_dqn_continuous_agent",
                "--n-rounds", "1", "--seed", "11",
                "--init-from-checkpoint", str(source_checkpoint),
                "--output", str(child), "--replay-policy", "none",
            ])

        self.assertEqual(result, 0)
        self.outputs.append(child)
        metadata = json.loads((child / "metadata.json").read_text())
        self.assertEqual(metadata["lineage"]["kind"], "warm_start")
        self.assertEqual(metadata["lineage"]["inherited"], ["policy_weights"])
        self.assertEqual(metadata["reward_id"], "r5_conditional_loop")
        child_checkpoint = torch.load(
            child / "checkpoints" / "final.pt", map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(child_checkpoint["reward_id"], "r5_conditional_loop")
        self.assertLessEqual(child_checkpoint["action_steps"], 3)

    def test_same_task_resume_matches_uninterrupted_next_round(self):
        continuous = self.output("continuous")
        parent = self.output("split-parent")
        child = self.output("split-child")
        self.outputs.remove(child)
        with patch.object(s, "MAX_STEPS", 3):
            run_agent_session(
                BASE_CONFIG, "train", 11, continuous, "q_learning_agent", (),
                "coin-heaven", 2, continuous / "checkpoints" / "final.pkl",
                "coin_navigation", "all", 1, BASE_EARLY_STOPPING,
            )
            run_agent_session(
                BASE_CONFIG, "train", 11, parent, "q_learning_agent", (),
                "coin-heaven", 1, parent / "checkpoints" / "final.pkl",
                "coin_navigation", "all", 1, BASE_EARLY_STOPPING,
            )
            result = experiment_main([
                "--config", str(BASE_CONFIG), "--mode", "train", "--task", "1",
                "--agent", "q_learning_agent", "--n-rounds", "1", "--seed", "11",
                "--resume-from", str(parent), "--output", str(child),
                "--replay-policy", "all",
            ])
        self.assertEqual(result, 0)
        self.outputs.append(child)

        with (continuous / "replays" / "round_00002.pt").open("rb") as file:
            uninterrupted_replay = pickle.load(file)
        with (child / "replays" / "round_00002.pt").open("rb") as file:
            resumed_replay = pickle.load(file)
        np.testing.assert_array_equal(
            uninterrupted_replay["arena"], resumed_replay["arena"]
        )
        self.assertEqual(uninterrupted_replay["coins"], resumed_replay["coins"])
        self.assertEqual(uninterrupted_replay["actions"], resumed_replay["actions"])
        self.assertEqual(
            [list(value) for value in uninterrupted_replay["permutations"]],
            [list(value) for value in resumed_replay["permutations"]],
        )
        with (continuous / "checkpoints" / "final.pkl").open("rb") as file:
            uninterrupted_model = pickle.load(file)
        with (child / "checkpoints" / "final.pkl").open("rb") as file:
            resumed_model = pickle.load(file)
        self.assertEqual(
            uninterrupted_model["training_steps"], resumed_model["training_steps"]
        )
        self.assertEqual(
            uninterrupted_model["agent_rng_state"], resumed_model["agent_rng_state"]
        )
        self.assertEqual(set(uninterrupted_model["q_table"]), set(resumed_model["q_table"]))
        for key in uninterrupted_model["q_table"]:
            np.testing.assert_array_equal(
                uninterrupted_model["q_table"][key], resumed_model["q_table"][key]
            )

    def test_direct_next_task_inherits_learning_but_restarts_world_rounds(self):
        parent = self.output("promotion-parent")
        child = self.output("promotion-child")
        self.outputs.remove(child)
        with patch.object(s, "MAX_STEPS", 3):
            run_agent_session(
                BASE_CONFIG, "train", 11, parent, "q_learning_agent", (),
                "coin-heaven", 1, parent / "checkpoints" / "final.pkl",
                "coin_navigation", "none", 1,
            )
            with (parent / "checkpoints" / "final.pkl").open("rb") as file:
                parent_steps = pickle.load(file)["training_steps"]
            result = experiment_main([
                "--config", str(BASE_CONFIG), "--mode", "train", "--task", "2",
                "--agent", "q_learning_agent", "--n-rounds", "1", "--seed", "11",
                "--resume-from", str(parent), "--output", str(child),
                "--replay-policy", "none",
            ])
        self.assertEqual(result, 0)
        self.outputs.append(child)
        metadata = json.loads((child / "metadata.json").read_text())
        self.assertEqual(metadata["lineage"]["resume_kind"], "next_task")
        episode = json.loads((child / "episodes.jsonl").read_text().splitlines()[0])
        self.assertEqual(episode["round_index"], 1)
        with (child / "checkpoints" / "final.pkl").open("rb") as file:
            child_model = pickle.load(file)
        self.assertGreater(child_model["training_steps"], parent_steps)
        self.assertEqual(child_model["training_task"], "crate_navigation")

    def test_corrupt_latest_snapshot_falls_back_and_corrects_lineage_rounds(self):
        parent = self.output("fallback-parent")
        child = self.output("fallback-child")
        self.outputs.remove(child)
        with patch.object(s, "MAX_STEPS", 3):
            run_agent_session(
                BASE_CONFIG, "train", 11, parent, "q_learning_agent", (),
                "coin-heaven", 2, parent / "checkpoints" / "final.pkl",
                "coin_navigation", "none", 1, BASE_EARLY_STOPPING,
            )
            latest = json.loads((parent / "resume" / "latest.json").read_text())
            newest = parent / "resume" / latest["generations"][0] / "q_table.npz"
            newest.write_bytes(b"corrupt")
            crashed_metadata = json.loads((parent / "metadata.json").read_text())
            crashed_metadata["status"] = "running"
            crashed_metadata.pop("termination", None)
            (parent / "metadata.json").write_text(
                json.dumps(crashed_metadata), encoding="utf-8"
            )
            result = experiment_main([
                "--config", str(BASE_CONFIG), "--mode", "train", "--task", "1",
                "--agent", "q_learning_agent", "--n-rounds", "1", "--seed", "11",
                "--resume-from", str(parent), "--output", str(child),
                "--replay-policy", "none",
            ])

        self.assertEqual(result, 0)
        self.outputs.append(child)
        metadata = json.loads((child / "metadata.json").read_text())
        self.assertEqual(metadata["lineage"]["fallback_lost_rounds"], 1)
        self.assertIn("integrity validation", metadata["lineage"]["fallback_reason"])
        self.assertEqual(metadata["lineage"]["parent_generation"], "generation-00000001")
        self.assertEqual(metadata["termination"]["cumulative_completed_rounds"], 2)
        episode = json.loads((child / "episodes.jsonl").read_text().splitlines()[0])
        self.assertEqual(episode["round_index"], 2)

    def test_training_early_stopping_detects_reward_plateau(self):
        output = self.output("early-stop-unit")
        output.mkdir()
        training_path = output / "training.csv"
        training_path.write_text("round,reward,action_steps\n", encoding="utf-8")
        stopper = TrainingEarlyStopping(training_path, {
            "window": 2,
            "patience": 2,
            "min_rounds": 4,
            "min_delta": 0.1,
            "target_reward": 9.0,
            "require_exploration_complete": True,
            "min_action_steps": 4,
        })

        decisions = []
        for round_index, reward in enumerate((10, 10, 10, 10), start=1):
            with training_path.open("a", encoding="utf-8") as file:
                file.write(f"{round_index},{reward},{round_index}\n")
            decisions.append(stopper(round_index))

        self.assertEqual(decisions, [False, False, False, True])
        self.assertEqual(stopper.result["reason"], "rolling_mean_plateau")
        self.assertEqual(stopper.result["completed_rounds"], 4)

    def test_early_stopping_waits_for_exploration_completion(self):
        output = self.output("early-stop-exploration")
        output.mkdir()
        training_path = output / "training.csv"
        training_path.write_text(
            "round,reward,action_steps\n", encoding="utf-8")
        stopper = TrainingEarlyStopping(training_path, {
            "window": 2,
            "patience": 1,
            "min_rounds": 3,
            "min_delta": 0.1,
            "target_reward": None,
            "require_exploration_complete": True,
            "min_action_steps": 10,
        })

        decisions = []
        for round_index, action_steps in enumerate((1, 2, 3, 10), start=1):
            with training_path.open("a", encoding="utf-8") as file:
                file.write(f"{round_index},1.0,{action_steps}\n")
            decisions.append(stopper(round_index))

        self.assertEqual(decisions, [False, False, False, True])
        self.assertEqual(stopper.result["action_steps"], 10)

    def test_early_stopping_is_enabled_by_default(self):
        self.assertEqual(early_stopping_config({}), {
            "enabled": True,
            "window": 200,
            "patience": 100,
            "min_rounds": 300,
            "min_delta": 0.1,
            "target_reward": None,
            "require_exploration_complete": True,
        })
        config = {"early_stopping": {"enabled": True, "window": 5}}
        self.assertEqual(early_stopping_config(config)["window"], 5)
        self.assertIsNone(early_stopping_config({
            "early_stopping": {"enabled": False},
        }))

    def test_training_replay_interval_tracks_progress_milestones(self):
        self.assertEqual(replay_progress_interval(1), 1)
        self.assertEqual(replay_progress_interval(1_000), 100)
        self.assertEqual(replay_progress_interval(10_000), 1_000)

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
        self.assertEqual(
            metadata["checkpoint_reward_contract"]["reward_version"], "r2_balanced",
        )
        self.assertEqual(metadata["expanded_config"]["execution"]["task"], "coin_navigation")
        self.assertTrue((evaluation_output / "episodes.jsonl").is_file())
        self.assertFalse((evaluation_output / "training.csv").exists())

    def test_multi_seed_evaluation_groups_runs_under_run_id_directory(self):
        output = self.output("multi-seed")
        checkpoint = output.parent / f"{output.name}-checkpoint.pkl"
        checkpoint.write_bytes(b"checkpoint")
        self.addCleanup(checkpoint.unlink, missing_ok=True)

        def fake_session(*args, **kwargs):
            self.assertEqual(kwargs["device_info"]["actual"], "cpu")
            self.assertFalse(kwargs["show_progress"])
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

    def test_legal_random_baseline_runs_through_the_frozen_cli(self):
        output = self.output("legal-random")
        diagnostic_output = self.output("legal-random-diagnostics")
        marker = PROJECT_ROOT / "agent_code" / "legal_random_agent" / "baseline.json"
        with patch.object(s, "MAX_STEPS", 3):
            result = experiment_main([
                "--config", str(PROJECT_ROOT / "experiments/configs/stage_gate.json"),
                "--mode", "evaluate", "--task", "1",
                "--agent", "legal_random_agent", "--n-rounds", "1",
                "--seed", "10000", "--device", "cpu",
                "--checkpoint", str(marker), "--output", str(output),
            ])
            diagnostic_result = experiment_main([
                "--config", str(
                    PROJECT_ROOT / "experiments/configs/stage_gate_coin3.json"),
                "--mode", "evaluate", "--task", "1",
                "--agent", "legal_random_agent", "--n-rounds", "1",
                "--seed", "10000", "--device", "cpu",
                "--checkpoint", str(marker), "--output", str(diagnostic_output),
            ])

        self.assertEqual(result, 0)
        self.assertEqual(diagnostic_result, 0)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["algorithm"], "legal_random")
        self.assertEqual(metadata["agent_seed"], 10000)
        actions = [json.loads(line)["action"] for line in (
            output / "timing.jsonl"
        ).read_text().splitlines()]
        diagnostic_records = [json.loads(line) for line in (
            diagnostic_output / "timing.jsonl"
        ).read_text().splitlines()]
        diagnostic_actions = [record["action"] for record in diagnostic_records]
        self.assertEqual(actions, diagnostic_actions)
        self.assertTrue(all("navigation" in record for record in diagnostic_records))
        self.assertNotIn("BOMB", actions)

    def test_official_baselines_run_without_checkpoints_and_emit_metrics(self):
        for agent in (
            "random_agent", "rule_based_agent", "peaceful_agent",
            "coin_collector_agent",
        ):
            with self.subTest(agent=agent):
                output = self.output(f"official-{agent}")
                with patch.object(s, "MAX_STEPS", 3):
                    result = experiment_main([
                        "--config", str(
                            PROJECT_ROOT / "experiments/configs/stage_gate_coin3.json"),
                        "--mode", "evaluate", "--task", "1",
                        "--agent", agent, "--n-rounds", "1",
                        "--seed", "10000", "--device", "cpu",
                        "--output", str(output),
                    ])

                self.assertEqual(result, 0)
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(metadata["algorithm"], "official_baseline")
                self.assertIsNone(metadata["checkpoint"])
                self.assertTrue((output / "episodes.jsonl").is_file())
                self.assertTrue((output / "timing.jsonl").is_file())
                self.assertTrue((output / "summary" / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()

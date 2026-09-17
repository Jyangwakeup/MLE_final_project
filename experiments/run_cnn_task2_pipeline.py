"""Train and evaluate the preregistered CNN Task-2 transfer/control experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs"
AGENT = "cnn_distilled_double_dqn_agent"
DEVELOPMENT_SEEDS = tuple(range(10000, 10020))
MAIN_SEEDS = tuple(range(11000, 11100))
SNAPSHOT_STEPS = (50_000, 100_000, 150_000, 200_000)
TRAINING_SEEDS = (11, 22, 33)
D01_CRATE_BASELINE = 34.35


def _run(arguments):
    subprocess.run(
        [sys.executable, "-u", "-m", "experiments.run", *arguments],
        cwd=PROJECT_ROOT, check=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _average(run_id):
    paths = list((RUNS / run_id).glob("*_summary/summary.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one aggregate summary for {run_id}; got {paths}")
    with paths[0].open(newline="", encoding="utf-8") as file:
        matches = [row for row in csv.DictReader(file) if row["run_id"] == "AVERAGE"]
    if len(matches) != 1:
        raise RuntimeError(f"aggregate summary for {run_id} has no AVERAGE row")
    return matches[0]


def _evaluate(checkpoint, run_id, config, task, seeds, *, reward_id):
    _run([
        "--config", str(config), "--mode", "evaluate", "--task", str(task),
        "--agent", AGENT, "--reward-id", reward_id,
        "--checkpoint", str(checkpoint), "--device", "cpu", "--n-rounds", "1",
        "--seeds", *(str(seed) for seed in seeds), "--run-id", run_id,
        "--replay-policy", "none",
    ])
    return _average(run_id)


def _number(metrics, name, default=0.0):
    value = metrics.get(name)
    return default if value in (None, "") else float(value)


def _gate(task1, task2, parent_task1_score, parent_task2_crates):
    score = _number(task1, "mean_score")
    retention = score / parent_task1_score if parent_task1_score > 0 else 0.0
    checks = {
        "task1_mean_score": score >= 48.0,
        "task1_retention": retention >= 0.90,
        "task2_d01_crate_floor": (
            _number(task2, "mean_crates") >= D01_CRATE_BASELINE),
        "task2_suicide_rate": _number(task2, "suicide_rate") <= 0.05,
        "zero_bomb_round_rate": _number(task2, "zero_bomb_round_rate") <= 0.10,
        "survived_bomb_rate": _number(task2, "survived_bomb_rate") >= 0.95,
        "invalid_action_rate": _number(task2, "invalid_action_rate") <= 0.01,
        "act_p95_seconds": _number(task2, "act_p95_time", float("inf")) < 0.05,
        "act_max_seconds": _number(task2, "act_max_time", float("inf")) < 0.5,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "task1_retention": retention}


def _rank(candidate):
    task1, task2 = candidate["task1"], candidate["task2"]
    return (
        int(candidate["gate"]["passed"]),
        -_number(task2, "suicide_rate"),
        -_number(task2, "zero_bomb_round_rate"),
        _number(task2, "survived_bomb_rate"),
        _number(task2, "mean_coins"),
        _number(task2, "all_coins_rate"),
        _number(task2, "mean_crates"),
        _number(task2, "crates_per_survived_bomb"),
        -_number(task2, "wait_rate"),
        -_number(task2, "long_wait_issue_rate"),
        candidate["gate"]["task1_retention"],
        -_number(task2, "act_p95_time", float("inf")),
        -int(candidate.get("step", 0)),
    )


def _write_config(base, destination, dataset, kl_weight):
    document = json.loads(Path(base).read_text(encoding="utf-8"))
    document["cnn_distillation"]["dataset"] = None if dataset is None else str(dataset)
    document["cnn_distillation"]["teacher_kl_weight"] = float(kl_weight)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return destination


def _train(
        config, run_id, seed, initial=None,
        *, reward_id="r7_safe_credit_sparse"):
    arguments = [
        "--config", str(config), "--mode", "train", "--task", "2",
        "--agent", AGENT, "--reward-id", reward_id,
        "--device", "cuda", "--seed", str(seed), "--n-rounds", "2000",
        "--target-stage-action-steps", "200000", "--min-rounds", "500",
        "--run-id", run_id, "--replay-policy", "sampled",
    ]
    if initial is not None:
        arguments.extend(["--init-from-checkpoint", str(initial)])
    _run(arguments)


def _evaluate_snapshots(
        family, seed, run_id, config, parent_score, parent_crates, work_directory,
        *, reward_id="r7_safe_credit_sparse"):
    candidates = []
    snapshots = RUNS / run_id / "checkpoints" / "snapshots"
    for step in SNAPSHOT_STEPS:
        checkpoint = snapshots / f"policy_{step:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing planned checkpoint: {checkpoint}")
        prefix = f"{work_directory.name}_{family}_s{seed}_{step}"
        task1 = _evaluate(
            checkpoint, prefix + "_t1", config, 1, DEVELOPMENT_SEEDS,
            reward_id=reward_id)
        task2 = _evaluate(
            checkpoint, prefix + "_t2", config, 2, DEVELOPMENT_SEEDS,
            reward_id=reward_id)
        candidate = {
            "family": family, "training_seed": seed, "step": step,
            "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
            "task1": task1, "task2": task2,
            "gate": _gate(task1, task2, parent_score, parent_crates),
        }
        candidates.append(candidate)
    best = max(candidates, key=_rank)
    return {"candidates": candidates, "best": best}


def pipeline(task1_checkpoint, teacher_dataset, work_directory, base_config):
    task1_checkpoint = Path(task1_checkpoint).resolve()
    teacher_dataset = Path(teacher_dataset).resolve()
    work_directory = Path(work_directory).resolve()
    if work_directory.exists():
        raise FileExistsError(f"refusing to overwrite pipeline directory: {work_directory}")
    if not task1_checkpoint.is_file() or not teacher_dataset.is_file():
        raise FileNotFoundError("Task-1 checkpoint and teacher dataset must exist")
    work_directory.mkdir(parents=True)
    transfer_config = _write_config(
        base_config, work_directory / "transfer_config.json", teacher_dataset, 2.0)
    scratch_config = _write_config(
        base_config, work_directory / "scratch_config.json", None, 0.0)

    parent_task1 = _evaluate(
        task1_checkpoint, f"{work_directory.name}_parent_t1", transfer_config,
        1, DEVELOPMENT_SEEDS, reward_id="r5_conditional_loop")
    parent_task2 = _evaluate(
        task1_checkpoint, f"{work_directory.name}_parent_t2", transfer_config,
        2, DEVELOPMENT_SEEDS, reward_id="r5_conditional_loop")
    parent_score = _number(parent_task1, "mean_score")
    parent_crates = _number(parent_task2, "mean_crates")

    results = {
        "contract": {
            "task1_checkpoint": str(task1_checkpoint),
            "task1_checkpoint_sha256": _sha256(task1_checkpoint),
            "teacher_dataset": str(teacher_dataset),
            "teacher_dataset_sha256": _sha256(teacher_dataset),
            "training_seeds": list(TRAINING_SEEDS),
            "development_seeds": [DEVELOPMENT_SEEDS[0], DEVELOPMENT_SEEDS[-1]],
            "main_seeds": [MAIN_SEEDS[0], MAIN_SEEDS[-1]],
            "target_action_steps": 200_000, "minimum_rounds": 500,
        },
        "parent": {"task1": parent_task1, "task2": parent_task2},
        "transfer": {}, "scratch": {}, "main_validation": None,
    }

    transfer_run = f"{work_directory.name}_transfer_s11"
    _train(transfer_config, transfer_run, 11, task1_checkpoint)
    results["transfer"]["11"] = _evaluate_snapshots(
        "transfer", 11, transfer_run, transfer_config, parent_score,
        parent_crates, work_directory)

    scratch_run = f"{work_directory.name}_scratch_s11"
    _train(scratch_config, scratch_run, 11)
    results["scratch"]["11"] = _evaluate_snapshots(
        "scratch", 11, scratch_run, scratch_config, parent_score,
        parent_crates, work_directory)

    if results["transfer"]["11"]["best"]["gate"]["passed"]:
        for seed in TRAINING_SEEDS[1:]:
            run_id = f"{work_directory.name}_transfer_s{seed}"
            _train(transfer_config, run_id, seed, task1_checkpoint)
            results["transfer"][str(seed)] = _evaluate_snapshots(
                "transfer", seed, run_id, transfer_config, parent_score,
                parent_crates, work_directory)
        selected = max(
            (entry["best"] for entry in results["transfer"].values()), key=_rank)
        best_path = work_directory / "best_task2.pt"
        shutil.copy2(selected["checkpoint"], best_path)
        main_parent_task1 = _evaluate(
            task1_checkpoint, f"{work_directory.name}_main_parent_t1",
            transfer_config, 1, MAIN_SEEDS,
            reward_id="r5_conditional_loop")
        main_parent_task2 = _evaluate(
            task1_checkpoint, f"{work_directory.name}_main_parent_t2",
            transfer_config, 2, MAIN_SEEDS,
            reward_id="r5_conditional_loop")
        main_task1 = _evaluate(
            best_path, f"{work_directory.name}_main_t1", transfer_config, 1,
            MAIN_SEEDS, reward_id="r7_safe_credit_sparse")
        main_task2 = _evaluate(
            best_path, f"{work_directory.name}_main_t2", transfer_config, 2,
            MAIN_SEEDS, reward_id="r7_safe_credit_sparse")
        results["selected"] = {**selected, "published_checkpoint": str(best_path),
                               "published_sha256": _sha256(best_path)}
        results["main_validation"] = {
            "parent_task1": main_parent_task1,
            "parent_task2": main_parent_task2,
            "task1": main_task1, "task2": main_task2,
            "gate": _gate(
                main_task1, main_task2,
                _number(main_parent_task1, "mean_score"),
                _number(main_parent_task2, "mean_crates")),
        }

    (work_directory / "selection.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-checkpoint", type=Path, default=(
        PROJECT_ROOT / "agent_code/cnn_distilled_double_dqn_agent/final.pt"))
    parser.add_argument("--teacher-dataset", type=Path, default=(
        PROJECT_ROOT / "runs/cnn_distilled_t1_j471456/teacher_task1.npz"))
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        PROJECT_ROOT / "experiments/configs/task2_cnn_distillation.json"))
    args = parser.parse_args()
    print(json.dumps(pipeline(
        args.task1_checkpoint, args.teacher_dataset, args.work_directory, args.config),
        indent=2))


if __name__ == "__main__":
    main()

"""Run the preregistered Task-2 avoidable-WAIT shaping ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .run_cnn_task2_pipeline import (
    MAIN_SEEDS, PROJECT_ROOT, TRAINING_SEEDS, _average, _evaluate,
    _evaluate_snapshots, _gate, _number, _rank, _sha256, _train, _write_config,
)


def pipeline(
        task1_checkpoint, teacher_dataset, work_directory, base_config,
        baseline_selection):
    task1_checkpoint = Path(task1_checkpoint).resolve()
    teacher_dataset = Path(teacher_dataset).resolve()
    work_directory = Path(work_directory).resolve()
    baseline_selection = Path(baseline_selection).resolve()
    if work_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite pipeline directory: {work_directory}")
    for required in (task1_checkpoint, teacher_dataset, baseline_selection):
        if not required.is_file():
            raise FileNotFoundError(required)
    baseline = json.loads(baseline_selection.read_text(encoding="utf-8"))
    work_directory.mkdir(parents=True)
    config = _write_config(
        base_config, work_directory / "transfer_config.json",
        teacher_dataset, 2.0)
    parent_task1 = baseline["parent"]["task1"]
    parent_task2 = baseline["parent"]["task2"]
    parent_score = _number(parent_task1, "mean_score")
    parent_crates = _number(parent_task2, "mean_crates")
    results = {
        "experiment_id": "T2-D02",
        "hypothesis": (
            "Penalizing only safe WAIT decisions that ignore objective progress "
            "reduces zero-bomb stuck rounds without sacrificing Task-1 retention."),
        "single_change": {
            "name": "task2_avoidable_wait_penalty", "baseline": 0.0,
            "candidate": -0.04, "version": "cnn-task2-progress-v1",
        },
        "contract": {
            "task1_checkpoint": str(task1_checkpoint),
            "task1_checkpoint_sha256": _sha256(task1_checkpoint),
            "teacher_dataset": str(teacher_dataset),
            "teacher_dataset_sha256": _sha256(teacher_dataset),
            "baseline_selection": str(baseline_selection),
            "baseline_best": baseline["transfer"]["11"]["best"],
            "training_seeds": list(TRAINING_SEEDS),
            "main_seeds": [MAIN_SEEDS[0], MAIN_SEEDS[-1]],
            "target_action_steps": 200_000,
            "minimum_rounds": 500,
        },
        "parent": {"task1": parent_task1, "task2": parent_task2},
        "transfer": {}, "main_validation": None,
    }
    run_id = f"{work_directory.name}_transfer_s11"
    _train(config, run_id, 11, task1_checkpoint)
    results["transfer"]["11"] = _evaluate_snapshots(
        "transfer_wait", 11, run_id, config, parent_score,
        parent_crates, work_directory)
    (work_directory / "selection.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if results["transfer"]["11"]["best"]["gate"]["passed"]:
        for seed in TRAINING_SEEDS[1:]:
            run_id = f"{work_directory.name}_transfer_s{seed}"
            _train(config, run_id, seed, task1_checkpoint)
            results["transfer"][str(seed)] = _evaluate_snapshots(
                "transfer_wait", seed, run_id, config, parent_score,
                parent_crates, work_directory)
            (work_directory / "selection.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8")
        selected = max(
            (entry["best"] for entry in results["transfer"].values()),
            key=_rank)
        best_path = work_directory / "best_task2.pt"
        shutil.copy2(selected["checkpoint"], best_path)
        main_task1 = _evaluate(
            best_path, f"{work_directory.name}_main_t1", config, 1,
            MAIN_SEEDS, reward_id="r7_safe_credit_sparse")
        main_task2 = _evaluate(
            best_path, f"{work_directory.name}_main_t2", config, 2,
            MAIN_SEEDS, reward_id="r7_safe_credit_sparse")
        results["selected"] = {
            **selected, "published_checkpoint": str(best_path),
            "published_sha256": _sha256(best_path),
        }
        results["main_validation"] = {
            "task1": main_task1, "task2": main_task2,
            "gate": _gate(
                main_task1, main_task2, parent_score, parent_crates),
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
    parser.add_argument("--baseline-selection", type=Path, default=(
        PROJECT_ROOT / "runs/cnn_distilled_t2_j471458/selection.json"))
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        PROJECT_ROOT / "experiments/configs/task2_cnn_wait_shaping.json"))
    args = parser.parse_args()
    print(json.dumps(pipeline(
        args.task1_checkpoint, args.teacher_dataset, args.work_directory,
        args.config, args.baseline_selection), indent=2))


if __name__ == "__main__":
    main()

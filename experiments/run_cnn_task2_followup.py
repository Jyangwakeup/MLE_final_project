"""Conditionally run Task-2 reward and n-step CNN follow-up experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .run_cnn_task2_pipeline import (
    DEVELOPMENT_SEEDS, MAIN_SEEDS, PROJECT_ROOT, TRAINING_SEEDS,
    _evaluate, _evaluate_snapshots, _gate, _number, _rank, _sha256, _train,
    _write_config,
)


def _read(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_gate(document, candidate):
    parent = document["parent"]
    return _gate(
        candidate["task1"], candidate["task2"],
        _number(parent["task1"], "mean_score"),
        _number(parent["task2"], "mean_crates"),
    )


def _current_best(document, entry):
    """Re-rank historical snapshots under the current strict contract."""
    candidates = entry.get("candidates") or [entry["best"]]
    reranked = []
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["gate"] = _strict_gate(document, candidate)
        reranked.append(candidate)
    return max(reranked, key=_rank)


def selection_qualified(document):
    transfer = document.get("transfer", {})
    if set(transfer) != {str(seed) for seed in TRAINING_SEEDS}:
        return False
    if not all(_current_best(document, entry)["gate"]["passed"]
               for entry in transfer.values()):
        return False
    main = document.get("main_validation")
    return bool(main and _strict_gate(document, main)["passed"])


def _write_result(work_directory, results):
    work_directory.mkdir(parents=True, exist_ok=True)
    (work_directory / "selection.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")


def candidate_pipeline(
        *, experiment_id, task1_checkpoint, teacher_dataset, work_directory,
        base_config, baseline_selection, predecessor_selections=()):
    task1_checkpoint = Path(task1_checkpoint).resolve()
    teacher_dataset = Path(teacher_dataset).resolve()
    work_directory = Path(work_directory).resolve()
    if work_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite pipeline directory: {work_directory}")
    baseline = _read(baseline_selection)
    predecessors = [(_read(path), str(Path(path).resolve()))
                    for path in predecessor_selections if Path(path).is_file()]
    for document, source in predecessors:
        if selection_qualified(document):
            results = {
                "experiment_id": experiment_id, "status": "skipped",
                "reason": "a predecessor passed the strict three-seed and main gate",
                "qualified_predecessor": source,
            }
            _write_result(work_directory, results)
            return results

    work_directory.mkdir(parents=True)
    config_document = json.loads(Path(base_config).read_text(encoding="utf-8"))
    reward_id = str(config_document["reward_id"])
    n_step = int(config_document["training"]["n_step"])
    config = _write_config(
        base_config, work_directory / "transfer_config.json",
        teacher_dataset, 2.0)
    parent_task1 = baseline["parent"]["task1"]
    parent_task2 = baseline["parent"]["task2"]
    parent_score = _number(parent_task1, "mean_score")
    parent_crates = _number(parent_task2, "mean_crates")
    results = {
        "experiment_id": experiment_id, "status": "running",
        "contract": {
            "reward_id": reward_id, "n_step": n_step,
            "task1_checkpoint": str(task1_checkpoint),
            "task1_checkpoint_sha256": _sha256(task1_checkpoint),
            "teacher_dataset": str(teacher_dataset),
            "teacher_dataset_sha256": _sha256(teacher_dataset),
            "baseline_selection": str(Path(baseline_selection).resolve()),
            "predecessor_selections": [source for _, source in predecessors],
            "training_seeds": list(TRAINING_SEEDS),
            "development_seeds": [DEVELOPMENT_SEEDS[0], DEVELOPMENT_SEEDS[-1]],
            "main_seeds": [MAIN_SEEDS[0], MAIN_SEEDS[-1]],
            "target_action_steps": 200_000, "minimum_rounds": 500,
        },
        "parent": {"task1": parent_task1, "task2": parent_task2},
        "transfer": {}, "main_validation": None,
    }
    _write_result(work_directory, results)

    for seed in TRAINING_SEEDS:
        run_id = f"{work_directory.name}_transfer_s{seed}"
        _train(
            config, run_id, seed, task1_checkpoint, reward_id=reward_id)
        results["transfer"][str(seed)] = _evaluate_snapshots(
            experiment_id.lower(), seed, run_id, config, parent_score,
            parent_crates, work_directory, reward_id=reward_id)
        _write_result(work_directory, results)
        if not results["transfer"][str(seed)]["best"]["gate"]["passed"]:
            results["status"] = "failed"
            results["failure"] = f"training seed {seed} failed the strict gate"
            _write_result(work_directory, results)
            return results

    selected = max(
        (entry["best"] for entry in results["transfer"].values()), key=_rank)
    best_path = work_directory / "best_task2.pt"
    shutil.copy2(selected["checkpoint"], best_path)
    main_task1 = _evaluate(
        best_path, f"{work_directory.name}_main_t1", config, 1,
        MAIN_SEEDS, reward_id=reward_id)
    main_task2 = _evaluate(
        best_path, f"{work_directory.name}_main_t2", config, 2,
        MAIN_SEEDS, reward_id=reward_id)
    main = {"task1": main_task1, "task2": main_task2}
    main["gate"] = _gate(
        main_task1, main_task2, parent_score, parent_crates)
    results["selected"] = {
        **selected, "published_checkpoint": str(best_path),
        "published_sha256": _sha256(best_path),
    }
    results["main_validation"] = main
    results["status"] = "completed" if main["gate"]["passed"] else "failed"
    _write_result(work_directory, results)
    return results


def _best_reward(selection_paths):
    candidates = []
    for path in selection_paths:
        if not Path(path).is_file():
            continue
        document = _read(path)
        if document.get("status") == "skipped":
            continue
        reward_id = document.get("contract", {}).get(
            "reward_id", "r7_safe_credit_sparse")
        entry = document.get("transfer", {}).get("11")
        if entry:
            candidate = _current_best(document, entry)
            candidates.append((candidate, reward_id))
    if not candidates:
        raise RuntimeError("no completed seed-11 candidate is available for n-step selection")
    return max(candidates, key=lambda item: _rank(item[0]))[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reward", "n5"), required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--task1-checkpoint", type=Path, default=(
        PROJECT_ROOT / "agent_code/cnn_distilled_double_dqn_agent/final.pt"))
    parser.add_argument("--teacher-dataset", type=Path, default=(
        PROJECT_ROOT / "runs/cnn_distilled_t1_j471456/teacher_task1.npz"))
    parser.add_argument("--baseline-selection", type=Path, default=(
        PROJECT_ROOT / "runs/cnn_distilled_t2_j471458/selection.json"))
    parser.add_argument("--d02-selection", type=Path, default=(
        PROJECT_ROOT / "runs/cnn_distilled_t2_wait_j472342/selection.json"))
    parser.add_argument("--d03-selection", type=Path, default=(
        PROJECT_ROOT / "runs/cnn_distilled_t2_potential_followup/selection.json"))
    args = parser.parse_args()

    if args.stage == "reward":
        result = candidate_pipeline(
            experiment_id="T2-D03", task1_checkpoint=args.task1_checkpoint,
            teacher_dataset=args.teacher_dataset,
            work_directory=args.work_directory,
            base_config=(PROJECT_ROOT / "experiments/configs/task2_cnn_reward_potential.json"),
            baseline_selection=args.baseline_selection,
            predecessor_selections=(args.d02_selection,),
        )
    else:
        predecessors = (args.d02_selection, args.d03_selection)
        documents = [_read(path) for path in predecessors if Path(path).is_file()]
        if any(selection_qualified(document) for document in documents):
            result = {
                "experiment_id": "T2-D04", "status": "skipped",
                "reason": "D02 or D03 passed; n=5 is not needed",
            }
            _write_result(args.work_directory.resolve(), result)
        else:
            reward_id = _best_reward(predecessors)
            config_name = (
                "task2_cnn_reward_potential_n5.json"
                if reward_id == "r7_safe_credit_potential"
                else "task2_cnn_wait_shaping_n5.json")
            result = candidate_pipeline(
                experiment_id="T2-D04", task1_checkpoint=args.task1_checkpoint,
                teacher_dataset=args.teacher_dataset,
                work_directory=args.work_directory,
                base_config=PROJECT_ROOT / "experiments/configs" / config_name,
                baseline_selection=args.baseline_selection,
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

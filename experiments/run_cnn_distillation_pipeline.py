"""Execute the preregistered Task-1 CNN teacher-distillation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.collect_cnn_teacher_data import collect
from experiments.pretrain_cnn_distilled import pretrain


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs"
ARCHITECTURES = ("global", "action_aligned", "action_aligned_d4")


def _run(arguments):
    subprocess.run(
        [sys.executable, "-u", "-m", "experiments.run", *arguments],
        cwd=PROJECT_ROOT, check=True)


def _average(run_id):
    paths = list((RUNS / run_id).glob("*_summary/summary.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one aggregate summary for {run_id}; got {paths}")
    with paths[0].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    matches = [row for row in rows if row["run_id"] == "AVERAGE"]
    if len(matches) != 1:
        raise RuntimeError(f"aggregate summary for {run_id} has no AVERAGE row")
    return matches[0]


def _evaluate(checkpoint, run_id, config, seeds, rounds):
    _run([
        "--config", str(config), "--mode", "evaluate", "--task", "1",
        "--agent", "cnn_distilled_double_dqn_agent",
        "--reward-id", "r5_conditional_loop", "--checkpoint", str(checkpoint),
        "--device", "cpu", "--n-rounds", str(rounds),
        "--seeds", *(str(seed) for seed in seeds), "--run-id", run_id,
        "--replay-policy", "none",
    ])
    return _average(run_id)


def _rank(metrics):
    completion = metrics.get("mean_all_coins_completion_steps")
    return (
        float(metrics["all_coins_rate"]), float(metrics["mean_coins"]),
        -float(completion) if completion else float("-inf"),
    )


def pipeline(teacher_checkpoint, work_directory, base_config):
    work_directory = Path(work_directory).resolve()
    if work_directory.exists():
        raise FileExistsError(f"refusing to overwrite pipeline directory: {work_directory}")
    work_directory.mkdir(parents=True)
    dataset = collect(
        Path(teacher_checkpoint), work_directory / "teacher_task1.npz",
        PROJECT_ROOT / "experiments/configs/task1_cnn_teacher_capture.json")
    config_data = json.loads(Path(base_config).read_text(encoding="utf-8"))
    config_data["cnn_distillation"]["dataset"] = str(dataset)
    config = work_directory / "config.json"
    config.write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")

    screening = []
    for architecture in ARCHITECTURES:
        checkpoint = pretrain(
            dataset, work_directory / f"pretrained_{architecture}.pt", architecture)
        run_id = f"{work_directory.name}_screen_{architecture}"
        metrics = _evaluate(checkpoint, run_id, config, range(12000, 12020), 1)
        screening.append({"architecture": architecture,
                          "checkpoint": str(checkpoint), "metrics": metrics})
    selected = max(screening, key=lambda item: _rank(item["metrics"]))
    pretrained = work_directory / "pretrained.pt"
    shutil.copy2(selected["checkpoint"], pretrained)
    (work_directory / "pretrained_selection.json").write_text(
        json.dumps({"candidates": screening, "selected": selected}, indent=2) + "\n",
        encoding="utf-8")

    fine_run_id = f"{work_directory.name}_finetune_s11"
    _run([
        "--config", str(config), "--mode", "train", "--task", "1",
        "--agent", "cnn_distilled_double_dqn_agent",
        "--reward-id", "r5_conditional_loop", "--init-from-checkpoint", str(pretrained),
        "--device", "cuda", "--seed", "11", "--n-rounds", "2000",
        "--target-stage-action-steps", "50000", "--min-rounds", "1",
        "--run-id", fine_run_id, "--replay-policy", "sampled",
    ])
    fine = RUNS / fine_run_id / "checkpoints"
    candidates = [
        ("pretrained", pretrained),
        ("10000", fine / "snapshots/policy_010000.pt"),
        ("25000", fine / "snapshots/policy_025000.pt"),
        ("50000", fine / "snapshots/policy_050000.pt"),
    ]
    development = []
    for label, checkpoint in candidates:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing planned candidate checkpoint: {checkpoint}")
        run_id = f"{work_directory.name}_dev_{label}"
        metrics = _evaluate(checkpoint, run_id, config, range(10001, 10006), 20)
        development.append({"label": label, "checkpoint": str(checkpoint),
                            "metrics": metrics})
    best = max(development, key=lambda item: _rank(item["metrics"]))
    best_path = work_directory / "best_task1.pt"
    shutil.copy2(best["checkpoint"], best_path)
    result = {"screening": screening, "development": development, "best": best,
              "reserved": None, "acceptance_met": False}
    if float(best["metrics"]["all_coins_rate"]) >= 0.90:
        reserved = _evaluate(
            best_path, f"{work_directory.name}_reserved", config,
            range(21000, 21100), 1)
        result["reserved"] = reserved
        result["acceptance_met"] = (
            float(reserved["all_coins_rate"]) >= 0.90
            and float(best["metrics"]["act_max_time"]) < 0.5
            and float(reserved["act_max_time"]) < 0.5
        )
    (work_directory / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=(
        PROJECT_ROOT / "agent_code/double_dqn_continuous_v2_agent/final.pt"))
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        PROJECT_ROOT / "experiments/configs/task1_cnn_distillation.json"))
    args = parser.parse_args()
    print(json.dumps(pipeline(
        args.teacher_checkpoint, args.work_directory, args.config), indent=2))


if __name__ == "__main__":
    main()

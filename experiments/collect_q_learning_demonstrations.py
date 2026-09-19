"""Collect frozen Task 2 teacher trajectories and encode the student dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.q_learning_demo_data import (
    CAPTURE_ENV, encode_raw_dataset, sha256, write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect(*, teacher: Path, student: Path, work_directory: Path,
            config: Path, seeds=range(6000, 6100)) -> dict:
    teacher = teacher.resolve()
    student = student.resolve()
    work_directory = work_directory.resolve()
    if work_directory.exists():
        raise FileExistsError(f"refusing to overwrite work directory: {work_directory}")
    if not teacher.is_file() or not student.is_file():
        raise FileNotFoundError("teacher and initial student checkpoints must exist")
    work_directory.mkdir(parents=True)
    raw = work_directory / "teacher_transitions.pklstream"
    dataset = work_directory / "student_transitions.npz"
    environment = os.environ.copy()
    environment[CAPTURE_ENV] = str(raw)
    run_id = work_directory.name + "_capture"
    command = [
        sys.executable, "-m", "experiments.run",
        "--config", str(config), "--mode", "evaluate", "--task", "2",
        "--agent", "double_dqn_continuous_v2_agent",
        "--checkpoint", str(teacher), "--device", "cpu",
        "--run-id", run_id, "--replay-policy", "none",
        "--seeds", *(str(seed) for seed in seeds), "--n-rounds", "1",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    encoded = encode_raw_dataset(raw, dataset)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=True).stdout.strip()
    manifest = {
        "schema_version": "q-learning-demonstration-manifest-v1",
        "source_commit": commit,
        "teacher": {"path": str(teacher), "sha256": sha256(teacher)},
        "initial_student": {"path": str(student), "sha256": sha256(student)},
        "config": {"path": str(config.resolve()), "sha256": sha256(config)},
        "raw": {"path": str(raw), "sha256": encoded["raw_sha256"]},
        "dataset": {
            "path": str(dataset), "sha256": encoded["dataset_sha256"],
            "transitions": encoded["transitions"],
            "terminal_transitions": encoded["terminal_transitions"],
            "inadmissible_teacher_actions": encoded[
                "inadmissible_teacher_actions"],
            "environment_seeds": encoded["environment_seeds"],
            "feature_id": "continuous-v2",
            "reward_id": "r20_safe_credit_targeted_wait",
        },
    }
    write_manifest(work_directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        PROJECT_ROOT / "experiments/configs/optimized_double_q_lambda_demo_capture.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6000, 6100)))
    args = parser.parse_args()
    result = collect(teacher=args.teacher, student=args.student,
                     work_directory=args.work_directory, config=args.config,
                     seeds=args.seeds)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

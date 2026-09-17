"""Collect deterministic frozen Task 1/2 facts for phase-policy distillation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(6000, 6100))


def _checkpoint(parent: Path) -> Path:
    metadata = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    path = Path(metadata["checkpoint"])
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def collect(parent: Path, output: Path, config: Path) -> Path:
    parent = parent.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "teacher.jsonl"
    checkpoint = _checkpoint(parent)
    environment = os.environ.copy()
    environment["BOMBERMAN_DISTILLATION_CAPTURE"] = str(raw)
    for task in (1, 2):
        command = [
            sys.executable, str(PROJECT_ROOT / "experiments" / "run.py"),
            "--config", str(config), "--mode", "evaluate", "--task", str(task),
            "--agent", "double_dqn_continuous_v2_agent",
            "--feature-id", "continuous-v2", "--reward-id", "r7_safe_credit_sparse",
            "--checkpoint", str(checkpoint), "--device", "cpu", "--n-rounds", "1",
            "--seeds", *(str(seed) for seed in SEEDS),
            "--run-id", f"{output.name}_capture_t{task}", "--replay-policy", "none",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    rng = np.random.default_rng(20260913)
    selected = []
    for task in (1, 2):
        task_rows = [row for row in rows if row["task_id"] == task]
        if not task_rows:
            raise RuntimeError(f"No Task {task} teacher states were captured")
        indices = np.arange(len(task_rows))
        if len(indices) > 10_000:
            indices = np.sort(rng.choice(indices, size=10_000, replace=False))
        selected.extend(task_rows[int(index)] for index in indices)
    destination = output / "teacher.npz"
    np.savez_compressed(
        destination,
        states=np.asarray([row["state"] for row in selected], dtype=np.float32),
        legal_masks=np.asarray([row["legal_mask"] for row in selected], dtype=bool),
        teacher_q=np.asarray([row["teacher_q"] for row in selected], dtype=np.float32),
        environment_seeds=np.asarray(
            [row["environment_seed"] for row in selected], dtype=np.int64),
        task_ids=np.asarray([row["task_id"] for row in selected], dtype=np.int8),
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "task3_phase_r7.json")
    args = parser.parse_args()
    print(collect(args.parent_run, args.output, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

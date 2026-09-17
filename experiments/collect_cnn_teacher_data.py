"""Collect Task-1 student boards and frozen continuous-v2 teacher Q values."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect(checkpoint: Path, output: Path, config: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {checkpoint}")
    environment = os.environ.copy()
    environment["BOMBERMAN_CNN_TEACHER_CAPTURE"] = str(output)
    command = [
        sys.executable, str(PROJECT_ROOT / "experiments" / "run.py"),
        "--config", str(config), "--mode", "evaluate", "--task", "1",
        "--agent", "cnn_distillation_teacher_agent",
        "--feature-id", "continuous-v2", "--reward-id", "r7_safe_credit_sparse",
        "--checkpoint", str(checkpoint), "--device", "cpu", "--n-rounds", "1",
        "--seeds", *(str(seed) for seed in range(6000, 6100)),
        "--run-id", output.parent.name + "_" + output.stem + "_capture",
        "--replay-policy", "none",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    if not output.is_file():
        raise RuntimeError("teacher capture produced no dataset")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        PROJECT_ROOT / "experiments" / "configs" / "task1_cnn_teacher_capture.json"))
    args = parser.parse_args()
    print(collect(args.checkpoint, args.output, args.config))


if __name__ == "__main__":
    main()

"""Training-mode orchestration and early stopping for experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


DEFAULT_REPLAY_PROGRESS_PERCENT = 10


class TrainingEarlyStopping:
    """Stop training when the rolling mean reward has stopped improving."""

    def __init__(self, training_path: Path, config: dict[str, Any]):
        self.training_path = training_path
        self.window = _positive_int(config.get("window", 100), "window")
        self.patience = _positive_int(config.get("patience", 100), "patience")
        self.min_rounds = _positive_int(config.get("min_rounds", 300), "min_rounds")
        self.min_delta = config.get("min_delta", 0.1)
        if (isinstance(self.min_delta, bool)
                or not isinstance(self.min_delta, (int, float))
                or not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError("early_stopping.min_delta must be finite and non-negative")
        raw_target = config.get("target_reward")
        if raw_target is not None and (
            isinstance(raw_target, bool)
            or not isinstance(raw_target, (int, float))
            or not math.isfinite(raw_target)
        ):
            raise ValueError("early_stopping.target_reward must be null or finite")
        self.target_reward = None if raw_target is None else float(raw_target)
        self.rewards: list[float] = []
        self._reward_column: int | None = None
        self._file_offset = 0
        self.best_mean: float | None = None
        self.rounds_without_improvement = 0
        self.result: dict[str, Any] | None = None

    def __call__(self, completed_rounds: int) -> bool:
        with self.training_path.open(newline="", encoding="utf-8") as file:
            if self._reward_column is None:
                header = next(csv.reader([file.readline()]))
                try:
                    self._reward_column = header.index("reward")
                except ValueError as exception:
                    raise ValueError("training.csv is missing the reward column") from exception
                self._file_offset = file.tell()
            else:
                file.seek(self._file_offset)
            new_lines = file.readlines()
            self._file_offset = file.tell()
        for row in csv.reader(new_lines):
            self.rewards.append(float(row[self._reward_column]))
        if len(self.rewards) < self.window:
            return False

        rolling_mean = float(np.mean(self.rewards[-self.window:]))
        if self.best_mean is None or rolling_mean > self.best_mean + self.min_delta:
            self.best_mean = rolling_mean
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        should_stop = (
            completed_rounds >= self.min_rounds
            and self.rounds_without_improvement >= self.patience
            and (self.target_reward is None or rolling_mean >= self.target_reward)
        )
        if should_stop:
            self.result = {
                "best_rolling_mean_reward": self.best_mean,
                "completed_rounds": completed_rounds,
                "reason": "rolling_mean_plateau",
                "rolling_mean_reward": rolling_mean,
                "rounds_without_improvement": self.rounds_without_improvement,
            }
        return should_stop


def early_stopping_config(training: dict[str, Any]) -> dict[str, Any] | None:
    config = training.get("early_stopping", {})
    if not isinstance(config, dict):
        raise ValueError("config.training.early_stopping must be an object")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("config.training.early_stopping.enabled must be a boolean")
    return config if enabled else None


def run_training_mode(
    args: Any,
    config: dict[str, Any],
    task_name: str,
    scenario: str,
    opponents: Sequence[str],
    *,
    output_directory: Callable,
    checkpoint_name: Callable[[str], str],
    run_session: Callable,
) -> Path:
    """Validate and execute the training-specific CLI branch."""
    if args.seeds is not None:
        raise ValueError("Training accepts one --seed, not --seeds")
    if args.checkpoint is not None:
        raise ValueError("Training writes its own checkpoint; do not pass --checkpoint")
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("config.training must be an object")
    n_rounds = args.n_rounds or training.get("n_rounds")
    if n_rounds is None:
        raise ValueError("Training requires --n-rounds or config.training.n_rounds")
    configured_seed = config.get("seed")
    seed = args.seed if args.seed is not None else (
        configured_seed if isinstance(configured_seed, int) else 11
    )
    output, _ = output_directory(args.run_id, args.output)
    checkpoint = output / "checkpoints" / checkpoint_name(args.agent)
    replay_interval = args.replay_interval or replay_progress_interval(n_rounds)
    return run_session(
        args.config, "train", seed, output, args.agent, opponents, scenario,
        n_rounds, checkpoint, task_name,
        "sampled" if args.replay_policy == "auto" else args.replay_policy,
        replay_interval, early_stopping_config(training),
    )


def replay_progress_interval(
    n_rounds: int,
    progress_percent: int = DEFAULT_REPLAY_PROGRESS_PERCENT,
) -> int:
    """Return the round interval for fixed percentage progress milestones."""
    if isinstance(n_rounds, bool) or not isinstance(n_rounds, int) or n_rounds < 1:
        raise ValueError("n_rounds must be a positive integer")
    if (isinstance(progress_percent, bool)
            or not isinstance(progress_percent, int)
            or not 1 <= progress_percent <= 100):
        raise ValueError("progress_percent must be an integer from 1 to 100")
    return max(1, math.ceil(n_rounds * progress_percent / 100))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"config.training.early_stopping.{name} must be a positive integer")
    return value

"""Training-mode orchestration and early stopping for experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agent_code.team_agent.feature_system import ACTIONS, normalize_feature_id
from agent_code.team_agent.exploration import resolve_exploration_spec
from agent_code.team_agent.rewards import REWARD_VERSION, resolve_reward_spec
from agent_code.team_agent.safety import resolve_safety_spec
from experiments.devices import resolve_device
from experiments.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    load_migration_snapshot,
    load_task3_safety_transfer_snapshot,
    load_training_snapshot,
    validate_task3_safety_transfer,
    load_task3_transfer_snapshot,
    validate_task3_transfer,
    validate_v6_migration,
    validate_resume_transition,
)
from experiments.performance_stopping import resolve_performance_stopping
from experiments.agent_contracts import resolve_agent_contract
from agent_code.learning_common.training_spec import (
    resolve_retention_spec, resolve_safety_replay_spec,
)


def allowed_n_steps(algorithm: str) -> set[int]:
    """Return replay horizons supported by a learning algorithm.

    The distilled CNN and Task 3 Double DQN support the registered five-step
    credit-assignment experiments. Other learners retain the 1/4-step contract.
    """
    return {1, 4, 5} if algorithm in {"cnn_distilled_double_dqn", "double_dqn"} else {1, 4}


DEFAULT_REPLAY_PROGRESS_PERCENT = 10
DEFAULT_EARLY_STOPPING_CONFIG = {
    "enabled": True,
    "window": 200,
    "patience": 100,
    "min_rounds": 300,
    "min_delta": 0.1,
    "target_reward": None,
    "require_exploration_complete": True,
}


class TrainingActionBudget:
    """Stop after both a local round minimum and a stage action target."""

    def __init__(self, training_path: Path, target_stage_action_steps: int,
                 min_rounds: int):
        self.training_path = training_path
        self.target = _positive_int(
            target_stage_action_steps, "target_stage_action_steps")
        self.min_rounds = _positive_int(min_rounds, "min_rounds")
        self.result: dict[str, Any] | None = None

    def __call__(self, completed_rounds: int) -> bool:
        with self.training_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            return False
        stage_steps = int(rows[-1]["stage_action_steps"])
        reached = completed_rounds >= self.min_rounds and stage_steps >= self.target
        if reached:
            self.result = {
                "reason": "stage_action_target_reached",
                "completed_rounds": completed_rounds,
                "stage_action_steps": stage_steps,
                "target_stage_action_steps": self.target,
                "min_rounds": self.min_rounds,
            }
        return reached


class CompositeTrainingStop:
    def __init__(self, *conditions):
        self.conditions = tuple(condition for condition in conditions if condition is not None)

    def __call__(self, completed_rounds: int) -> bool:
        return any(condition(completed_rounds) for condition in self.conditions)


class TrainingEarlyStopping:
    """Stop training when the rolling mean reward has stopped improving."""

    def __init__(
        self, training_path: Path, config: dict[str, Any],
        initial_rewards: Sequence[float] = (),
    ):
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
        require_exploration = config.get("require_exploration_complete", True)
        if not isinstance(require_exploration, bool):
            raise ValueError(
                "early_stopping.require_exploration_complete must be a boolean")
        self.min_action_steps = (
            _positive_int(
                config.get(
                    "min_action_steps",
                    resolve_exploration_spec()["decay_action_steps"],
                ),
                "min_action_steps",
            )
            if require_exploration else 0
        )
        self.latest_action_steps = 0
        self.rewards: list[float] = []
        self._reward_column: int | None = None
        self._file_offset = 0
        self.best_mean: float | None = None
        self.rounds_without_improvement = 0
        self.result: dict[str, Any] | None = None
        for reward in initial_rewards:
            self.rewards.append(float(reward))
            self._update_progress()

    def __call__(self, completed_rounds: int) -> bool:
        with self.training_path.open(newline="", encoding="utf-8") as file:
            if self._reward_column is None:
                header = next(csv.reader([file.readline()]))
                try:
                    self._reward_column = header.index("reward")
                except ValueError as exception:
                    raise ValueError("training.csv is missing the reward column") from exception
                try:
                    self._action_steps_column = header.index("action_steps")
                except ValueError as exception:
                    raise ValueError(
                        "training.csv is missing the action_steps column") from exception
                self._file_offset = file.tell()
            else:
                file.seek(self._file_offset)
            new_lines = file.readlines()
            self._file_offset = file.tell()
        for row in csv.reader(new_lines):
            self.rewards.append(float(row[self._reward_column]))
            self.latest_action_steps = int(row[self._action_steps_column])
        if len(self.rewards) < self.window:
            return False

        return self._update_progress(completed_rounds)

    def _update_progress(self, completed_rounds: int | None = None) -> bool:
        if len(self.rewards) < self.window:
            return False
        rolling_mean = float(np.mean(self.rewards[-self.window:]))
        if self.best_mean is None or rolling_mean > self.best_mean + self.min_delta:
            self.best_mean = rolling_mean
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        effective_rounds = len(self.rewards)
        should_stop = completed_rounds is not None and (
            effective_rounds >= self.min_rounds
            and self.latest_action_steps >= self.min_action_steps
            and self.rounds_without_improvement >= self.patience
            and (self.target_reward is None or rolling_mean >= self.target_reward)
        )
        if should_stop:
            self.result = {
                "best_rolling_mean_reward": self.best_mean,
                "completed_rounds": effective_rounds,
                "action_steps": self.latest_action_steps,
                "min_action_steps": self.min_action_steps,
                "reason": "rolling_mean_plateau",
                "rolling_mean_reward": rolling_mean,
                "rounds_without_improvement": self.rounds_without_improvement,
            }
        return should_stop


def early_stopping_config(training: dict[str, Any]) -> dict[str, Any] | None:
    config = training.get("early_stopping", DEFAULT_EARLY_STOPPING_CONFIG)
    if not isinstance(config, dict):
        raise ValueError("config.training.early_stopping must be an object")
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("config.training.early_stopping.enabled must be a boolean")
    if not enabled:
        return None
    return {**DEFAULT_EARLY_STOPPING_CONFIG, **config, "enabled": True}


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
    source_commit: str | None,
    source_hash: str,
    source_hash_scope: str | None = None,
) -> Path:
    """Validate and execute the training-specific CLI branch."""
    if args.seeds is not None:
        raise ValueError("Training accepts one --seed, not --seeds")
    if args.checkpoint is not None:
        raise ValueError("Training writes its own checkpoint; do not pass --checkpoint")
    transfer_from = getattr(args, "transfer_task3_from", None)
    resume_sources = [
        value for value in (
            args.resume_from, getattr(args, "migrate_resume_from", None), transfer_from,
            getattr(args, "transfer_task3_safety_from", None),
            getattr(args, "transfer_task4_from_checkpoint", None),
            getattr(args, "transfer_task4_exploration_from_checkpoint", None),
            getattr(args, "transfer_task4_frozen_opponents_from_checkpoint", None),
            getattr(args, "transfer_task4_score_from_checkpoint", None))
        if value is not None
    ]
    if len(resume_sources) > 1:
        raise ValueError("resume, migration, and Task 3 transfer are mutually exclusive")
    if resume_sources and getattr(args, "init_from_checkpoint", None) is not None:
        raise ValueError("resume/transfer and --init-from-checkpoint are mutually exclusive")
    if config.get('task4_contract') and args.resume_from is None and getattr(args, 'transfer_task4_from_checkpoint', None) is None:
        raise ValueError('Registered Task 4 requires explicit transfer or exact resume')
    if config.get('exploration_contract') and args.resume_from is None and getattr(args, 'transfer_task4_exploration_from_checkpoint', None) is None:
        raise ValueError('Exploration requires explicit transfer or strict resume')
    if config.get('frozen_opponents') and not config.get('score_contract') and args.resume_from is None and getattr(args, 'transfer_task4_frozen_opponents_from_checkpoint', None) is None:
        raise ValueError('Frozen-opponent training requires explicit transfer or strict resume')
    if config.get("score_contract") and args.resume_from is None and getattr(args, "transfer_task4_score_from_checkpoint", None) is None:
        raise ValueError("Score experiment requires explicit transfer or strict resume")
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
    stopping_config = early_stopping_config(training)
    performance_stopping = resolve_performance_stopping(
        training.get("performance_stopping"), task=task_name)
    checkpoint_snapshot_interval = training.get(
        "checkpoint_snapshot_interval_action_steps")
    if checkpoint_snapshot_interval is not None:
        checkpoint_snapshot_interval = _positive_int(
            checkpoint_snapshot_interval,
            "checkpoint_snapshot_interval_action_steps")
    target_steps = (
        getattr(args, "target_stage_action_steps", None)
        if getattr(args, "target_stage_action_steps", None) is not None
        else training.get("target_stage_action_steps")
    )
    min_rounds = getattr(args, "min_rounds", None) or training.get("min_rounds", 1)
    budget_config = {
        "target_stage_action_steps": (
            None if target_steps is None
            else _positive_int(target_steps, "target_stage_action_steps")
        ),
        "min_rounds": _positive_int(min_rounds, "min_rounds"),
    }
    if budget_config["min_rounds"] > int(n_rounds):
        raise ValueError("--min-rounds cannot exceed --n-rounds")
    if performance_stopping is not None:
        if budget_config["target_stage_action_steps"] is not None:
            raise ValueError(
                "Task 1 performance_stopping conflicts with an action-step target")
        if stopping_config is not None:
            raise ValueError(
                "Task 1 performance_stopping conflicts with reward early stopping")
    safe_exploration = training.get("safe_exploration", False)
    if not isinstance(safe_exploration, bool):
        raise ValueError("config.training.safe_exploration must be a boolean")
    configured_safety = config.get("safety")
    if configured_safety is not None and "safe_exploration" in training:
        raise ValueError("config.safety conflicts with training.safe_exploration")
    safety_spec = resolve_safety_spec(
        configured_safety,
        legacy_safe_exploration=(
            safe_exploration if configured_safety is None else None),
    )
    effective_safe_exploration = safety_spec["mode"] in {"exploration", "all"}
    safety_replay_spec = resolve_safety_replay_spec(training.get("safety_replay"))
    configured_id = getattr(args, "feature_id", None) or config.get("feature_id")
    configured_legacy = (
        None if getattr(args, "feature_id", None) is not None
        else config.get("feature_version"))
    requested_feature_id = (
        None if configured_id is None and configured_legacy is None
        else normalize_feature_id(configured_id, configured_legacy)
    )
    agent_contract = resolve_agent_contract(args.agent, requested_feature_id, config)
    algorithm = agent_contract.algorithm
    n_step = training.get("n_step", 1)
    supported_n_steps = allowed_n_steps(algorithm)
    if n_step not in supported_n_steps:
        values = ", ".join(str(value) for value in sorted(supported_n_steps))
        raise ValueError(
            f"config.training.n_step must be one of {values} for {algorithm}")
    retention_spec = resolve_retention_spec(training.get("retention"))
    adaptation_triggers = tuple(getattr(args, "adaptation_trigger", ()) or ())
    init_checkpoint = getattr(args, "init_from_checkpoint", None)
    if init_checkpoint is not None:
        if algorithm not in {
            "dqn", "double_dqn", "cnn_distilled_double_dqn", "rainbow_lite",
            "expected_sarsa_lambda",
        }:
            raise ValueError(
                "--init-from-checkpoint only supports neural agents and "
                "expected_sarsa_lambda")
        init_checkpoint = Path(init_checkpoint).expanduser().resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(
                f"Warm-start checkpoint does not exist: {init_checkpoint}")
    requested_device = args.device or training.get("device", "auto")
    device_info = resolve_device(algorithm, "train", requested_device)
    positional = (
        args.config, "train", seed, output, args.agent, opponents, scenario,
        n_rounds, checkpoint, task_name,
        "sampled" if args.replay_policy == "auto" else args.replay_policy,
        replay_interval, stopping_config,
    )
    migration_from = getattr(args, "migrate_resume_from", None)
    safety_transfer_from = getattr(args, "transfer_task3_safety_from", None)
    if args.resume_from is None and migration_from is None and transfer_from is None and safety_transfer_from is None:
        if getattr(args, "distillation_dataset", None) is not None:
            raise ValueError("--distillation-dataset requires --transfer-task3-from")
        return run_session(
            *positional, init_checkpoint=init_checkpoint, device_info=device_info,
            task4_transfer=getattr(args, "transfer_task4_from_checkpoint", None),
            task4_exploration_transfer=getattr(args, "transfer_task4_exploration_from_checkpoint", None),
            task4_score_transfer=getattr(args, "transfer_task4_score_from_checkpoint", None),
            task4_frozen_transfer=getattr(args, "transfer_task4_frozen_opponents_from_checkpoint", None),
            action_budget_config=budget_config,
            safe_exploration=safe_exploration,
            safety_spec=safety_spec,
            n_step=n_step,
            retention_spec=retention_spec,
            safety_replay_spec=safety_replay_spec,
            adaptation_triggers=adaptation_triggers,
            feature_id_override=getattr(args, "feature_id", None),
            reward_id_override=getattr(args, "reward_id", None),
            performance_stopping=performance_stopping,
            checkpoint_snapshot_interval=checkpoint_snapshot_interval,
        )

    migrating = migration_from is not None
    safety_transferring = safety_transfer_from is not None
    transferring = transfer_from is not None
    parent_run = Path(
        safety_transfer_from if safety_transferring else transfer_from if transferring else migration_from if migrating else args.resume_from)
    if not parent_run.is_absolute():
        parent_run = Path.cwd() / parent_run
    parent_run = parent_run.resolve()
    snapshot = (
        load_task3_safety_transfer_snapshot(parent_run) if safety_transferring
        else load_task3_transfer_snapshot(parent_run) if transferring
        else load_migration_snapshot(parent_run) if migrating
        else load_training_snapshot(parent_run)
    )
    if task_name == 'full_match' and args.agent == 'double_dqn_continuous_v2_agent':
        if config.get("score_contract"):
            from experiments.task4_score_transfer import validate_resume
        elif config.get("frozen_opponents"):
            from experiments.task4_frozen_transfer import validate_resume
        elif config.get("exploration_contract"):
            from experiments.task4_exploration_transfer import validate_resume
        else:
            from experiments.task4_transfer import validate_resume
        validate_resume(snapshot.contract.get('transfer_contract'), root=Path(__file__).resolve().parents[1],
                        config_path=args.config, seed=seed)
    metadata_path = parent_run / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Parent run is missing metadata.json")
    import json
    parent_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reward_id = getattr(args, "reward_id", None) or config.get(
        "reward_id", config.get("reward_version", REWARD_VERSION))
    child_contract = {
        "algorithm": algorithm,
        "seed": seed,
        "task": task_name,
        "feature_id": agent_contract.feature_id,
        "feature_schema": agent_contract.feature_schema,
        "feature_version": (
            "v1" if agent_contract.feature_id == "discrete-v1" else None),
        "reward_id": reward_id,
        "reward_version": reward_id,
        "reward_spec": resolve_reward_spec(reward_id),
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "actions": list(ACTIONS),
        "training_device_name": device_info["name"],
        "training_device_type": device_info["type"],
        "agent_seed": seed,
        "exploration_spec": resolve_exploration_spec(training.get("exploration")),
        "safe_exploration": effective_safe_exploration,
        "safety_spec": safety_spec,
        "n_step": n_step,
        "retention_spec": retention_spec,
        "safety_replay_spec": safety_replay_spec,
        "training_budget": budget_config,
        "performance_stopping": performance_stopping,
        "source_commit": source_commit,
        "source_hash": source_hash,
        "source_hash_scope": source_hash_scope,
        "network_spec": agent_contract.network_spec,
        "hyperparameters": agent_contract.hyperparameters,
        "transfer_contract": (
            snapshot.contract.get("transfer_contract")
            if not transferring else None
        ),
    }
    parent_status = parent_metadata.get("status", "unknown")
    if safety_transferring:
        validate_task3_safety_transfer(
            snapshot.contract, child_contract, parent_status=parent_status)
        resume_kind = "task3_safety_transfer"
    elif transferring:
        validate_task3_transfer(
            snapshot.contract, child_contract, parent_status=parent_status)
        resume_kind = "task3_transfer"
    elif migrating:
        validate_v6_migration(
            snapshot.contract, child_contract, parent_status=parent_status)
        resume_kind = "v6_migration"
    else:
        resume_kind = validate_resume_transition(
            snapshot.contract, child_contract, parent_status=parent_status,
        )
    if (
        resume_kind == "same_task"
        and snapshot.runner_state.get("early_stopping_config") != stopping_config
    ):
        raise ValueError("Same-Task resume requires the same early-stopping config")
    distillation_path = getattr(args, "distillation_dataset", None)
    if transferring and distillation_path is None:
        distillation_path = (
            parent_run.parent / f"task3_distillation_{parent_run.name}" / "teacher.npz")
    return run_session(
        *positional,
        resume_snapshot=snapshot,
        resume_kind=resume_kind,
        parent_metadata=parent_metadata,
        device_info=device_info,
        action_budget_config=budget_config,
        safe_exploration=safe_exploration,
        safety_spec=safety_spec,
        n_step=n_step,
        retention_spec=retention_spec,
        safety_replay_spec=safety_replay_spec,
        adaptation_triggers=adaptation_triggers,
        feature_id_override=getattr(args, "feature_id", None),
        reward_id_override=getattr(args, "reward_id", None),
        performance_stopping=performance_stopping,
        checkpoint_snapshot_interval=checkpoint_snapshot_interval,
        migration=migrating,
        task3_safety_transfer=safety_transferring,
        task3_transfer=transferring,
        distillation_path=distillation_path,
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
        raise ValueError(f"config.training.{name} must be a positive integer")
    return value

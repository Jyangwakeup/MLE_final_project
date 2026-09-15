"""Versioned task-local training controls supplied by the experiment runner."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping


N_STEP_ENV = "BOMBERMAN_N_STEP"
RETENTION_SPEC_ENV = "BOMBERMAN_RETENTION_SPEC"
TRAINING_BUDGET_ENV = "BOMBERMAN_TRAINING_BUDGET"
SAFETY_REPLAY_SPEC_ENV = "BOMBERMAN_SAFETY_REPLAY_SPEC"
DEFAULT_RETENTION_SPEC = {
    "parent_fraction": 0.5,
    "distillation_weight": 1.0,
    "temperature": 1.0,
    "per_task_capacity": 20_000,
    "current_warmup": 2_000,
}
DEFAULT_SAFETY_REPLAY_SPEC = {
    "enabled": False,
    "parent_samples": 48,
    "ordinary_task3_samples": 8,
    "own_bomb_cycle_samples": 8,
    "fatal_prefix_steps": 4,
}


def n_step_from_environment(default: int = 1) -> int:
    raw = os.getenv(N_STEP_ENV)
    value = default if raw is None else int(raw)
    if value not in {1, 4}:
        raise ValueError("training.n_step must be 1 or 4")
    return value


def resolve_retention_spec(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = DEFAULT_RETENTION_SPEC if value is None else value
    if not isinstance(raw, Mapping) or set(raw) != set(DEFAULT_RETENTION_SPEC):
        raise ValueError(
            "training.retention must contain exactly: "
            + ", ".join(sorted(DEFAULT_RETENTION_SPEC)))
    parent_fraction = _finite_float(raw["parent_fraction"], "parent_fraction")
    weight = _finite_float(raw["distillation_weight"], "distillation_weight")
    temperature = _finite_float(raw["temperature"], "temperature")
    capacity = _positive_int(raw["per_task_capacity"], "per_task_capacity")
    warmup = _positive_int(raw["current_warmup"], "current_warmup")
    if not 0.0 <= parent_fraction < 1.0:
        raise ValueError("training.retention.parent_fraction must be in [0,1)")
    if weight < 0.0:
        raise ValueError("training.retention.distillation_weight must be non-negative")
    if temperature <= 0.0:
        raise ValueError("training.retention.temperature must be positive")
    return {
        "parent_fraction": parent_fraction,
        "distillation_weight": weight,
        "temperature": temperature,
        "per_task_capacity": capacity,
        "current_warmup": warmup,
    }


def retention_from_environment() -> dict[str, Any]:
    raw = os.getenv(RETENTION_SPEC_ENV)
    if raw is None:
        return resolve_retention_spec()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{RETENTION_SPEC_ENV} must be valid JSON") from exception
    return resolve_retention_spec(value)


def resolve_safety_replay_spec(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = DEFAULT_SAFETY_REPLAY_SPEC if value is None else value
    if not isinstance(raw, Mapping) or set(raw) != set(DEFAULT_SAFETY_REPLAY_SPEC):
        raise ValueError(
            "training.safety_replay must contain exactly: "
            + ", ".join(sorted(DEFAULT_SAFETY_REPLAY_SPEC)))
    result = {"enabled": raw["enabled"]}
    if not isinstance(result["enabled"], bool):
        raise ValueError("training.safety_replay.enabled must be a boolean")
    for name in (
        "parent_samples", "ordinary_task3_samples", "own_bomb_cycle_samples",
        "fatal_prefix_steps",
    ):
        result[name] = _positive_int(raw[name], name)
    if sum(result[name] for name in (
        "parent_samples", "ordinary_task3_samples", "own_bomb_cycle_samples"
    )) != 64:
        raise ValueError("training.safety_replay sample counts must sum to 64")
    return result


def safety_replay_from_environment() -> dict[str, Any]:
    raw = os.getenv(SAFETY_REPLAY_SPEC_ENV)
    if raw is None:
        return resolve_safety_replay_spec()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{SAFETY_REPLAY_SPEC_ENV} must be valid JSON") from exception
    return resolve_safety_replay_spec(value)


def training_budget_from_environment() -> dict[str, Any]:
    raw = os.getenv(TRAINING_BUDGET_ENV)
    if raw is None:
        return {"target_stage_action_steps": None, "min_rounds": 1}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{TRAINING_BUDGET_ENV} must be valid JSON") from exception
    if not isinstance(value, dict) or set(value) != {
        "target_stage_action_steps", "min_rounds"
    }:
        raise ValueError("training budget has an incompatible contract")
    target = value["target_stage_action_steps"]
    if target is not None:
        target = _positive_int(target, "target_stage_action_steps")
    return {
        "target_stage_action_steps": target,
        "min_rounds": _positive_int(value["min_rounds"], "min_rounds"),
    }


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"training.retention.{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"training.retention.{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"training.retention.{name} must be a positive integer")
    return value

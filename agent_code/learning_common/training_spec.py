"""Versioned task-local training controls supplied by the experiment runner."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping


N_STEP_ENV = "BOMBERMAN_N_STEP"
RETENTION_SPEC_ENV = "BOMBERMAN_RETENTION_SPEC"
TRAINING_BUDGET_ENV = "BOMBERMAN_TRAINING_BUDGET"
DEFAULT_RETENTION_SPEC = {
    "parent_fraction": 0.5,
    "distillation_weight": 1.0,
    "temperature": 1.0,
    "per_task_capacity": 20_000,
    "current_warmup": 2_000,
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

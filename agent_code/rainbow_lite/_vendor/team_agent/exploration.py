"""Shared, versioned exploration schedules for trainable agents."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

import numpy as np


EXPLORATION_VERSION = "linear-v1"
DEFAULT_EXPLORATION_SPEC = {
    "version": EXPLORATION_VERSION,
    "start": 1.0,
    "end": 0.05,
    "decay_action_steps": 80_000,
}
AGENT_SEED_ENV = "BOMBERMAN_AGENT_SEED"
EXPLORATION_SPEC_ENV = "BOMBERMAN_EXPLORATION_SPEC"
SAFE_EXPLORATION_ENV = "BOMBERMAN_SAFE_EXPLORATION"


def resolve_exploration_spec(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and normalize one complete linear exploration specification."""
    raw = DEFAULT_EXPLORATION_SPEC if value is None else value
    if not isinstance(raw, Mapping):
        raise ValueError("training.exploration must be an object")
    required = {"version", "start", "end", "decay_action_steps"}
    if set(raw) != required:
        raise ValueError(
            "training.exploration must contain exactly: " + ", ".join(sorted(required))
        )
    if raw["version"] != EXPLORATION_VERSION:
        raise ValueError(f"Unsupported exploration version: {raw['version']!r}")
    start = _probability(raw["start"], "start")
    end = _probability(raw["end"], "end")
    if start < end:
        raise ValueError("training.exploration.start must be at least end")
    decay = raw["decay_action_steps"]
    if isinstance(decay, bool) or not isinstance(decay, int) or decay < 1:
        raise ValueError("training.exploration.decay_action_steps must be a positive integer")
    return {
        "version": EXPLORATION_VERSION,
        "start": start,
        "end": end,
        "decay_action_steps": decay,
    }


def epsilon_at(action_step: int, specification: Mapping[str, Any] | None = None) -> float:
    """Return epsilon before the decision at ``action_step``."""
    if isinstance(action_step, bool) or not isinstance(action_step, int) or action_step < 0:
        raise ValueError("action_step must be a non-negative integer")
    spec = resolve_exploration_spec(specification)
    progress = min(action_step / spec["decay_action_steps"], 1.0)
    return spec["start"] + (spec["end"] - spec["start"]) * progress


def agent_seed_from_environment(default: int = 0) -> int:
    """Read the runner-provided Agent seed, retaining a direct-use default."""
    raw = os.getenv(AGENT_SEED_ENV)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exception:
        raise ValueError(f"{AGENT_SEED_ENV} must be an integer") from exception
    return value


def exploration_from_environment() -> dict[str, Any]:
    """Read the complete runner-provided exploration spec, or use the default."""
    raw = os.getenv(EXPLORATION_SPEC_ENV)
    if raw is None:
        return resolve_exploration_spec()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{EXPLORATION_SPEC_ENV} must be valid JSON") from exception
    return resolve_exploration_spec(value)


def safe_exploration_from_environment(default: bool = False) -> bool:
    raw = os.getenv(SAFE_EXPLORATION_ENV)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{SAFE_EXPLORATION_ENV} must be a boolean")


def survivable_exploration_mask(
    game_state: dict,
    legal_mask: np.ndarray,
    *,
    allow_bomb: bool,
) -> tuple[np.ndarray, bool]:
    """Restrict a stochastic training choice to horizon-survivable actions.

    The returned boolean is true only when no survivable action existed and
    the function had to fall back to the physical curriculum mask.
    """
    from agent_code.rainbow_lite._vendor.team_agent.safety import survival_mask

    return survival_mask(
        game_state, legal_mask, allow_bomb=allow_bomb, horizon=7)


def _probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"training.exploration.{name} must be between 0 and 1")
    return float(value)

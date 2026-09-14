"""Versioned survival constraints shared by training and frozen inference."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np


SAFETY_VERSION = "survival-mask-v1"
SAFETY_VERSION_V2 = "survival-mask-v2"
SAFETY_SPEC_ENV = "BOMBERMAN_SAFETY_SPEC"
SAFETY_MODES = ("off", "exploration", "all")
DEFAULT_SAFETY_SPEC = {
    "version": SAFETY_VERSION,
    "mode": "off",
    "horizon": 7,
    "fallback": "physical_q",
}


def resolve_safety_spec(
    value: Mapping[str, Any] | None = None,
    *,
    legacy_safe_exploration: bool | None = None,
) -> dict[str, Any]:
    """Validate one complete policy or translate the legacy boolean."""
    if value is not None and legacy_safe_exploration is not None:
        raise ValueError("config.safety conflicts with training.safe_exploration")
    if value is None:
        mode = "exploration" if legacy_safe_exploration else "off"
        return {**DEFAULT_SAFETY_SPEC, "mode": mode}
    if not isinstance(value, Mapping):
        raise ValueError("config.safety must be an object")
    version = value.get("version")
    required = {"version", "mode", "horizon", "fallback"}
    if version == SAFETY_VERSION_V2:
        required.add("escape_area_fraction")
    if set(value) != required:
        raise ValueError(
            "config.safety must contain exactly: " + ", ".join(sorted(required)))
    if version not in {SAFETY_VERSION, SAFETY_VERSION_V2}:
        raise ValueError(f"Unsupported safety version: {value['version']!r}")
    if value["mode"] not in SAFETY_MODES:
        raise ValueError(f"Unsupported safety mode: {value['mode']!r}")
    if value["horizon"] != 7:
        raise ValueError(f"{version} requires horizon=7")
    if value["fallback"] != "physical_q":
        raise ValueError(f"{version} requires fallback='physical_q'")
    resolved = {
        "version": str(version),
        "mode": str(value["mode"]),
        "horizon": 7,
        "fallback": "physical_q",
    }
    if version == SAFETY_VERSION_V2:
        fraction = value["escape_area_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError("survival-mask-v2 escape_area_fraction must be numeric")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("survival-mask-v2 escape_area_fraction must be in (0,1]")
        resolved["escape_area_fraction"] = float(fraction)
    return resolved


def safety_from_environment() -> tuple[dict[str, Any], bool]:
    """Return the runtime contract and whether the caller configured it."""
    raw = os.getenv(SAFETY_SPEC_ENV)
    if raw is None:
        return resolve_safety_spec(), False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{SAFETY_SPEC_ENV} must be valid JSON") from exception
    return resolve_safety_spec(value), True


def survival_mask(
    game_state: dict,
    physical_mask: np.ndarray,
    *,
    allow_bomb: bool,
    horizon: int = 7,
) -> tuple[np.ndarray, bool]:
    """Return horizon-survivable actions, or the physical mask if none exist."""
    from agent_code.team_agent.feature_system.common import ACTIONS, build_safety_context

    if horizon != 7:
        raise ValueError("survival-mask-v1 requires horizon=7")
    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    context = build_safety_context(game_state)
    safe = np.zeros_like(physical)
    for index, action in enumerate(ACTIONS):
        reachability = (
            context.bomb_reachability
            if action == "BOMB" else context.movement_reachability[action]
        )
        safe[index] = bool(
            physical[index] and reachability and reachability.survives_horizon)
    if safe.any():
        return safe, False
    return physical, True


def margin_preserving_mask(
    game_state: dict,
    physical_mask: np.ndarray,
    *,
    allow_bomb: bool,
    own_bomb_pending: bool,
    escape_area_fraction: float = 0.75,
) -> tuple[np.ndarray, bool, bool]:
    """Apply v1, then reject low-area actions while escaping one's own bomb.

    Returns ``(mask, physical_fallback, margin_fallback)``.  The second-stage
    fallback is deliberately the full v1 set, never a hand-picked direction.
    """
    first, physical_fallback = survival_mask(
        game_state, physical_mask, allow_bomb=allow_bomb)
    if physical_fallback or not own_bomb_pending:
        return first, physical_fallback, False
    diagnostics = survival_diagnostics(
        game_state, physical_mask, allow_bomb=allow_bomb)
    areas = np.asarray(diagnostics["escape_area"], dtype=np.float64)
    best = float(np.max(areas[first]))
    refined = first & (areas >= float(escape_area_fraction) * best)
    if refined.any():
        return refined, False, False
    return first, False, True


def survival_diagnostics(
    game_state: dict, physical_mask: np.ndarray, *, allow_bomb: bool,
) -> dict[str, list[int | bool]]:
    """Return factual per-action reachability used for experiment auditing."""
    from agent_code.team_agent.feature_system.common import ACTIONS, build_safety_context

    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    context = build_safety_context(game_state)
    survives, safe_horizon, escape_area = [], [], []
    for index, action in enumerate(ACTIONS):
        reachability = (
            context.bomb_reachability
            if action == "BOMB" else context.movement_reachability[action]
        )
        allowed = bool(physical[index] and reachability)
        survives.append(bool(allowed and reachability.survives_horizon))
        safe_horizon.append(int(reachability.safe_horizon) if allowed else 0)
        escape_area.append(int(reachability.reachable_area) if allowed else 0)
    return {
        "survives_horizon": survives,
        "safe_horizon": safe_horizon,
        "escape_area": escape_area,
    }


def mask_for_decision(
    game_state: dict,
    physical_mask: np.ndarray,
    safety_spec: Mapping[str, Any],
    *,
    allow_bomb: bool,
    exploring: bool,
    own_bomb_pending: bool = False,
) -> tuple[np.ndarray, bool]:
    """Apply the configured constraint for one behavior decision."""
    spec = resolve_safety_spec(safety_spec)
    enabled = spec["mode"] == "all" or (
        spec["mode"] == "exploration" and exploring)
    if not enabled:
        return np.asarray(physical_mask, dtype=bool).copy(), False
    if spec["version"] == SAFETY_VERSION_V2:
        decision, physical_fallback, _ = margin_preserving_mask(
            game_state, physical_mask, allow_bomb=allow_bomb,
            own_bomb_pending=own_bomb_pending,
            escape_area_fraction=float(spec["escape_area_fraction"]),
        )
        return decision, physical_fallback
    return survival_mask(
        game_state, physical_mask, allow_bomb=allow_bomb,
        horizon=int(spec["horizon"]))


def avoidable_fatal_action(
    game_state: dict,
    action: str,
    physical_mask: np.ndarray,
    *,
    allow_bomb: bool,
) -> bool:
    """Whether a selected doomed action had at least one survivable alternative."""
    from agent_code.team_agent.feature_system.common import ACTIONS, build_safety_context

    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    context = build_safety_context(game_state)
    survivable = []
    for index, candidate in enumerate(ACTIONS):
        reachability = (
            context.bomb_reachability
            if candidate == "BOMB" else context.movement_reachability[candidate]
        )
        survivable.append(bool(
            physical[index] and reachability and reachability.survives_horizon))
    action_index = ACTIONS.index(action)
    return bool(physical[action_index] and not survivable[action_index] and any(survivable))

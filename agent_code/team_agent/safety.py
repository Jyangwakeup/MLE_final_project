"""Versioned survival constraints shared by training and frozen inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SAFETY_VERSION = "survival-mask-v1"
ROBUST_SAFETY_VERSION = "survival-mask-v3"
SAFETY_SPEC_ENV = "BOMBERMAN_SAFETY_SPEC"
SAFETY_MODES = ("off", "exploration", "all")
DEFAULT_SAFETY_SPEC = {
    "version": SAFETY_VERSION,
    "mode": "off",
    "horizon": 7,
    "fallback": "physical_q",
}


@dataclass(frozen=True)
class SafetyDecision:
    mask: np.ndarray
    v1_mask: np.ndarray
    physical_fallback: bool
    robust_fallback: bool
    route_counts: tuple[int, ...]
    escape_slack: tuple[float, ...]


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
    if version == ROBUST_SAFETY_VERSION:
        required |= {"required_independent_routes", "robust_fallback"}
    if set(value) != required:
        raise ValueError(
            "config.safety must contain exactly: " + ", ".join(sorted(required)))
    if version not in {SAFETY_VERSION, ROBUST_SAFETY_VERSION}:
        raise ValueError(f"Unsupported safety version: {value['version']!r}")
    if value["mode"] not in SAFETY_MODES:
        raise ValueError(f"Unsupported safety mode: {value['mode']!r}")
    if value["horizon"] != 7:
        raise ValueError(f"{version} requires horizon=7")
    if value["fallback"] != "physical_q":
        raise ValueError(f"{version} requires fallback='physical_q'")
    result = {
        "version": str(version),
        "mode": str(value["mode"]),
        "horizon": 7,
        "fallback": "physical_q",
    }
    if version == ROBUST_SAFETY_VERSION:
        if value["required_independent_routes"] != 2:
            raise ValueError("survival-mask-v3 requires two independent routes")
        if value["robust_fallback"] != SAFETY_VERSION:
            raise ValueError("survival-mask-v3 must fall back to survival-mask-v1")
        result.update({
            "required_independent_routes": 2,
            "robust_fallback": SAFETY_VERSION,
        })
    return result


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
    decision = safety_decision(
        game_state, physical_mask, spec, allow_bomb=allow_bomb,
        exploring=exploring, own_bomb_pending=own_bomb_pending)
    return decision.mask, decision.physical_fallback


def safety_decision(
    game_state: dict,
    physical_mask: np.ndarray,
    safety_spec: Mapping[str, Any],
    *,
    allow_bomb: bool,
    exploring: bool,
    own_bomb_pending: bool = False,
) -> SafetyDecision:
    """Apply v1 or the own-bomb redundant-route veto."""
    from agent_code.team_agent.feature_system.common import ACTIONS
    from agent_code.team_agent.temporal_safety_features import (
        robust_routes_after_first_step,
    )

    spec = resolve_safety_spec(safety_spec)
    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    enabled = spec["mode"] == "all" or (
        spec["mode"] == "exploration" and exploring)
    if not enabled:
        return SafetyDecision(
            physical, physical.copy(), False, False, (0,) * 6, (-1.0,) * 6)
    base, physical_fallback = survival_mask(
        game_state, physical, allow_bomb=allow_bomb,
        horizon=int(spec["horizon"]))
    if spec["version"] != ROBUST_SAFETY_VERSION or physical_fallback:
        return SafetyDecision(
            base, base.copy(), physical_fallback, False,
            (0,) * 6, (-1.0,) * 6)

    position = tuple(game_state["self"][3])
    routes, slack = [0] * len(ACTIONS), [-1.0] * len(ACTIONS)
    from agent_code.team_agent.temporal_safety_features import temporal_maps
    candidates = range(len(ACTIONS) - 1) if own_bomb_pending else (
        ACTIONS.index("BOMB"),)
    normal_maps = None
    for index in candidates:
        action = ACTIONS[index]
        if action == "BOMB":
            danger, blocked = temporal_maps(
                game_state, int(spec["horizon"]), hypothetical_bomb=True)
            first_action = "WAIT"
        else:
            if normal_maps is None:
                normal_maps = temporal_maps(
                    game_state, int(spec["horizon"]), hypothetical_bomb=False)
            danger, blocked = normal_maps
            first_action = action
        result = robust_routes_after_first_step(
            position, first_action, danger, blocked,
            required_routes=int(spec["required_independent_routes"]))
        routes[index] = result.independent_routes
        slack[index] = result.escape_slack

    robust = base & (np.asarray(routes) >= int(spec["required_independent_routes"]))
    bomb_index = ACTIONS.index("BOMB")
    robust_fallback = False
    if own_bomb_pending:
        if robust.any():
            selected = robust
        else:
            selected = base
            robust_fallback = True
    else:
        selected = base.copy()
        safe_non_bomb = base.copy()
        safe_non_bomb[bomb_index] = False
        if base[bomb_index] and safe_non_bomb.any() and not robust[bomb_index]:
            selected[bomb_index] = False
    return SafetyDecision(
        selected, base.copy(), False, robust_fallback,
        tuple(routes), tuple(slack))


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

"""Versioned survival constraints shared by training and frozen inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SAFETY_VERSION = "survival-mask-v1"
ROBUST_SAFETY_VERSION = "survival-mask-v3"
OPPONENT_ROBUST_SAFETY_VERSION = "survival-mask-v4"
CONTROLLABLE_SAFETY_VERSION = "survival-mask-v5"
SAFETY_VERSION_V2 = "survival-mask-v2"
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
    opponent_fallback: bool = False
    opponent_scenario_counts: tuple[int, ...] = (0,) * 6
    opponent_passing_counts: tuple[int, ...] = (0,) * 6
    opponent_failing_profiles: tuple[tuple[str, ...] | None, ...] = (None,) * 6
    opponent_failing_orders: tuple[tuple[int, ...] | None, ...] = (None,) * 6
    robust_guarantee_loss: bool = False
    robust_search_timed_out: bool = False
    robust_states_evaluated: int = 0


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
    if version in {ROBUST_SAFETY_VERSION, OPPONENT_ROBUST_SAFETY_VERSION}:
        required |= {"required_independent_routes", "robust_fallback"}
    if version == OPPONENT_ROBUST_SAFETY_VERSION:
        required |= {
            "opponent_transition_horizon", "opponent_action_space",
            "include_opponent_bombs", "execution_orders",
            "minimum_scenario_routes", "opponent_robust_fallback",
        }
    if version == CONTROLLABLE_SAFETY_VERSION:
        required |= {
            "opponent_action_space", "include_opponent_bombs",
            "execution_orders", "danger_interval", "search_budget_ms",
            "robust_fallback",
        }
    if version == SAFETY_VERSION_V2:
        required.add("escape_area_fraction")
    if set(value) != required:
        raise ValueError(
            "config.safety must contain exactly: " + ", ".join(sorted(required)))
    if version not in {
        SAFETY_VERSION, ROBUST_SAFETY_VERSION, OPPONENT_ROBUST_SAFETY_VERSION,
        CONTROLLABLE_SAFETY_VERSION, SAFETY_VERSION_V2,
    }:
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
    if version in {ROBUST_SAFETY_VERSION, OPPONENT_ROBUST_SAFETY_VERSION}:
        if value["required_independent_routes"] != 2:
            raise ValueError("survival-mask-v3 requires two independent routes")
        if value["robust_fallback"] != SAFETY_VERSION:
            raise ValueError("survival-mask-v3 must fall back to survival-mask-v1")
        result.update({
            "required_independent_routes": 2,
            "robust_fallback": SAFETY_VERSION,
        })
    if version == OPPONENT_ROBUST_SAFETY_VERSION:
        expected = {
            "opponent_transition_horizon": 1,
            "opponent_action_space": "all_physical",
            "include_opponent_bombs": True,
            "execution_orders": "all",
            "minimum_scenario_routes": 1,
            "opponent_robust_fallback": ROBUST_SAFETY_VERSION,
        }
        for field, expected_value in expected.items():
            if value[field] != expected_value:
                raise ValueError(
                    f"survival-mask-v4 requires {field}={expected_value!r}")
        result.update(expected)
    if version == CONTROLLABLE_SAFETY_VERSION:
        expected = {
            "opponent_action_space": "all_physical",
            "include_opponent_bombs": True,
            "execution_orders": "all",
            "danger_interval": "own_bomb_and_lingering",
            "search_budget_ms": 400,
            "robust_fallback": SAFETY_VERSION,
        }
        for field, expected_value in expected.items():
            if value[field] != expected_value:
                raise ValueError(
                    f"survival-mask-v5 requires {field}={expected_value!r}")
        result.update(expected)
    if version == SAFETY_VERSION_V2:
        fraction = value["escape_area_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError("survival-mask-v2 escape_area_fraction must be numeric")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("survival-mask-v2 escape_area_fraction must be in (0,1]")
        result["escape_area_fraction"] = float(fraction)
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
    context=None,
) -> tuple[np.ndarray, bool]:
    """Return horizon-survivable actions, or the physical mask if none exist."""
    from agent_code.team_agent.feature_system.common import ACTIONS, build_safety_context

    if horizon != 7:
        raise ValueError("survival-mask-v1 requires horizon=7")
    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    context = context if context is not None else build_safety_context(game_state)
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
    game_state: dict, physical_mask: np.ndarray, *, allow_bomb: bool, context=None,
) -> dict[str, list[int | bool]]:
    """Return factual per-action reachability used for experiment auditing."""
    from agent_code.team_agent.feature_system.common import ACTIONS, build_safety_context

    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    context = context if context is not None else build_safety_context(game_state)
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
    own_bomb_state: Mapping[str, Any] | None = None,
    context=None,
) -> tuple[np.ndarray, bool]:
    """Apply the configured constraint for one behavior decision."""
    spec = resolve_safety_spec(safety_spec)
    enabled = spec["mode"] == "all" or (
        spec["mode"] == "exploration" and exploring)
    if not enabled:
        return np.asarray(physical_mask, dtype=bool).copy(), False
    decision = safety_decision(
        game_state, physical_mask, spec, allow_bomb=allow_bomb,
        exploring=exploring, own_bomb_pending=own_bomb_pending,
        own_bomb_state=own_bomb_state, context=context)
    return decision.mask, decision.physical_fallback


def safety_decision(
    game_state: dict,
    physical_mask: np.ndarray,
    safety_spec: Mapping[str, Any],
    *,
    allow_bomb: bool,
    exploring: bool,
    own_bomb_pending: bool = False,
    own_bomb_state: Mapping[str, Any] | None = None,
    context=None,
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
    if spec["version"] == SAFETY_VERSION_V2:
        selected, fallback, _ = margin_preserving_mask(
            game_state, physical, allow_bomb=allow_bomb,
            own_bomb_pending=own_bomb_pending,
            escape_area_fraction=float(spec["escape_area_fraction"]))
        return SafetyDecision(selected, selected.copy(), fallback, False, (0,) * 6, (-1.0,) * 6)
    base, physical_fallback = survival_mask(
        game_state, physical, allow_bomb=allow_bomb,
        horizon=int(spec["horizon"]), context=context)
    if spec["version"] == CONTROLLABLE_SAFETY_VERSION and not physical_fallback:
        from agent_code.team_agent.controllable_survival import (
            controllable_survival_actions, danger_interval_steps,
        )
        bomb_state = dict(own_bomb_state or {})
        bomb_state.setdefault("pending", own_bomb_pending)
        bomb_index = ACTIONS.index("BOMB")
        if own_bomb_pending:
            candidate_indices = np.flatnonzero(base).tolist()
            candidate_indices = [index for index in candidate_indices
                                 if index != bomb_index]
            placing_bomb = False
        else:
            candidate_indices = [bomb_index] if base[bomb_index] else []
            placing_bomb = True
        candidates = tuple(ACTIONS[index] for index in candidate_indices)
        remaining = danger_interval_steps(
            bomb_state, placing_bomb=placing_bomb)
        result = controllable_survival_actions(
            game_state, candidates, remaining_steps=remaining,
            budget_ms=int(spec["search_budget_ms"])) if candidates else None
        proven = set(() if result is None else result.proven_actions)
        selected = base.copy()
        opponent_fallback = False
        guarantee_loss = False
        if own_bomb_pending:
            robust = np.asarray(
                [action in proven for action in ACTIONS], dtype=bool) & base
            if robust.any():
                selected = robust
            else:
                opponent_fallback = True
                guarantee_loss = True
        else:
            safe_non_bomb = base.copy()
            safe_non_bomb[bomb_index] = False
            if safe_non_bomb.any() and "BOMB" not in proven:
                selected[bomb_index] = False
        counts = [0] * len(ACTIONS)
        passing = [0] * len(ACTIONS)
        if result is not None:
            target_indices = candidate_indices or [bomb_index]
            for index in target_indices:
                counts[index] = int(result.scenarios_evaluated)
                passing[index] = counts[index] if ACTIONS[index] in proven else 0
        failing_profiles = [None] * len(ACTIONS)
        failing_orders = [None] * len(ACTIONS)
        if result is not None and candidate_indices:
            failing_profiles[candidate_indices[0]] = result.first_failing_profile
            failing_orders[candidate_indices[0]] = result.first_failing_order
        return SafetyDecision(
            selected, base.copy(), False, False, (0,) * 6, (-1.0,) * 6,
            opponent_fallback, tuple(counts), tuple(passing),
            tuple(failing_profiles), tuple(failing_orders), guarantee_loss,
            False if result is None else result.timed_out,
            0 if result is None else result.states_evaluated,
        )
    if spec["version"] not in {
        ROBUST_SAFETY_VERSION, OPPONENT_ROBUST_SAFETY_VERSION,
    } or physical_fallback:
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
    if spec["version"] != OPPONENT_ROBUST_SAFETY_VERSION:
        return SafetyDecision(
            selected, base.copy(), False, robust_fallback,
            tuple(routes), tuple(slack))

    from agent_code.team_agent.temporal_safety_features import (
        opponent_robust_survival_after_action,
    )
    scenario_counts = [0] * len(ACTIONS)
    passing_counts = [0] * len(ACTIONS)
    failing_profiles = [None] * len(ACTIONS)
    failing_orders = [None] * len(ACTIONS)
    opponent_safe = np.zeros(len(ACTIONS), dtype=bool)
    opponent_candidates = (
        np.flatnonzero(selected).tolist()
        if own_bomb_pending else [bomb_index]
    )
    for index in opponent_candidates:
        if not selected[index]:
            continue
        result = opponent_robust_survival_after_action(
            game_state, ACTIONS[index], horizon=int(spec["horizon"]))
        scenario_counts[index] = result.scenario_count
        passing_counts[index] = result.passing_scenarios
        failing_profiles[index] = result.first_failing_profile
        failing_orders[index] = result.first_failing_order
        opponent_safe[index] = result.survives_all

    opponent_fallback = False
    if own_bomb_pending:
        opponent_selected = selected & opponent_safe
        if opponent_selected.any():
            selected = opponent_selected
        else:
            opponent_fallback = True
    else:
        safe_non_bomb = base.copy()
        safe_non_bomb[bomb_index] = False
        if selected[bomb_index] and safe_non_bomb.any() and not opponent_safe[bomb_index]:
            selected[bomb_index] = False
    return SafetyDecision(
        selected, base.copy(), False, robust_fallback,
        tuple(routes), tuple(slack), opponent_fallback,
        tuple(scenario_counts), tuple(passing_counts),
        tuple(failing_profiles), tuple(failing_orders))
    if spec["version"] == SAFETY_VERSION_V2:
        decision, physical_fallback, _ = margin_preserving_mask(
            game_state, physical_mask, allow_bomb=allow_bomb,
            own_bomb_pending=own_bomb_pending,
            escape_area_fraction=float(spec["escape_area_fraction"]),
        )
        return decision, physical_fallback
    return survival_mask(
        game_state, physical_mask, allow_bomb=allow_bomb,
        horizon=int(spec["horizon"]), context=context)


def avoidable_fatal_action(
    game_state: dict,
    action: str,
    physical_mask: np.ndarray,
    *,
    allow_bomb: bool,
    context=None,
) -> bool:
    """Whether a selected doomed action had at least one survivable alternative."""
    from agent_code.team_agent.feature_system.common import ACTIONS, build_safety_context

    physical = np.asarray(physical_mask, dtype=bool).copy()
    if not allow_bomb:
        physical[ACTIONS.index("BOMB")] = False
    context = context if context is not None else build_safety_context(game_state)
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

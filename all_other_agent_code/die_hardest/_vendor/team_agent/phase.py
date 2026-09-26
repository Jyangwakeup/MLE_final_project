"""Observable match-phase facts shared by features, rewards, and diagnostics."""

from __future__ import annotations

from math import exp, log1p
from typing import Any

import numpy as np

import settings as s
from .feature_system.common import (
    ACTIONS, build_safety_context, crate_frontiers, distance_to_targets,
    navigation_blocked,
)


PHASE_VERSION = "observable-phase-v1"
PHASE_STATE_FIELDS = (
    "early_weight", "middle_weight", "late_weight",
    "crate_depletion", "opponent_depletion", "score_margin",
    "self_mobility", "opponent_closeness", "opponent_mobility",
    "late_mobility_crisis",
)


def phase_weights(progress: float) -> tuple[float, float, float]:
    """Return the continuous triangular early/middle/late mixture."""
    progress = float(np.clip(progress, 0.0, 1.0))
    early = max(0.0, 1.0 - 2.0 * progress)
    late = max(0.0, 2.0 * progress - 1.0)
    middle = 1.0 - early - late
    return early, middle, late


def init_phase_history(owner) -> None:
    owner.phase_history_round = None
    owner.phase_initial_crates = 0
    owner.phase_initial_opponents = 0
    owner.phase_facts_cache_key = None
    owner.phase_facts_cache_value = None


def phase_history_state(owner) -> dict[str, int | None]:
    return {
        "version": PHASE_VERSION,
        "round": getattr(owner, "phase_history_round", None),
        "initial_crates": int(getattr(owner, "phase_initial_crates", 0)),
        "initial_opponents": int(getattr(owner, "phase_initial_opponents", 0)),
    }


def load_phase_history_state(owner, state: dict[str, Any] | None) -> None:
    init_phase_history(owner)
    if not state:
        return
    if state.get("version") != PHASE_VERSION:
        raise ValueError("checkpoint uses an incompatible phase-history version")
    owner.phase_history_round = state.get("round")
    owner.phase_initial_crates = int(state.get("initial_crates", 0))
    owner.phase_initial_opponents = int(state.get("initial_opponents", 0))


def ensure_phase_history(owner, game_state: dict) -> None:
    round_index = game_state.get("round")
    if getattr(owner, "phase_history_round", None) == round_index:
        return
    owner.phase_history_round = round_index
    owner.phase_initial_crates = int(np.count_nonzero(game_state["field"] == 1))
    owner.phase_initial_opponents = len(game_state.get("others", ()))
    owner.phase_facts_cache_key = None
    owner.phase_facts_cache_value = None


def _mobility(game_state: dict) -> tuple[float, int, float]:
    """Return normalized best H=7 endpoint reserve, safe actions, raw ratio."""
    context = build_safety_context(game_state)
    reachable = [
        item.reachable_area
        for item in context.movement_reachability.values()
        if item is not None and item.survives_horizon
    ]
    if context.bomb_reachability is not None and context.bomb_reachability.survives_horizon:
        reachable.append(context.bomb_reachability.reachable_area)
    safe_actions = len(reachable)
    walkable = max(1, int(np.count_nonzero(game_state["field"] == 0)))
    best_area = max(reachable, default=0)
    raw_ratio = min(1.0, best_area / float(walkable))
    normalized = log1p(float(best_area)) / log1p(float(walkable))
    return float(normalized), safe_actions, float(raw_ratio)


def _opponent_view(game_state: dict, opponent: tuple) -> dict:
    name, score, can_bomb, position = opponent
    own = game_state["self"]
    return {
        **game_state,
        "self": (name, score, can_bomb, position),
        "others": [own, *(item for item in game_state.get("others", ()) if item is not opponent)],
    }


def phase_facts(
    game_state: dict,
    *,
    initial_crates: int,
    initial_opponents: int,
) -> dict[str, float]:
    current_crates = int(np.count_nonzero(game_state["field"] == 1))
    crate_depletion = (
        0.0 if initial_crates <= 0
        else float(np.clip(1.0 - current_crates / float(initial_crates), 0.0, 1.0))
    )
    alive_opponents = list(game_state.get("others", ()))
    opponent_depletion = (
        0.0 if initial_opponents <= 0
        else float(np.clip(
            1.0 - len(alive_opponents) / float(initial_opponents), 0.0, 1.0))
    )
    round_progress = (
        min(max(int(game_state.get("step", 1)), 1), s.MAX_STEPS) - 1
    ) / float(max(1, s.MAX_STEPS - 1))
    progress = (
        0.45 * crate_depletion + 0.30 * opponent_depletion
        + 0.25 * round_progress
    )
    early, middle, late = phase_weights(progress)

    self_score = float(game_state["self"][1])
    score_margin = 0.0
    if alive_opponents:
        score_margin = float(np.clip(
            (self_score - max(float(item[1]) for item in alive_opponents)) / 10.0,
            -1.0, 1.0,
        ))
    self_mobility, safe_actions, endpoint_ratio = _mobility(game_state)

    opponent_closeness = 0.0
    opponent_mobility = 0.0
    if alive_opponents:
        position = tuple(game_state["self"][3])
        nearest = min(
            alive_opponents,
            key=lambda item: (
                abs(item[3][0] - position[0]) + abs(item[3][1] - position[1]),
                item[0],
            ),
        )
        distance = abs(nearest[3][0] - position[0]) + abs(nearest[3][1] - position[1])
        opponent_closeness = exp(-float(distance) / 4.0)
        opponent_mobility, _, _ = _mobility(_opponent_view(game_state, nearest))

    crisis = bool(safe_actions <= 1 and endpoint_ratio <= 0.10)
    return {
        "progress": float(progress),
        "early_weight": float(early),
        "middle_weight": float(middle),
        "late_weight": float(late),
        "crate_depletion": crate_depletion,
        "opponent_depletion": opponent_depletion,
        "score_margin": score_margin,
        "self_mobility": self_mobility,
        "safe_action_fraction": safe_actions / float(len(ACTIONS)),
        "safe_actions": int(safe_actions),
        "safe_endpoint_ratio": endpoint_ratio,
        "opponent_closeness": float(opponent_closeness),
        "opponent_mobility": float(opponent_mobility),
        "late_mobility_crisis": float(late > 0.0 and crisis),
    }


def phase_facts_for_owner(owner, game_state: dict) -> dict[str, float]:
    ensure_phase_history(owner, game_state)
    key = (game_state.get("round"), game_state.get("step"))
    if getattr(owner, "phase_facts_cache_key", None) == key:
        return dict(owner.phase_facts_cache_value)
    facts = phase_facts(
        game_state,
        initial_crates=int(owner.phase_initial_crates),
        initial_opponents=int(owner.phase_initial_opponents),
    )
    owner.phase_facts_cache_key = key
    owner.phase_facts_cache_value = dict(facts)
    return facts


def objective_closeness(game_state: dict) -> tuple[float, float]:
    """Return bounded reachable coin and crate-frontier closeness."""
    blocked = navigation_blocked(game_state)
    position = tuple(game_state["self"][3])
    coin_distance = float(distance_to_targets(
        blocked, tuple(game_state.get("coins", ())))[position])
    crate_distance = float(distance_to_targets(
        blocked, crate_frontiers(game_state, blocked))[position])
    return (
        0.0 if not np.isfinite(coin_distance) else exp(-coin_distance / 4.0),
        0.0 if not np.isfinite(crate_distance) else exp(-crate_distance / 4.0),
    )

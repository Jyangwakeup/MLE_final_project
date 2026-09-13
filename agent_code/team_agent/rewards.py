"""Versioned rewards shared by all learning agents."""

from __future__ import annotations

from math import exp
from typing import Sequence

import numpy as np

import events as e


REWARD_VERSION = "r1"
REWARD_SPECS = {
    "r1": {
        "step": -0.01,
        "coin_collected": 1.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "death": -10.0,
        "invalid_action": -0.1,
    },
    "r1_no_crate": {
        "step": -0.01,
        "coin_collected": 1.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.0,
        "death": -10.0,
        "invalid_action": -0.1,
    },
    "r2_balanced": {
        "step": -0.01,
        "coin_collected": 3.0,
        "coin_found": 0.25,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.1,
        "killed_self": -7.0,
        "got_killed": -5.0,
        "survived_round": 0.25,
        "invalid_action": -0.2,
    },
    "r3_potential": {
        "step": -0.01,
        "coin_collected": 3.0,
        "coin_found": 0.25,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.1,
        "killed_self": -7.0,
        "got_killed": -5.0,
        "survived_round": 0.25,
        "invalid_action": -0.2,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_safety_weight": 0.25,
    },
    "r4_anti_oscillation": {
        "step": -0.01,
        "coin_collected": 3.0,
        "coin_found": 0.25,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.1,
        "killed_self": -7.0,
        "got_killed": -5.0,
        "survived_round": 0.25,
        "invalid_action": -0.2,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_safety_weight": 0.25,
        "oscillation_penalty": -0.08,
        "idle_penalty_step": -0.04,
        "idle_penalty_cap": 3,
    },
}
DEATH_EVENTS = frozenset((e.KILLED_SELF, e.GOT_KILLED))


def resolve_reward_spec(version: str = REWARD_VERSION) -> dict[str, float]:
    """Return an independent copy of a registered reward definition."""
    try:
        return dict(REWARD_SPECS[version])
    except KeyError as exception:
        raise ValueError(f"Unknown reward version: {version!r}") from exception


def identify_checkpoint_reward_version(
    version: str | None, spec: dict[str, float] | None,
) -> str | None:
    """Return the embedded reward ID; retained for metadata compatibility."""
    return version


def _event_reward(events: Sequence[str], spec: dict[str, float]) -> float:
    """Calculate event rewards, counting a transition death at most once."""
    if "death" in spec:
        event_rewards = {
            e.COIN_COLLECTED: spec["coin_collected"],
            e.KILLED_OPPONENT: spec["killed_opponent"],
            e.CRATE_DESTROYED: spec["crate_destroyed"],
            e.INVALID_ACTION: spec["invalid_action"],
        }
        reward = spec["step"]
        for event, event_reward in event_rewards.items():
            reward += events.count(event) * event_reward
        if DEATH_EVENTS.intersection(events):
            reward += spec["death"]
        return float(reward)

    event_rewards = {
        e.COIN_COLLECTED: spec["coin_collected"],
        e.COIN_FOUND: spec["coin_found"],
        e.KILLED_OPPONENT: spec["killed_opponent"],
        e.CRATE_DESTROYED: spec["crate_destroyed"],
        e.INVALID_ACTION: spec["invalid_action"],
        e.SURVIVED_ROUND: spec["survived_round"],
    }
    reward = spec["step"]
    for event, event_reward in event_rewards.items():
        reward += events.count(event) * event_reward
    if e.KILLED_SELF in events:
        reward += spec["killed_self"]
    elif e.GOT_KILLED in events:
        reward += spec["got_killed"]
    return float(reward)


def _state_potential(game_state: dict, spec: dict[str, float]) -> float:
    """Return a bounded, state-only progress and safety potential."""
    from agent_code.team_agent.danger import HORIZON, predict_danger
    from agent_code.team_agent.feature_system.common import (
        crate_frontiers, distance_to_targets, navigation_blocked,
    )

    position = game_state["self"][3]
    blocked = navigation_blocked(game_state)
    value = 0.0
    coin_distances = distance_to_targets(blocked, tuple(game_state["coins"]))
    coin_distance = float(coin_distances[position])
    if np.isfinite(coin_distance):
        value += spec["potential_coin_weight"] * exp(-coin_distance / 4.0)
    else:
        crate_distances = distance_to_targets(
            blocked, crate_frontiers(game_state, blocked))
        crate_distance = float(crate_distances[position])
        if np.isfinite(crate_distance):
            value += spec["potential_crate_weight"] * exp(-crate_distance / 4.0)

    danger = predict_danger(game_state, horizon=HORIZON).danger
    dangerous_steps = np.flatnonzero(danger[1:, position[0], position[1]])
    earliest = HORIZON + 1 if dangerous_steps.size == 0 else int(dangerous_steps[0]) + 1
    safety = 1.0 if earliest > HORIZON else max(0.0, earliest - 1) / HORIZON
    value += spec["potential_safety_weight"] * safety
    return float(value)


def reward_from_events(
    events: Sequence[str],
    version: str = REWARD_VERSION,
    *,
    old_game_state: dict | None = None,
    new_game_state: dict | None = None,
    terminal: bool = False,
    repeated_oscillation: bool = False,
    idle_streak: int = 0,
) -> float:
    """Convert framework events and optional temporal context into a scalar."""
    spec = resolve_reward_spec(version)
    reward = _event_reward(events, spec)
    if version in {"r3_potential", "r4_anti_oscillation"} and old_game_state is not None:
        old_potential = _state_potential(old_game_state, spec)
        next_potential = (
            0.0 if terminal or new_game_state is None
            else _state_potential(new_game_state, spec)
        )
        reward += spec["potential_gamma"] * next_potential - old_potential
    if version == "r4_anti_oscillation" and repeated_oscillation:
        reward += spec["oscillation_penalty"]
    if version == "r4_anti_oscillation" and idle_streak >= 2:
        multiplier = min(idle_streak - 1, int(spec["idle_penalty_cap"]))
        reward += spec["idle_penalty_step"] * multiplier
    return float(reward)

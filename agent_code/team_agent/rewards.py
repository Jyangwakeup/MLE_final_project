"""Versioned rewards shared by all learning agents."""

from __future__ import annotations

from math import exp
from typing import Sequence

import numpy as np

import events as e


REWARD_VERSION = "r1"
TASK4_SCORE_REWARD = "task4-score-v1"
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
    "r1_coin3": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "death": -10.0,
        "invalid_action": -0.1,
    },
    "r1_coin3_no_crate": {
        "step": -0.01,
        "coin_collected": 3.0,
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
    "r5_coin_potential": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "death": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
    },
    "r5_conditional_loop": {
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
        "conditional_loop_penalty": -0.08,
        "avoidable_wait_penalty": -0.04,
    },
    "r6_safe_sparse": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -10.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
    },
    "r6_safe_potential": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -10.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
    },
    "r7_safe_credit_sparse": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
    },
    "r7_safe_credit_potential": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
    },
    "r20_safe_credit_targeted_wait": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "avoidable_wait_penalty": -0.04,
        "useful_bomb_counts_as_wait_progress": 1.0,
    },
    "r21_safe_credit_targeted_reverse": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "avoidable_wait_penalty": -0.04,
        "useful_bomb_counts_as_wait_progress": 1.0,
        "conditional_loop_penalty": -0.08,
    },
    "r8_safe_constrained": {
        "step": -0.01,
        "coin_collected": 3.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "avoidable_fatal_action": -20.0,
        "suicide_dominates_positive_events": 1.0,
    },
    "r9_phase_resource": {
        "step": -0.01,
        "coin_early": 3.0, "coin_middle": 1.5, "coin_late": 1.0,
        "crate_early": 0.2, "crate_middle": 0.05, "crate_late": 0.0,
        "kill_early": 5.0, "kill_middle": 8.0, "kill_late": 7.5,
        "killed_self": -30.0, "got_killed": -10.0,
        "invalid_action": -0.1, "unsafe_bomb_penalty": -20.0,
        "potential_gamma": 0.95, "potential_danger_weight": 1.0,
        "phase_resource_potential": 1.0,
        "suicide_dominates_positive_events": 1.0,
    },
    "r9_phase_combat": {
        "step": -0.01,
        "coin_early": 3.0, "coin_middle": 1.5, "coin_late": 1.0,
        "crate_early": 0.2, "crate_middle": 0.05, "crate_late": 0.0,
        "kill_early": 5.0, "kill_middle": 8.0, "kill_late": 7.5,
        "killed_self": -30.0, "got_killed": -10.0,
        "invalid_action": -0.1, "unsafe_bomb_penalty": -20.0,
        "potential_gamma": 0.95, "potential_danger_weight": 1.0,
        "phase_resource_potential": 1.0, "phase_combat_potential": 1.0,
        "suicide_dominates_positive_events": 1.0,
    },
    "r9_phase_full": {
        "step": -0.01,
        "coin_early": 3.0, "coin_middle": 1.5, "coin_late": 1.0,
        "crate_early": 0.2, "crate_middle": 0.05, "crate_late": 0.0,
        "kill_early": 5.0, "kill_middle": 8.0, "kill_late": 7.5,
        "killed_self": -30.0, "got_killed": -10.0,
        "invalid_action": -0.1, "unsafe_bomb_penalty": -20.0,
        "potential_gamma": 0.95, "potential_danger_weight": 1.0,
        "phase_resource_potential": 1.0, "phase_combat_potential": 1.0,
        "phase_mobility_potential": 1.0,
        "suicide_dominates_positive_events": 1.0,
    },
    "r9_safe_credit_anti_loop": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.2,
        "killed_opponent": 5.0,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "useless_bomb_penalty": -0.2,
        "conditional_loop_penalty": -0.08,
        "avoidable_wait_penalty": -0.04,
        "avoidable_fatal_action": -20.0,
        "suicide_dominates_positive_events": 1.0,
    },
    "r10_bounded_history_anti_loop": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.2,
        "killed_opponent": 5.0,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "avoidable_fatal_action": -20.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.04,
        "history_loop_penalty_cap": 3,
        "history_wait_penalty_step": -0.04,
        "history_wait_penalty_cap": 3,
    },
    "r11_causal_bomb_credit": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.0,
        "killed_opponent": 5.0,
        "killed_self": -40.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useless_bomb_penalty": -0.3,
        "avoidable_fatal_action": -20.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.04,
        "history_loop_penalty_cap": 3,
        "history_wait_penalty_step": -0.04,
        "history_wait_penalty_cap": 3,
        "causal_bomb_death_penalty": -25.0,
        "resolved_bomb_survival_reward": 3.0,
        "resolved_bomb_crate_reward": 0.1,
    },
    "r12_coin_priority_anti_loop": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.1,
        "killed_opponent": 5.0,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 1.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.1,
        "useful_bomb_crate_cap": 3,
        "suppress_useful_bomb_when_coin_reachable": 1.0,
        "avoidable_fatal_action": -20.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.2,
        "history_loop_penalty_cap": 5,
        "history_wait_penalty_step": -0.08,
        "history_wait_penalty_cap": 5,
    },
    "r13_no_safety_survival_credit": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.2,
        "killed_opponent": 5.0,
        "killed_self": -40.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "potential_survival_options_weight": 2.0,
        "unsafe_bomb_penalty": -20.0,
        "useless_bomb_penalty": -0.3,
        "avoidable_fatal_action": -40.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.04,
        "history_loop_penalty_cap": 3,
        "history_wait_penalty_step": -0.04,
        "history_wait_penalty_cap": 3,
        "causal_bomb_death_penalty": -25.0,
        "resolved_bomb_survival_reward": 3.0,
        "resolved_bomb_crate_reward": 0.1,
    },
    "r14_no_safety_useful_bomb_credit": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.0,
        "killed_opponent": 5.0,
        "killed_self": -40.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "potential_survival_options_weight": 2.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "useless_bomb_penalty": -1.0,
        "avoidable_fatal_action": -40.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.04,
        "history_loop_penalty_cap": 3,
        "history_wait_penalty_step": -0.04,
        "history_wait_penalty_cap": 3,
        "causal_bomb_death_penalty": -25.0,
        "causal_bomb_success_base": 1.0,
        "causal_bomb_success_per_crate": 1.0,
        "resolved_bomb_survival_reward": 1.0,
        "resolved_bomb_crate_reward": 0.5,
        "resolved_bomb_requires_utility": 1.0,
    },
    "r15_no_safety_objective_credit": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.0,
        "killed_opponent": 5.0,
        "killed_self": -40.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 1.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "potential_survival_options_weight": 2.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "suppress_useful_bomb_when_coin_reachable": 1.0,
        "useless_bomb_penalty": -2.0,
        "avoidable_fatal_action": -40.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.2,
        "history_loop_penalty_cap": 5,
        "history_wait_penalty_step": -0.08,
        "history_wait_penalty_cap": 5,
        "causal_bomb_death_penalty": -25.0,
        "causal_bomb_success_base": 1.0,
        "causal_bomb_success_per_crate": 1.0,
        "causal_bomb_zero_utility_penalty": -2.0,
        "resolved_bomb_survival_reward": 1.0,
        "resolved_bomb_crate_reward": 0.5,
        "resolved_bomb_requires_utility": 1.0,
    },
    "r17_global_crate_bomb_discipline": {
        "step": -0.01,
        "coin_collected": 3.0,
        "crate_destroyed": 0.2,
        "killed_opponent": 5.0,
        "killed_self": -20.0,
        "got_killed": -10.0,
        "invalid_action": -0.1,
        "potential_gamma": 0.95,
        "potential_coin_weight": 0.5,
        "potential_crate_weight": 0.25,
        "potential_danger_weight": 1.0,
        "unsafe_bomb_penalty": -20.0,
        "useful_bomb_per_crate": 0.2,
        "useful_bomb_crate_cap": 3,
        "useless_bomb_penalty": -1.0,
        "causal_bomb_success_base": 0.5,
        "causal_bomb_success_per_crate": 0.5,
        "causal_bomb_zero_utility_penalty": -1.0,
        "avoidable_fatal_action": -20.0,
        "suicide_dominates_positive_events": 1.0,
        "history_loop_penalty_step": -0.04,
        "history_loop_penalty_cap": 3,
        "history_wait_penalty_step": -0.04,
        "history_wait_penalty_cap": 3,
    },
}
# Final no-safety reward contract.  Older r11-r15 IDs remain registered only
# so their checkpoints stay reproducible; new no-safety runs use this copy.
REWARD_SPECS["r16_no_safety_locked"] = dict(
    REWARD_SPECS["r15_no_safety_objective_credit"])
REWARD_SPECS["r18_wait_attractor_escape"] = {
    **REWARD_SPECS["r17_global_crate_bomb_discipline"],
    "history_wait_penalty_step": -0.2,
    "history_wait_penalty_cap": 5,
    "useful_bomb_counts_as_wait_progress": 1.0,
}
# Same learned reward signal as r12, with a distinct ID so the shared linear
# runtime does not apply r12's rule-based repeated-cycle action elimination.
REWARD_SPECS["r12_coin_priority_anti_loop_reward_only"] = dict(
    REWARD_SPECS["r12_coin_priority_anti_loop"])
REWARD_SPECS['r9_task3_score_aligned'] = {
    **REWARD_SPECS['r7_safe_credit_sparse'], 'killed_opponent': 15.0,
}
DEATH_EVENTS = frozenset((e.KILLED_SELF, e.GOT_KILLED))


REWARD_SPECS[TASK4_SCORE_REWARD] = {
    'step': 0.0, 'coin_collected': 1.0, 'killed_opponent': 5.0,
    'crate_destroyed': 0.0, 'death': 0.0, 'invalid_action': 0.0,
}


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

    names = (
        (e.COIN_COLLECTED, "coin_collected"),
        (e.COIN_FOUND, "coin_found"),
        (e.KILLED_OPPONENT, "killed_opponent"),
        (e.CRATE_DESTROYED, "crate_destroyed"),
        (e.INVALID_ACTION, "invalid_action"),
        (e.SURVIVED_ROUND, "survived_round"),
    )
    event_rewards = {
        event: spec[key] for event, key in names if key in spec
    }
    reward = spec["step"]
    suppress_positive = (
        e.KILLED_SELF in events
        and bool(spec.get("suicide_dominates_positive_events", 0.0))
    )
    for event, event_reward in event_rewards.items():
        if not (suppress_positive and event_reward > 0.0):
            reward += events.count(event) * event_reward
    if e.KILLED_SELF in events and "killed_self" in spec:
        reward += spec["killed_self"]
    elif e.GOT_KILLED in events and "got_killed" in spec:
        reward += spec["got_killed"]
    return float(reward)


def _phase_event_reward(
    events: Sequence[str], spec: dict[str, float], phase: dict[str, float],
) -> float:
    """Score real events using the observable phase before the action."""
    early = float(phase["early_weight"])
    middle = float(phase["middle_weight"])
    late = float(phase["late_weight"])
    values = {
        e.COIN_COLLECTED: (
            early * spec["coin_early"] + middle * spec["coin_middle"]
            + late * spec["coin_late"]),
        e.CRATE_DESTROYED: (
            early * spec["crate_early"] + middle * spec["crate_middle"]
            + late * spec["crate_late"]),
        e.KILLED_OPPONENT: (
            early * spec["kill_early"] + middle * spec["kill_middle"]
            + late * spec["kill_late"]),
        e.INVALID_ACTION: spec["invalid_action"],
    }
    suicide = e.KILLED_SELF in events
    reward = spec["step"]
    for event, value in values.items():
        if not (suicide and value > 0.0):
            reward += events.count(event) * value
    if suicide:
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
        value += spec.get("potential_coin_weight", 0.0) * exp(-coin_distance / 4.0)
    else:
        crate_distances = distance_to_targets(
            blocked, crate_frontiers(game_state, blocked))
        crate_distance = float(crate_distances[position])
        if np.isfinite(crate_distance):
            value += spec.get("potential_crate_weight", 0.0) * exp(-crate_distance / 4.0)

    danger = predict_danger(game_state, horizon=HORIZON).danger
    dangerous_steps = np.flatnonzero(danger[1:, position[0], position[1]])
    earliest = HORIZON + 1 if dangerous_steps.size == 0 else int(dangerous_steps[0]) + 1
    if "potential_danger_weight" in spec:
        danger_value = (
            0.0 if earliest > HORIZON
            else -(HORIZON + 1 - earliest) / HORIZON
        )
        value += spec["potential_danger_weight"] * danger_value
    elif "potential_safety_weight" in spec:
        safety = 1.0 if earliest > HORIZON else max(0.0, earliest - 1) / HORIZON
        value += spec["potential_safety_weight"] * safety
    if "potential_survival_options_weight" in spec:
        from agent_code.team_agent.feature_system.common import ACTIONS, build_context

        context = build_context(game_state)
        survivable = 0
        legal = 0
        for index, action in enumerate(ACTIONS):
            if not bool(context.legal_mask[index]):
                continue
            legal += 1
            reachability = (
                context.bomb_reachability
                if action == "BOMB" else context.movement_reachability[action]
            )
            survivable += int(bool(
                reachability is not None and reachability.survives_horizon))
        fraction = survivable / float(legal) if legal else 0.0
        value += spec["potential_survival_options_weight"] * fraction
    return float(value)


def _phase_potential(
    game_state: dict, spec: dict[str, float], phase: dict[str, float],
) -> float:
    """Bounded state-only phase shaping; it never emits a preferred action."""
    from agent_code.team_agent.phase import objective_closeness

    value = _state_potential(game_state, spec)
    early = float(phase["early_weight"])
    middle = float(phase["middle_weight"])
    late = float(phase["late_weight"])
    score_margin = float(phase["score_margin"])
    self_mobility = float(phase["self_mobility"])
    if spec.get("phase_resource_potential", 0.0):
        coin_closeness, crate_closeness = objective_closeness(game_state)
        value += early * (0.5 * coin_closeness + 0.25 * crate_closeness)
    if spec.get("phase_combat_potential", 0.0):
        value += (
            0.75 * (middle + 0.5 * late) * (1.0 - 0.25 * score_margin)
            * self_mobility
            * (0.5 * float(phase["opponent_closeness"])
               + 0.5 * (1.0 - float(phase["opponent_mobility"])))
        )
    if spec.get("phase_mobility_potential", 0.0):
        value += (
            late * (1.0 + 0.25 * score_margin)
            * (0.5 * float(phase["safe_action_fraction"])
               + 0.5 * self_mobility)
        )
    return float(value)


def _bomb_action_reward(game_state: dict, action: str | None, spec: dict[str, float]) -> float:
    if action != "BOMB" or "unsafe_bomb_penalty" not in spec:
        return 0.0
    from agent_code.team_agent.feature_system.common import build_context

    context = build_context(game_state)
    bomb = context.bomb_reachability
    if bomb is None or not bomb.survives_horizon:
        return float(spec["unsafe_bomb_penalty"])
    if (
        context.crates_in_blast == 0
        and context.opponents_in_blast == 0
        and "useless_bomb_penalty" in spec
    ):
        return float(spec["useless_bomb_penalty"])
    if (
        context.reachable_coin_exists
        and bool(spec.get("suppress_useful_bomb_when_coin_reachable", 0.0))
    ):
        return 0.0
    if "useful_bomb_per_crate" not in spec:
        return 0.0
    useful_crates = min(
        int(context.crates_in_blast), int(spec["useful_bomb_crate_cap"]))
    return float(spec["useful_bomb_per_crate"] * useful_crates)


def reward_from_events(
    events: Sequence[str],
    version: str = REWARD_VERSION,
    *,
    old_game_state: dict | None = None,
    new_game_state: dict | None = None,
    terminal: bool = False,
    repeated_oscillation: bool = False,
    idle_streak: int = 0,
    action: str | None = None,
    conditional_loop: bool = False,
    avoidable_wait: bool = False,
    loop_length: int = 0,
    loop_repeat_count: int = 0,
    avoidable_wait_streak: int = 0,
    diagnostic: dict | None = None,
    avoidable_fatal: bool = False,
    old_phase_facts: dict[str, float] | None = None,
    new_phase_facts: dict[str, float] | None = None,
    temporal_adjustment: float = 0.0,
    bomb_resolved_alive: bool = False,
    resolved_bomb_crates: int = 0,
) -> float:
    """Convert framework events and optional temporal context into a scalar."""
    spec = resolve_reward_spec(version)
    if version == TASK4_SCORE_REWARD:
        return float(events.count(e.COIN_COLLECTED) + 5 * events.count(e.KILLED_OPPONENT))
    if conditional_loop and avoidable_wait:
        raise ValueError(
            "conditional loop and avoidable WAIT penalties are mutually exclusive")
    phase_reward = version.startswith("r9_phase_")
    if phase_reward and old_phase_facts is None:
        raise ValueError(f"{version} requires old_phase_facts")
    reward = (
        _phase_event_reward(events, spec, old_phase_facts)
        if phase_reward else _event_reward(events, spec)
    )
    if "potential_gamma" in spec and old_game_state is not None:
        old_potential = (
            _phase_potential(old_game_state, spec, old_phase_facts)
            if phase_reward else _state_potential(old_game_state, spec)
        )
        next_potential = (
            0.0 if terminal or new_game_state is None
            else (
                _phase_potential(new_game_state, spec, new_phase_facts)
                if phase_reward else _state_potential(new_game_state, spec)
            )
        )
        reward += spec["potential_gamma"] * next_potential - old_potential
    if old_game_state is not None:
        reward += _bomb_action_reward(old_game_state, action, spec)
    if avoidable_fatal and "avoidable_fatal_action" in spec:
        reward += spec["avoidable_fatal_action"]
    reward += float(temporal_adjustment)
    if version == "r4_anti_oscillation" and repeated_oscillation:
        reward += spec["oscillation_penalty"]
    if version == "r4_anti_oscillation" and idle_streak >= 2:
        multiplier = min(idle_streak - 1, int(spec["idle_penalty_cap"]))
        reward += spec["idle_penalty_step"] * multiplier
    if conditional_loop and "conditional_loop_penalty" in spec:
        reward += spec["conditional_loop_penalty"]
    if avoidable_wait and "avoidable_wait_penalty" in spec:
        reward += spec["avoidable_wait_penalty"]
    if loop_repeat_count >= 2 and "history_loop_penalty_step" in spec:
        multiplier = min(
            loop_repeat_count - 1, int(spec["history_loop_penalty_cap"]))
        reward += spec["history_loop_penalty_step"] * multiplier
    if avoidable_wait_streak >= 2 and "history_wait_penalty_step" in spec:
        multiplier = min(
            avoidable_wait_streak - 1, int(spec["history_wait_penalty_cap"]))
        reward += spec["history_wait_penalty_step"] * multiplier
    return float(reward)

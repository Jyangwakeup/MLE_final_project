"""Shared history-dependent reward context for every learning agent."""

from __future__ import annotations

import numpy as np
import events as e

from agent_code.team_agent.danger import predict_danger
from agent_code.team_agent.feature_system.common import (
    MOVE_ACTIONS, action_destination, build_context, crate_frontiers,
    distance_to_targets, navigation_blocked,
)
from agent_code.team_agent.feature_system.discrete_q_v2 import nearest_coin_distance

OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
MOVED_EVENT = {
    "UP": e.MOVED_UP, "RIGHT": e.MOVED_RIGHT,
    "DOWN": e.MOVED_DOWN, "LEFT": e.MOVED_LEFT,
}


def init_temporal_reward_state(owner) -> None:
    owner.previous_action = None
    owner.move_history = []
    owner.stationary_streak = 0
    owner.reward_previous_position = None
    owner.reward_previous_coin_target = None
    owner.reward_position_history = []
    owner.reward_avoidable_wait_streak = 0
    owner.reward_bomb_pending = False
    owner.reward_bomb_crates = 0


def prepare_frozen_temporal_state(owner, game_state) -> None:
    """Reset inference-only history before the first decision of each round."""
    round_index = game_state.get("round")
    if getattr(owner, "frozen_temporal_round", None) != round_index:
        init_temporal_reward_state(owner)
        owner.frozen_temporal_round = round_index


def advance_frozen_temporal_state(owner, action, game_state) -> None:
    """Advance history fields normally maintained by training callbacks."""
    target, _ = _reachable_coin_target(game_state)
    owner.previous_action = action
    owner.reward_previous_position = game_state["self"][3]
    owner.reward_previous_coin_target = target


DIAGNOSTIC_COUNT_FIELDS = (
    "conditional_loop_count", "loop_not_move_count", "loop_no_movement_count",
    "loop_current_danger_count", "loop_no_reachable_coin_count",
    "loop_target_changed_count", "loop_not_returned_count",
    "avoidable_wait_count", "wait_current_danger_count", "wait_next_danger_count",
    "wait_no_reachable_coin_count", "wait_no_safe_progress_move_count",
)


def reset_reward_diagnostics(owner) -> None:
    owner.reward_diagnostics = {field: 0 for field in DIAGNOSTIC_COUNT_FIELDS}
    owner.reward_diagnostics.update({
        "conditional_loop_reward": 0.0, "avoidable_wait_reward": 0.0,
    })


def accumulate_reward_diagnostics(owner, diagnostic, reward_spec) -> None:
    if not hasattr(owner, "reward_diagnostics"):
        reset_reward_diagnostics(owner)
    for field in DIAGNOSTIC_COUNT_FIELDS:
        owner.reward_diagnostics[field] += int(bool(diagnostic.get(field)))
    if diagnostic.get("conditional_loop_count"):
        owner.reward_diagnostics["conditional_loop_reward"] += float(
            reward_spec["conditional_loop_penalty"])
    if diagnostic.get("avoidable_wait_count"):
        owner.reward_diagnostics["avoidable_wait_reward"] += float(
            reward_spec["avoidable_wait_penalty"])


def _reachable_coin_target(game_state):
    """Return the deterministic nearest reachable coin and its distance map."""
    blocked = navigation_blocked(game_state)
    position = game_state["self"][3]
    from_position = distance_to_targets(blocked, (position,))
    reachable = [
        (float(from_position[target]), target)
        for target in sorted(set(game_state["coins"]))
        if np.isfinite(float(from_position[target]))
    ]
    if not reachable:
        return None, None
    _, target = min(reachable, key=lambda item: (item[0], item[1]))
    return target, distance_to_targets(blocked, (target,))


def _reachable_navigation_target(game_state):
    """Prefer a reachable coin, otherwise the nearest crate frontier."""
    target, distances = _reachable_coin_target(game_state)
    if target is not None:
        return target, distances
    blocked = navigation_blocked(game_state)
    position = game_state["self"][3]
    from_position = distance_to_targets(blocked, (position,))
    reachable = [
        (float(from_position[target]), target)
        for target in crate_frontiers(game_state, blocked)
        if np.isfinite(float(from_position[target]))
    ]
    if not reachable:
        return None, None
    _, target = min(reachable, key=lambda item: (item[0], item[1]))
    return target, distance_to_targets(blocked, (target,))


def observed_terminal_state(old_state, action, events):
    """Reconstruct the post-action position needed by terminal diagnostics."""
    state = dict(old_state)
    self_record = list(old_state["self"])
    if action in MOVE_ACTIONS and MOVED_EVENT[action] in events:
        self_record[3] = action_destination(old_state["self"][3], action)
    state["self"] = tuple(self_record)
    return state


def _conditional_navigation_flags(
    owner, action, old_state, new_state, *, useful_bomb_progress=False,
):
    """Evaluate the narrow Task 1 anti-loop and avoidable-WAIT predicates."""
    position = old_state["self"][3]
    target, target_distances = _reachable_navigation_target(old_state)
    danger = predict_danger(old_state).danger
    current_safe = not bool(danger[1, position[0], position[1]])

    is_move = action in MOVE_ACTIONS
    actually_moved = (
        new_state is not None and new_state["self"][3] != position)
    returned_to_previous_position = (
        is_move and actually_moved
        and new_state is not None
        and getattr(owner, "reward_previous_position", None) is not None
        and new_state["self"][3] == owner.reward_previous_position
    )
    target_unchanged = (
        target is not None
        and target == getattr(owner, "reward_previous_coin_target", None)
    )
    conditional_loop = bool(
        current_safe and target_unchanged and returned_to_previous_position)

    diagnostic = {
        "conditional_loop_count": conditional_loop,
        "loop_not_move_count": not is_move,
        "loop_no_movement_count": is_move and not actually_moved,
        "loop_current_danger_count": is_move and not current_safe,
        "loop_no_reachable_coin_count": is_move and target is None,
        "loop_target_changed_count": is_move and target is not None and not target_unchanged,
        "loop_not_returned_count": (
            is_move and actually_moved and current_safe and target_unchanged
            and not returned_to_previous_position),
    }

    avoidable_wait = False
    wait_selected = action == "WAIT"
    next_safe = False
    has_safe_progress_move = False
    if wait_selected and target_distances is not None and current_safe:
        next_safe = not bool(danger[2, position[0], position[1]])
        current_distance = float(target_distances[position])
        if next_safe:
            context = build_context(old_state)
            for move_index, move in enumerate(MOVE_ACTIONS):
                destination = action_destination(position, move)
                if (
                    context.legal_mask[move_index]
                    and context.movement_reachability[move].survives_horizon
                    and float(target_distances[destination]) < current_distance
                ):
                    has_safe_progress_move = True
                    break
            if not has_safe_progress_move and useful_bomb_progress:
                bomb = context.bomb_reachability
                has_safe_progress_move = bool(
                    context.legal_mask[5]
                    and bomb is not None
                    and bomb.survives_horizon
                    and (context.crates_in_blast > 0 or context.opponents_in_blast > 0)
                )
            avoidable_wait = has_safe_progress_move

    diagnostic.update({
        "avoidable_wait_count": avoidable_wait,
        "wait_current_danger_count": wait_selected and not current_safe,
        "wait_next_danger_count": wait_selected and current_safe and not next_safe,
        "wait_no_reachable_coin_count": (
            wait_selected and current_safe and next_safe and target is None),
        "wait_no_safe_progress_move_count": (
            wait_selected and current_safe and next_safe and target is not None
            and not has_safe_progress_move),
    })

    owner.reward_previous_position = position
    owner.reward_previous_coin_target = target
    return conditional_loop, avoidable_wait, diagnostic


def _bounded_history_flags(
    owner, action, old_state, new_state, events, *, useful_bomb_progress=False,
):
    """Return v4 anti-loop counts while excluding progress and forced danger."""
    from agent_code.team_agent.feature_system.continuous_v4 import _cycle_detail

    position = tuple(old_state["self"][3])
    target, _ = _reachable_navigation_target(old_state)
    previous_target = getattr(owner, "reward_previous_coin_target", None)
    target_changed = previous_target is not None and target != previous_target
    danger = predict_danger(old_state).danger
    threatened = bool(danger[1:, position[0], position[1]].any())
    made_progress = e.COIN_COLLECTED in events or e.CRATE_DESTROYED in events

    conditional_loop, avoidable_wait, diagnostic = _conditional_navigation_flags(
        owner, action, old_state, new_state,
        useful_bomb_progress=useful_bomb_progress)
    wait_streak = (
        int(getattr(owner, "reward_avoidable_wait_streak", 0)) + 1
        if avoidable_wait else 0
    )
    owner.reward_avoidable_wait_streak = wait_streak

    history = [tuple(value) for value in getattr(
        owner, "reward_position_history", ())]
    if not history or history[-1] != position:
        history.append(position)
    loop_length = loop_repeats = 0
    actually_moved = (
        action in MOVE_ACTIONS and new_state is not None
        and tuple(new_state["self"][3]) != position
    )
    if made_progress or target_changed or threatened:
        history = [position]
    elif actually_moved:
        _, loop_length, loop_repeats = _cycle_detail(
            history, tuple(new_state["self"][3]))
    if new_state is not None:
        next_position = tuple(new_state["self"][3])
        if not history or history[-1] != next_position:
            history.append(next_position)
    owner.reward_position_history = history[-33:]
    diagnostic.update({
        "history_loop_count": loop_repeats >= 2,
        "history_wait_count": wait_streak >= 2,
    })
    return loop_length, loop_repeats, wait_streak, diagnostic


def temporal_reward_context(
    owner, action, old_state, new_state, events, reward_id=None,
) -> dict:
    if not hasattr(owner, "move_history"):
        init_temporal_reward_state(owner)
    history = owner.move_history
    oscillating = (
        action in OPPOSITE and len(history) >= 2
        and history[-1] == OPPOSITE[action] and history[-2] == action
        and new_state is not None and e.COIN_COLLECTED not in events
        and nearest_coin_distance(new_state) >= nearest_coin_distance(old_state)
    )
    idle_streak = 0
    if (
        action == "WAIT" and new_state is not None
        and e.COIN_COLLECTED not in events
        and old_state["self"][3] == new_state["self"][3]
    ):
        position = old_state["self"][3]
        dangerous = predict_danger(old_state).danger[
            1:, position[0], position[1]].any()
        distance = nearest_coin_distance(old_state)
        has_objective = (
            np.isfinite(distance) or bool(old_state["others"])
            or bool(np.any(old_state["field"] == 1))
        )
        if not dangerous and has_objective:
            idle_streak = int(owner.stationary_streak) + 1

    if action in OPPOSITE:
        history.append(action)
        history[:] = history[-2:]
    else:
        history.clear()
    owner.previous_action = action
    owner.stationary_streak = idle_streak
    conditional_loop = avoidable_wait = False
    loop_length = loop_repeat_count = avoidable_wait_streak = 0
    diagnostic = {}
    from agent_code.team_agent.rewards import resolve_reward_spec
    reward_spec = resolve_reward_spec(reward_id) if reward_id is not None else {}
    temporal_adjustment = 0.0
    bomb_resolved_alive = False
    resolved_bomb_crates = 0
    if "resolved_bomb_survival_reward" in reward_spec:
        if e.BOMB_DROPPED in events:
            owner.reward_bomb_pending = True
            owner.reward_bomb_crates = 0
        if getattr(owner, "reward_bomb_pending", False):
            owner.reward_bomb_crates += events.count(e.CRATE_DESTROYED)
            bomb_resolved_alive = bool(
                e.KILLED_SELF not in events
                and new_state is not None
                and bool(new_state["self"][2])
                and e.BOMB_DROPPED not in events
            )
            if bomb_resolved_alive:
                resolved_bomb_crates = int(owner.reward_bomb_crates)
                qualifies = (
                    resolved_bomb_crates > 0
                    or not bool(reward_spec.get(
                        "resolved_bomb_requires_utility", 0.0))
                )
                if qualifies:
                    temporal_adjustment += float(
                        reward_spec["resolved_bomb_survival_reward"])
                    temporal_adjustment += (
                        float(reward_spec["resolved_bomb_crate_reward"])
                        * resolved_bomb_crates
                    )
                owner.reward_bomb_pending = False
                owner.reward_bomb_crates = 0
    if {
        "history_loop_penalty_step", "history_wait_penalty_step",
    }.intersection(reward_spec):
        (
            loop_length, loop_repeat_count, avoidable_wait_streak, diagnostic,
        ) = _bounded_history_flags(
            owner, action, old_state, new_state, events,
            useful_bomb_progress=bool(reward_spec.get(
                "useful_bomb_counts_as_wait_progress", 0.0)))
        conditional_loop = bool(loop_repeat_count >= 2)
        avoidable_wait = bool(avoidable_wait_streak >= 2)
    elif {
        "conditional_loop_penalty", "avoidable_wait_penalty",
    }.intersection(reward_spec):
        conditional_loop, avoidable_wait, diagnostic = _conditional_navigation_flags(
            owner, action, old_state, new_state)
    return {
        "repeated_oscillation": bool(oscillating),
        "idle_streak": idle_streak,
        "conditional_loop": conditional_loop,
        "avoidable_wait": avoidable_wait,
        "loop_length": loop_length,
        "loop_repeat_count": loop_repeat_count,
        "avoidable_wait_streak": avoidable_wait_streak,
        "diagnostic": diagnostic,
        "temporal_adjustment": temporal_adjustment,
        "bomb_resolved_alive": bomb_resolved_alive,
        "resolved_bomb_crates": resolved_bomb_crates,
    }


def reset_temporal_reward_state(owner) -> None:
    init_temporal_reward_state(owner)

"""Shared history-dependent reward context for every learning agent."""

from __future__ import annotations

import numpy as np
import events as e

from agent_code.team_agent.danger import predict_danger
from agent_code.team_agent.feature_system.discrete_q_v2 import nearest_coin_distance

OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def init_temporal_reward_state(owner) -> None:
    owner.previous_action = None
    owner.move_history = []
    owner.stationary_streak = 0


def temporal_reward_context(owner, action, old_state, new_state, events) -> dict:
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
    return {
        "repeated_oscillation": bool(oscillating),
        "idle_streak": idle_streak,
    }


def reset_temporal_reward_state(owner) -> None:
    init_temporal_reward_state(owner)

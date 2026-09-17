"""Agent-local Task-2 shaping facts; never used to override inference actions."""

from __future__ import annotations

import numpy as np

from agent_code.team_agent.feature_system.common import (
    ACTIONS, MOVE_ACTIONS, action_destination, build_context,
)


def avoidable_task2_wait(game_state, action, safe_mask):
    """Return whether WAIT ignored a safe objective-progress action and why."""
    result = {
        "avoidable": False,
        "safe_progress_move": False,
        "safe_useful_bomb": False,
    }
    if game_state is None or action != "WAIT":
        return result
    safe = np.asarray(safe_mask, dtype=bool)
    if safe.shape != (len(ACTIONS),) or not safe[ACTIONS.index("WAIT")]:
        return result
    context = build_context(game_state)
    position = tuple(game_state["self"][3])
    objective_maps = (context.coin_distance, context.crate_frontier_distance)
    for index, move in enumerate(MOVE_ACTIONS):
        if not safe[index]:
            continue
        destination = action_destination(position, move)
        for distances in objective_maps:
            current = float(distances[position])
            candidate = float(distances[destination])
            if np.isfinite(candidate) and (
                    not np.isfinite(current) or candidate < current):
                result["safe_progress_move"] = True
                break
        if result["safe_progress_move"]:
            break
    bomb_index = ACTIONS.index("BOMB")
    result["safe_useful_bomb"] = bool(
        safe[bomb_index] and context.crates_in_blast > 0)
    result["avoidable"] = bool(
        result["safe_progress_move"] or result["safe_useful_bomb"])
    return result

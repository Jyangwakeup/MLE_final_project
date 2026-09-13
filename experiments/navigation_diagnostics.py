"""Read-only coin-navigation diagnostics for frozen evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_code.team_agent.feature_system.common import distance_to_targets


MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
OPPOSITE_ACTION = {
    "UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT",
}


def navigation_diagnostic(
    game_state: dict[str, Any],
    action: str,
    *,
    previous_target: tuple[int, int] | None = None,
    previous_action: str | None = None,
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """Describe one chosen action without changing policy inputs or RNG state.

    The post-action distance is a static counterfactual on the state that was
    supplied to ``act``. It deliberately runs only after the action has been
    returned, and it never replaces or filters that action.
    """
    position = tuple(int(value) for value in game_state["self"][3])
    field = game_state["field"]
    blocked = np.asarray(field != 0, dtype=bool).copy()
    for bomb_position, _ in game_state.get("bombs", ()):
        blocked[tuple(int(value) for value in bomb_position)] = True
    for other in game_state.get("others", ()):
        blocked[tuple(int(value) for value in other[3])] = True
    blocked[position] = False

    from_position = distance_to_targets(blocked, (position,))
    reachable = [
        (
            float(from_position[tuple(coin)]),
            tuple(int(value) for value in coin),
        )
        for coin in game_state.get("coins", ())
        if np.isfinite(from_position[tuple(coin)])
    ]
    reachable.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    if reachable:
        distance_before = reachable[0][0]
        nearest_targets = [
            target for distance, target in reachable if distance == distance_before
        ]
        target = nearest_targets[0]
    else:
        distance_before = None
        nearest_targets = []
        target = None

    candidate = position
    movement_legal = action not in MOVE_DELTAS
    if action in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[action]
        proposed = position[0] + dx, position[1] + dy
        movement_legal = (
            0 <= proposed[0] < blocked.shape[0]
            and 0 <= proposed[1] < blocked.shape[1]
            and not blocked[proposed]
        )
        if movement_legal:
            candidate = proposed

    distance_after = None
    if reachable and movement_legal:
        from_candidate = distance_to_targets(blocked, (candidate,))
        candidate_distances = [
            float(from_candidate[tuple(coin)])
            for coin in game_state.get("coins", ())
            if np.isfinite(from_candidate[tuple(coin)])
        ]
        if candidate_distances:
            distance_after = min(candidate_distances)

    comparable = distance_before is not None and distance_after is not None
    previous_still_available = (
        previous_target is not None
        and previous_target in {
            tuple(int(value) for value in coin)
            for coin in game_state.get("coins", ())
        }
    )
    record = {
        "coin_count_before": len(game_state.get("coins", ())),
        "nearest_coin_distance_before": distance_before,
        "predicted_nearest_coin_distance_after": distance_after,
        "distance_comparable": comparable,
        "distance_reduced": bool(
            comparable and distance_after < distance_before
        ),
        "multiple_nearest_coins": len(nearest_targets) > 1,
        "selected_target": None if target is None else list(target),
        "previous_target_still_available": bool(previous_still_available),
        "target_switched_while_previous_available": bool(
            previous_still_available and target != previous_target
        ),
        "waited": action == "WAIT",
        "immediate_reverse": bool(
            action in OPPOSITE_ACTION
            and previous_action == OPPOSITE_ACTION[action]
        ),
        "movement_legal_in_observed_state": bool(movement_legal),
    }
    return record, target

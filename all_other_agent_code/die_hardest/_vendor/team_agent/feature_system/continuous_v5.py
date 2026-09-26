"""Continuous-v4 plus global crate-region and post-bomb objective features."""

from __future__ import annotations

import numpy as np

import settings as s
from . import continuous_v4
from .common import (
    ACTIONS, MOVE_ACTIONS, action_destination, crate_frontiers,
    distance_to_targets, navigation_blocked,
)
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v5"
CRATE_FIELDS = tuple(
    f"{action.lower()}_{field}"
    for action in ACTIONS
    for field in ("crate_half_plane_density", "bomb_objective_distance_delta")
)
GLOBAL_FIELDS = ("crate_fraction", "bomb_objective_active")
VECTOR_FIELDS = continuous_v4.VECTOR_FIELDS + CRATE_FIELDS + GLOBAL_FIELDS
FEATURE_DIM = 140


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v4.schema().normalization,
            "crate_half_plane_density": "crates in the action-facing half-plane / all crates",
            "bomb_objective_distance_delta": "clip((before-after)/(width*height-1),-1,1)",
            "crate_fraction": "remaining crates/(width*height)",
            "bomb_objective_active": "binary retained pre-bomb crate-frontier objective",
        },
    )


def selected_crate_target(game_state: dict) -> tuple[int, int] | None:
    """Choose a deterministic reachable crate frontier for target persistence."""
    blocked = navigation_blocked(game_state)
    position = tuple(game_state["self"][3])
    targets = crate_frontiers(game_state, blocked)
    distances = distance_to_targets(blocked, (position,))
    reachable = [
        (float(distances[target]), target)
        for target in targets if np.isfinite(float(distances[target]))
    ]
    return None if not reachable else min(reachable, key=lambda item: (item[0], item[1]))[1]


def _distance_delta(distances, position, destination, area):
    before, after = float(distances[position]), float(distances[destination])
    if not np.isfinite(before) or not np.isfinite(after):
        return 0.0
    return float(np.clip((before - after) / max(1.0, area - 1.0), -1.0, 1.0))


def extract(
    game_state: dict,
    previous_action=None,
    wait_streak=0,
    own_bomb=None,
    previous_position=None,
    previous_coin_target=None,
    position_history=(),
    bomb_objective=None,
) -> VectorFeatures | None:
    if game_state is None:
        return None
    base = continuous_v4.extract(
        game_state, previous_action=previous_action, wait_streak=wait_streak,
        own_bomb=own_bomb, previous_position=previous_position,
        previous_coin_target=previous_coin_target,
        position_history=position_history,
    )
    field = game_state["field"]
    position = tuple(game_state["self"][3])
    crates = [tuple(map(int, item)) for item in np.argwhere(field == 1)]
    total = len(crates)
    blocked = navigation_blocked(game_state)
    objective = None if bomb_objective is None else tuple(bomb_objective)
    objective_distances = (
        None if objective is None
        else distance_to_targets(blocked, (objective,))
    )
    area = float(field.shape[0] * field.shape[1])
    values = []
    directions = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
    for action in ACTIONS:
        if action not in MOVE_ACTIONS:
            values.extend((0.0, 0.0))
            continue
        dx, dy = directions[action]
        facing = sum(
            1 for x, y in crates
            if (x - position[0]) * dx + (y - position[1]) * dy > 0
        )
        destination = action_destination(position, action)
        objective_delta = 0.0
        if (
            objective_distances is not None
            and 0 <= destination[0] < field.shape[0]
            and 0 <= destination[1] < field.shape[1]
            and field[destination] == 0
        ):
            objective_delta = _distance_delta(
                objective_distances, position, destination, area)
        values.extend((facing / float(total) if total else 0.0, objective_delta))
    values.extend((total / area, float(objective is not None)))
    vector = np.concatenate((base.vector, np.asarray(values, dtype=np.float32)))
    return VectorFeatures(
        FEATURE_ID, vector.astype(np.float32, copy=False),
        base.legal_mask.copy(), base.context)

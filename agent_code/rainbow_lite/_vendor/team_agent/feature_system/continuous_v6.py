"""Continuous-v5 plus persistent opponent tracking and pressure features."""

from __future__ import annotations

import numpy as np

import settings as s
from ..danger import blast_coords
from . import continuous_v5
from .common import (
    ACTIONS, MOVE_ACTIONS, DIRECTIONS, action_destination, distance_to_targets,
    in_bounds, navigation_blocked,
)
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v6-opponent-tracking"
TRACKING_FIELDS = tuple(
    f"{action.lower()}_{field}"
    for action in ACTIONS
    for field in ("tracked_opponent_distance_delta", "tracked_opponent_blast_line")
)
GLOBAL_FIELDS = (
    "tracked_opponent_exists", "tracked_opponent_persisted",
    "tracked_opponent_distance", "tracked_opponent_relative_x",
    "tracked_opponent_relative_y", "tracked_opponent_motion_x",
    "tracked_opponent_motion_y", "tracked_opponent_escape_fraction",
)
VECTOR_FIELDS = continuous_v5.VECTOR_FIELDS + TRACKING_FIELDS + GLOBAL_FIELDS
FEATURE_DIM = 160


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v5.schema().normalization,
            "tracked_opponent_distance_delta": "clip((before-after)/(width*height-1),-1,1)",
            "tracked_opponent_blast_line": "binary line-of-fire from post-action tile",
            "tracked_opponent_exists": "binary",
            "tracked_opponent_persisted": "binary same named target as previous state",
            "tracked_opponent_distance": "reachable distance/(width*height-1), else 1",
            "tracked_opponent_relative_x": "relative x/(width-1)",
            "tracked_opponent_relative_y": "relative y/(height-1)",
            "tracked_opponent_motion_x": "one-step target dx/(width-1)",
            "tracked_opponent_motion_y": "one-step target dy/(height-1)",
            "tracked_opponent_escape_fraction": "physical free neighbours plus WAIT / 5",
        },
    )


def _blast_line(field, origin, target):
    return float(target in set(blast_coords(field, origin, s.BOMB_POWER)))


def extract(
    game_state: dict,
    previous_action=None,
    wait_streak=0,
    own_bomb=None,
    previous_position=None,
    previous_coin_target=None,
    position_history=(),
    bomb_objective=None,
    tracked_opponent=None,
    tracked_opponent_persisted=False,
    previous_opponent_position=None,
) -> VectorFeatures | None:
    if game_state is None:
        return None
    base = continuous_v5.extract(
        game_state, previous_action=previous_action, wait_streak=wait_streak,
        own_bomb=own_bomb, previous_position=previous_position,
        previous_coin_target=previous_coin_target, position_history=position_history,
        bomb_objective=bomb_objective,
    )
    field = game_state["field"]
    position = tuple(game_state["self"][3])
    area = float(field.shape[0] * field.shape[1])
    target = None if tracked_opponent is None else tuple(tracked_opponent)
    blocked = navigation_blocked(game_state)
    distances = None if target is None else distance_to_targets(
        blocked, (target,), allow_blocked_targets=True)
    before = np.inf if distances is None else float(distances[position])
    values = []
    for action in ACTIONS:
        destination = action_destination(position, action)
        delta = 0.0
        aligned = 0.0
        if target is not None and in_bounds(destination, field.shape):
            after = float(distances[destination])
            if np.isfinite(before) and np.isfinite(after):
                delta = float(np.clip((before - after) / max(1.0, area - 1.0), -1.0, 1.0))
            aligned = _blast_line(field, destination, target)
        values.extend((delta, aligned))

    if target is None:
        values.extend((0.0,) * len(GLOBAL_FIELDS))
    else:
        dx, dy = target[0] - position[0], target[1] - position[1]
        previous = target if previous_opponent_position is None else tuple(previous_opponent_position)
        motion_x, motion_y = target[0] - previous[0], target[1] - previous[1]
        occupied = {tuple(item[0]) for item in game_state["bombs"]}
        occupied.update(tuple(other[3]) for other in game_state["others"] if tuple(other[3]) != target)
        escape_count = 1
        for action in MOVE_ACTIONS:
            candidate = action_destination(target, action)
            escape_count += int(
                in_bounds(candidate, field.shape)
                and field[candidate] == 0 and candidate not in occupied)
        normalized_distance = 1.0 if not np.isfinite(before) else min(1.0, before / max(1.0, area - 1.0))
        values.extend((
            1.0, float(tracked_opponent_persisted), normalized_distance,
            dx / max(1.0, field.shape[0] - 1.0),
            dy / max(1.0, field.shape[1] - 1.0),
            motion_x / max(1.0, field.shape[0] - 1.0),
            motion_y / max(1.0, field.shape[1] - 1.0),
            escape_count / 5.0,
        ))
    vector = np.concatenate((base.vector, np.asarray(values, dtype=np.float32)))
    return VectorFeatures(FEATURE_ID, vector, base.legal_mask.copy(), base.context)

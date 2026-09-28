"""Continuous-v7 plus reachable crate-region objectives by Agent quadrant."""

from __future__ import annotations

import numpy as np

from . import continuous_v7
from .common import (
    ACTIONS, DIRECTIONS, MOVE_ACTIONS, distance_to_targets, in_bounds,
    navigation_blocked,
)
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v11-agent-quadrant-crate-objectives"
QUADRANTS = ("nw", "ne", "sw", "se")
QUADRANT_FIELDS = tuple(
    [f"reachable_crate_{quadrant}_share" for quadrant in QUADRANTS]
    + [f"nearest_crate_{quadrant}_distance" for quadrant in QUADRANTS]
)
VECTOR_FIELDS = continuous_v7.VECTOR_FIELDS + QUADRANT_FIELDS
FEATURE_DIM = continuous_v7.FEATURE_DIM + len(QUADRANT_FIELDS)


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v7.schema().normalization,
            "reachable_crate_quadrant_share": (
                "weighted reachable crates in quadrant / all reachable crates"
            ),
            "nearest_crate_quadrant_distance": (
                "nearest reachable crate-frontier BFS distance/(width*height-1); "
                "1 means no reachable crate in quadrant"
            ),
        },
    )


def _side_weights(delta: int) -> tuple[float, float]:
    if delta < 0:
        return 1.0, 0.0
    if delta > 0:
        return 0.0, 1.0
    return 0.5, 0.5


def _quadrant_weights(position, target) -> np.ndarray:
    west, east = _side_weights(int(target[0]) - int(position[0]))
    north, south = _side_weights(int(target[1]) - int(position[1]))
    return np.asarray(
        (west * north, east * north, west * south, east * south),
        dtype=np.float64,
    )


def quadrant_crate_objectives(game_state: dict) -> np.ndarray:
    """Return reachable-crate shares then nearest frontier distances by quadrant."""
    field = game_state["field"]
    position = tuple(game_state["self"][3])
    blocked = navigation_blocked(game_state)
    from_agent = distance_to_targets(blocked, (position,))
    counts = np.zeros(4, dtype=np.float64)
    nearest = np.full(4, np.inf, dtype=np.float64)

    for x, y in np.argwhere(field == 1):
        crate = int(x), int(y)
        frontier_distances = []
        for action in MOVE_ACTIONS:
            dx, dy = DIRECTIONS[action]
            frontier = crate[0] + dx, crate[1] + dy
            if in_bounds(frontier, field.shape) and not blocked[frontier]:
                distance = float(from_agent[frontier])
                if np.isfinite(distance):
                    frontier_distances.append(distance)
        if not frontier_distances:
            continue
        distance = min(frontier_distances)
        weights = _quadrant_weights(position, crate)
        counts += weights
        nearest[weights > 0.0] = np.minimum(nearest[weights > 0.0], distance)

    total = float(counts.sum())
    shares = counts / total if total else counts
    scale = max(1.0, float(field.size - 1))
    distances = np.where(np.isfinite(nearest), np.minimum(nearest / scale, 1.0), 1.0)
    return np.concatenate((shares, distances)).astype(np.float32)


def extract(game_state: dict, **kwargs) -> VectorFeatures | None:
    base = continuous_v7.extract(game_state, **kwargs)
    if base is None:
        return None
    vector = np.concatenate((base.vector, quadrant_crate_objectives(game_state)))
    return VectorFeatures(
        FEATURE_ID, vector.astype(np.float32, copy=False),
        base.legal_mask.copy(), base.context,
    )

"""Continuous-v7 plus Agent-centred quadrant object densities."""

from __future__ import annotations

import numpy as np

from . import continuous_v7
from .common import ACTIONS
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v10-agent-quadrant-density"
QUADRANTS = ("nw", "ne", "sw", "se")
QUADRANT_FIELDS = tuple(
    f"{kind}_{quadrant}_density"
    for kind in ("crate", "opponent")
    for quadrant in QUADRANTS
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
            **{
                field: "weighted object count / weighted non-stone cells in [0,1]"
                for field in QUADRANT_FIELDS
            },
        },
    )


def _side_weights(delta: int) -> tuple[float, float]:
    if delta < 0:
        return 1.0, 0.0
    if delta > 0:
        return 0.0, 1.0
    return 0.5, 0.5


def quadrant_densities(game_state: dict) -> np.ndarray:
    """Return crate then opponent densities in NW, NE, SW, SE order."""
    field = game_state["field"]
    px, py = game_state["self"][3]
    capacity = np.zeros(4, dtype=np.float64)
    crates = np.zeros(4, dtype=np.float64)
    opponents = np.zeros(4, dtype=np.float64)

    def weights(x: int, y: int) -> np.ndarray:
        west, east = _side_weights(x - px)
        north, south = _side_weights(y - py)
        return np.asarray(
            (west * north, east * north, west * south, east * south),
            dtype=np.float64,
        )

    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == -1:
                continue
            contribution = weights(x, y)
            capacity += contribution
            if field[x, y] == 1:
                crates += contribution

    for other in game_state["others"]:
        x, y = other[3]
        opponents += weights(x, y)

    denominator = np.maximum(capacity, 1.0)
    return np.concatenate((crates / denominator, opponents / denominator)).astype(
        np.float32)


def extract(game_state: dict, **kwargs) -> VectorFeatures | None:
    base = continuous_v7.extract(game_state, **kwargs)
    if base is None:
        return None
    vector = np.concatenate((base.vector, quadrant_densities(game_state)))
    return VectorFeatures(
        FEATURE_ID, vector.astype(np.float32, copy=False),
        base.legal_mask.copy(), base.context,
    )

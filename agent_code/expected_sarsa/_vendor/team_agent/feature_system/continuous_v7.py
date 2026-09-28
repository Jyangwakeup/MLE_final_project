"""Continuous-v6 plus observable own-kill phase state."""

from __future__ import annotations

import numpy as np

from . import continuous_v6
from .types import FeatureSchema, VectorFeatures

FEATURE_ID = "continuous-v7-phase-aware"
VECTOR_FIELDS = continuous_v6.VECTOR_FIELDS + (
    "own_kills_fraction", "own_kill_achieved",
)
FEATURE_DIM = 162


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=continuous_v6.ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v6.schema().normalization,
            "own_kills_fraction": "own kills in current round, capped at 3, divided by 3",
            "own_kill_achieved": "binary own kills >= 1 in current round",
        },
    )


def extract(game_state: dict, *, own_kills: int = 0, **kwargs) -> VectorFeatures | None:
    base = continuous_v6.extract(game_state, **kwargs)
    if base is None:
        return None
    kills = max(0, min(int(own_kills), 3))
    vector = np.concatenate((
        base.vector,
        np.asarray((kills / 3.0, float(kills >= 1)), dtype=np.float32),
    ))
    return VectorFeatures(FEATURE_ID, vector, base.legal_mask.copy(), base.context)

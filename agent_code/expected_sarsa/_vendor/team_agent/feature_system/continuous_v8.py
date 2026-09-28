"""Continuous-v7 plus an explicit survivable crate-bomb interaction."""

from __future__ import annotations

import numpy as np

import settings as s
from . import continuous_v7
from .common import ACTIONS
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v8-safe-crate-opportunity"
VECTOR_FIELDS = continuous_v7.VECTOR_FIELDS + (
    "survivable_crate_bomb_utility",
)
FEATURE_DIM = 163


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v7.schema().normalization,
            "survivable_crate_bomb_utility": (
                "crates hit/(4*BOMB_POWER) when BOMB is physically legal and "
                "horizon-survivable, otherwise zero"
            ),
        },
    )


def extract(game_state: dict, **kwargs) -> VectorFeatures | None:
    base = continuous_v7.extract(game_state, **kwargs)
    if base is None:
        return None
    context = base.context
    bomb_index = ACTIONS.index("BOMB")
    bomb = context.bomb_reachability
    useful = (
        context.crates_in_blast / float(4 * s.BOMB_POWER)
        if bool(context.legal_mask[bomb_index])
        and bomb is not None
        and bomb.survives_horizon
        and context.crates_in_blast > 0
        else 0.0
    )
    vector = np.concatenate((
        base.vector,
        np.asarray((useful,), dtype=np.float32),
    ))
    return VectorFeatures(
        FEATURE_ID, vector.astype(np.float32, copy=False),
        base.legal_mask.copy(), context,
    )

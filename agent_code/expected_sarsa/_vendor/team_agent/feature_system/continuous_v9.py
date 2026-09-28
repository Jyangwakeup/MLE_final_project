"""Information-preserving compact projection of continuous-v8."""

from __future__ import annotations

import numpy as np

from . import continuous_v8
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v9-compact"

# These fields are structurally constant or exact aliases under the feature
# contract.  ``own_bomb_pending`` is deliberately retained because, unlike the
# aliases below, it is maintained by temporal state and can diagnose drift
# from the framework's bomb-availability flag.
DROPPED_FIELDS = frozenset({
    "wait_legal",
    "wait_coin_distance_delta",
    "wait_crate_frontier_distance_delta",
    "wait_opponent_distance_delta",
    "bomb_coin_distance_delta",
    "bomb_crate_frontier_distance_delta",
    "bomb_opponent_distance_delta",
    "can_drop_bomb",
    "escape_after_bomb",
    "wait_returns_previous_position",
    "bomb_returns_previous_position",
    "wait_revisits_recent_position",
    "wait_cycle_length",
    "wait_cycle_repeats",
    "bomb_revisits_recent_position",
    "bomb_cycle_length",
    "bomb_cycle_repeats",
    "wait_crate_half_plane_density",
    "wait_bomb_objective_distance_delta",
    "bomb_crate_half_plane_density",
    "bomb_bomb_objective_distance_delta",
    "wait_tracked_opponent_distance_delta",
    "bomb_tracked_opponent_distance_delta",
    "bomb_tracked_opponent_blast_line",
})
KEEP_INDICES = tuple(
    index for index, name in enumerate(continuous_v8.VECTOR_FIELDS)
    if name not in DROPPED_FIELDS
)
VECTOR_FIELDS = tuple(continuous_v8.VECTOR_FIELDS[index] for index in KEEP_INDICES)
FEATURE_DIM = len(VECTOR_FIELDS)


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=continuous_v8.ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v8.schema().normalization,
            "compact_projection": (
                "continuous-v8 with 20 structural zeros, wait_legal, and "
                "three exact aliases removed"
            ),
        },
    )


def compact(vector: np.ndarray) -> np.ndarray:
    source = np.asarray(vector, dtype=np.float32)
    if source.shape != (continuous_v8.FEATURE_DIM,):
        raise ValueError(
            f"expected continuous-v8 vector shape {(continuous_v8.FEATURE_DIM,)}, "
            f"got {source.shape}"
        )
    return source[np.asarray(KEEP_INDICES)].astype(np.float32, copy=True)


def extract(game_state: dict, **kwargs) -> VectorFeatures | None:
    base = continuous_v8.extract(game_state, **kwargs)
    if base is None:
        return None
    return VectorFeatures(
        FEATURE_ID, compact(base.vector), base.legal_mask.copy(), base.context)

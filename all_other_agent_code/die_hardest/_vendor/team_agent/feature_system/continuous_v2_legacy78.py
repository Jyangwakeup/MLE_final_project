"""Read-only 78-dimensional feature contract for archived v5 checkpoints."""

from __future__ import annotations

import numpy as np

from . import continuous_v1
from .common import ACTIONS, action_destination, build_context
from .types import FeatureSchema, VectorFeatures

FEATURE_ID = "continuous-v2-legacy78"
HISTORY_FIELDS = ("previous_none",) + tuple(
    f"previous_{action.lower()}" for action in ACTIONS)
VECTOR_FIELDS = continuous_v1.VECTOR_FIELDS + HISTORY_FIELDS + ("wait_streak",)


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID, output_kind="vector", action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS, vector_shape=(78,),
        normalization={
            **continuous_v1.schema().normalization,
            "distance_delta": "clip(before-after, -1, 1); unreachable transitions map to +/-1",
            "previous_action": "one-hot none plus six world actions",
            "wait_streak": "min(consecutive WAIT,3)/3",
        },
    )


def _raw_delta(distances: np.ndarray, before, after) -> float:
    before_distance = float(distances[before])
    after_distance = float(distances[after])
    if not np.isfinite(before_distance) and not np.isfinite(after_distance):
        return 0.0
    if np.isfinite(before_distance) and not np.isfinite(after_distance):
        return -1.0
    if not np.isfinite(before_distance) and np.isfinite(after_distance):
        return 1.0
    return float(np.clip(before_distance - after_distance, -1.0, 1.0))


def extract(game_state, previous_action=None, wait_streak=0):
    if game_state is None:
        return None
    context = build_context(game_state)
    base = continuous_v1.build_from_context(game_state, context).vector.copy()
    for index, action in enumerate(ACTIONS):
        if not context.legal_mask[index]:
            continue
        destination = action_destination(context.position, action)
        offset = index * len(continuous_v1.ACTION_FIELDS)
        base[offset + 7] = _raw_delta(context.coin_distance, context.position, destination)
        base[offset + 8] = _raw_delta(context.crate_frontier_distance, context.position, destination)
        base[offset + 9] = _raw_delta(context.opponent_distance, context.position, destination)
    history = np.zeros(7, dtype=np.float32)
    history[(None, *ACTIONS).index(previous_action) if previous_action in ACTIONS else 0] = 1.0
    vector = np.concatenate((
        base, history,
        np.asarray([min(max(int(wait_streak), 0), 3) / 3.0], dtype=np.float32),
    )).astype(np.float32, copy=False)
    return VectorFeatures(FEATURE_ID, vector, context.legal_mask.copy())

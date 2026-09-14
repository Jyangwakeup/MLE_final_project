"""History-aware extension of the continuous-v1 action representation."""

from __future__ import annotations

import numpy as np

from .common import (
    ACTIONS, MOVE_ACTIONS, action_destination, build_context,
    distance_to_targets, navigation_blocked,
)
from .continuous_v1 import VECTOR_FIELDS as V1_FIELDS, build_from_context as build_v1
from .types import FeatureSchema, VectorFeatures

FEATURE_ID = "continuous-v2"
PREVIOUS_ACTION_FIELDS = tuple(
    f"previous_action_{action.lower()}" for action in ACTIONS
) + ("previous_action_none",)
RETURN_FIELDS = tuple(f"{action.lower()}_returns_previous_position" for action in ACTIONS)
VECTOR_FIELDS = V1_FIELDS + PREVIOUS_ACTION_FIELDS + ("coin_target_continues",) + RETURN_FIELDS
FEATURE_DIM = 84


def selected_coin_target(game_state):
    blocked = navigation_blocked(game_state)
    position = game_state["self"][3]
    distances = distance_to_targets(blocked, (position,))
    reachable = [
        (float(distances[tuple(coin)]), tuple(coin))
        for coin in game_state.get("coins", ())
        if np.isfinite(float(distances[tuple(coin)]))
    ]
    return None if not reachable else min(reachable, key=lambda item: (item[0], item[1]))[1]


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            "continuous-v1-prefix": "first 70 values retain continuous-v1 semantics",
            "previous_action": "one-hot over six actions plus none",
            "coin_target_continues": "binary deterministic BFS target continuity",
            "returns_previous_position": "binary per candidate action",
        },
    )


def extract(
    game_state, *, previous_action=None, previous_position=None,
    previous_coin_target=None,
) -> VectorFeatures | None:
    if game_state is None:
        return None
    context = build_context(game_state)
    base = build_v1(game_state, context)
    previous = np.zeros(7, dtype=np.float32)
    previous[ACTIONS.index(previous_action) if previous_action in ACTIONS else 6] = 1.0
    target = selected_coin_target(game_state)
    target_continues = float(target is not None and target == previous_coin_target)
    returns = []
    for action in ACTIONS:
        returns.append(float(
            action in MOVE_ACTIONS
            and previous_position is not None
            and action_destination(context.position, action) == previous_position
        ))
    vector = np.concatenate((
        base.vector, previous, np.asarray((target_continues,), dtype=np.float32),
        np.asarray(returns, dtype=np.float32),
    )).astype(np.float32, copy=False)
    return VectorFeatures(FEATURE_ID, vector, base.legal_mask.copy())

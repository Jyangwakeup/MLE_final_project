"""Normalized hand-crafted vector features for MLP value models."""

from __future__ import annotations

import numpy as np

import settings as s
from ..danger import HORIZON

from .common import (
    ACTIONS,
    FeatureContext,
    action_destination,
    build_context,
    danger_after_action,
    normalized_distance_delta,
)
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v1"
ACTION_FIELDS = (
    "legal", "danger_t1", "danger_t2", "danger_t3", "earliest_danger",
    "safe_horizon", "safe_area", "coin_distance_delta",
    "crate_frontier_distance_delta", "opponent_distance_delta",
)
GLOBAL_FIELDS = (
    "reachable_coin_exists", "reachable_crate_frontier_exists",
    "reachable_opponent_exists", "can_drop_bomb", "escape_after_bomb",
    "safe_area_after_bomb", "crates_hit", "opponents_threatened",
    "alive_opponents", "round_progress",
)
VECTOR_FIELDS = tuple(
    f"{action.lower()}_{field}" for action in ACTIONS for field in ACTION_FIELDS
) + GLOBAL_FIELDS


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(70,),
        normalization={
            "legal/danger/availability": "binary 0 or 1",
            "earliest_danger": "(t-1)/HORIZON; 1 means no predicted danger",
            "safe_horizon": "last survivable time/HORIZON",
            "safe_area": "last safe frontier size/(width*height)",
            "distance_delta": "clip((before-after)/(width*height-1), -1, 1)",
            "unreachable_distance": "0 when unchanged; finite->unreachable=-1; reverse=1",
            "crates_hit": "count/(4*BOMB_POWER)",
            "opponent_counts": "count/(MAX_AGENTS-1)",
            "round_progress": "(clip(step,1,MAX_STEPS)-1)/(MAX_STEPS-1)",
        },
    )


def _earliest_for_action(context: FeatureContext, action: str) -> float:
    index = ACTIONS.index(action)
    if not context.legal_mask[index]:
        return 0.0
    danger = context.bomb_danger if action == "BOMB" else context.normal_danger
    if danger is None:
        return 0.0
    destination = action_destination(context.position, action)
    for time_step in range(1, HORIZON + 1):
        if danger[time_step, destination[0], destination[1]]:
            return float(time_step - 1) / HORIZON
    return 1.0


def build_from_context(game_state: dict, context: FeatureContext) -> VectorFeatures:
    values = []
    board_area = float(context.width * context.height)
    for index, action in enumerate(ACTIONS):
        legal = bool(context.legal_mask[index])
        destination = action_destination(context.position, action)
        danger_t1, danger_t2, danger_t3 = danger_after_action(context, action)
        reachability = context.movement_reachability[action]
        if legal:
            coin_delta = normalized_distance_delta(
                context.coin_distance, context.position, destination)
            crate_delta = normalized_distance_delta(
                context.crate_frontier_distance, context.position, destination)
            opponent_delta = normalized_distance_delta(
                context.opponent_distance, context.position, destination)
        else:
            coin_delta = crate_delta = opponent_delta = 0.0
        values.extend((
            float(legal), float(danger_t1), float(danger_t2), float(danger_t3),
            _earliest_for_action(context, action),
            reachability.safe_horizon / HORIZON,
            reachability.reachable_area / board_area,
            coin_delta, crate_delta, opponent_delta,
        ))

    can_bomb = bool(context.legal_mask[5])
    bomb = context.bomb_reachability
    values.extend((
        float(context.reachable_coin_exists),
        float(context.reachable_crate_frontier_exists),
        float(context.reachable_opponent_exists),
        float(can_bomb),
        float(bool(bomb and bomb.survives_horizon)),
        (bomb.reachable_area / board_area) if bomb is not None else 0.0,
        context.crates_in_blast / float(4 * s.BOMB_POWER) if can_bomb else 0.0,
        context.opponents_in_blast / float(max(1, s.MAX_AGENTS - 1)) if can_bomb else 0.0,
        len(game_state["others"]) / float(max(1, s.MAX_AGENTS - 1)),
        (min(max(int(game_state.get("step", 1)), 1), s.MAX_STEPS) - 1)
        / float(max(1, s.MAX_STEPS - 1)),
    ))
    vector = np.asarray(values, dtype=np.float32)
    return VectorFeatures(FEATURE_ID, vector, context.legal_mask.copy())


def extract(game_state: dict) -> VectorFeatures:
    if game_state is None:
        return None
    return build_from_context(game_state, build_context(game_state))

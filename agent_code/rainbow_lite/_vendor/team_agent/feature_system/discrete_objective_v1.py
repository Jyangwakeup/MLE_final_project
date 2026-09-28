"""Compact categorical facts for learning navigation, bombing, and escape."""

from __future__ import annotations

import numpy as np

from .common import ACTIONS, MOVE_ACTIONS, action_destination, build_context
from .types import DiscreteFeatures, FeatureSchema


FEATURE_ID = "discrete-objective-v1"
STATE_FIELDS = (
    "up_safety", "right_safety", "down_safety", "left_safety",
    "up_objective_progress", "right_objective_progress",
    "down_objective_progress", "left_objective_progress",
    "current_danger", "bomb_outcome", "objective_distance",
    "objective_kind", "previous_action", "wait_streak",
)
CATEGORY_COUNTS = (3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 3, 7, 3)
PREVIOUS_ACTION = {
    None: 0, "UP": 1, "RIGHT": 2, "DOWN": 3, "LEFT": 4, "WAIT": 5, "BOMB": 6,
}


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="discrete",
        action_order=ACTIONS,
        state_fields=STATE_FIELDS,
        category_counts=CATEGORY_COUNTS,
        state_shape=(14,),
        vector_shape=(60,),
        theoretical_state_count=398_671_875,
        normalization={
            "movement_safety": "0=illegal, 1=legal but not horizon-survivable, 2=survives H=7",
            "objective_progress": "0=illegal, 1=no target, 2=worse/unreachable, 3=equal, 4=closer/newly reachable",
            "current_danger": "0=safe through H, 1=t1, 2=t2-3, 3=t4-H, 4=current explosion",
            "bomb_outcome": "0=unavailable, 1=no escape, 2=safe/0 crates, 3=safe/1 crate, 4=safe/2+ crates",
            "objective_distance": "0=none, 1=0, 2=1-2, 3=3-5, 4=6+",
            "objective_kind": "0=none, 1=reachable coin, 2=reachable crate frontier",
            "previous_action": "0=none, then world action order",
            "wait_streak": "0, 1, or 2+ consecutive WAIT actions",
        },
    )


def _distance_bucket(distance: float) -> int:
    if not np.isfinite(distance):
        return 0
    if distance <= 0:
        return 1
    if distance <= 2:
        return 2
    if distance <= 5:
        return 3
    return 4


def _progress_category(before: float, after: float, legal: bool) -> int:
    if not legal:
        return 0
    if not np.isfinite(before):
        return 1
    if not np.isfinite(after) or after > before:
        return 2
    if after == before:
        return 3
    return 4


def _current_danger_category(game_state: dict, context) -> int:
    position = context.position
    if int(game_state["explosion_map"][position]) > 0:
        return 4
    earliest = int(context.earliest_danger[position])
    if earliest > context.normal_danger.shape[0] - 1:
        return 0
    if earliest == 1:
        return 1
    if earliest <= 3:
        return 2
    return 3


def _one_hot(state_key: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(sum(CATEGORY_COUNTS), dtype=np.float32)
    offset = 0
    for value, count in zip(state_key, CATEGORY_COUNTS):
        vector[offset + value] = 1.0
        offset += count
    return vector


def extract(
    game_state: dict,
    previous_action: str | None = None,
    wait_streak: int = 0,
) -> DiscreteFeatures | None:
    if game_state is None:
        return None
    context = build_context(game_state)
    position = context.position
    if context.reachable_coin_exists:
        objective_kind = 1
        distances = context.coin_distance
    elif context.reachable_crate_frontier_exists:
        objective_kind = 2
        distances = context.crate_frontier_distance
    else:
        objective_kind = 0
        distances = np.full(context.field.shape, np.inf, dtype=np.float32)
    before = float(distances[position])

    safety = []
    progress = []
    for index, action in enumerate(MOVE_ACTIONS):
        legal = bool(context.legal_mask[index])
        reachability = context.movement_reachability[action]
        safety.append(0 if not legal else (2 if reachability.survives_horizon else 1))
        destination = action_destination(position, action)
        after = float(distances[destination]) if legal else np.inf
        progress.append(_progress_category(before, after, legal))

    if not bool(context.legal_mask[ACTIONS.index("BOMB")]):
        bomb_outcome = 0
    elif not bool(context.bomb_reachability and context.bomb_reachability.survives_horizon):
        bomb_outcome = 1
    elif context.crates_in_blast == 0:
        bomb_outcome = 2
    elif context.crates_in_blast == 1:
        bomb_outcome = 3
    else:
        bomb_outcome = 4

    state_key = tuple(safety + progress + [
        _current_danger_category(game_state, context),
        bomb_outcome,
        _distance_bucket(before),
        objective_kind,
        PREVIOUS_ACTION.get(previous_action, 0),
        min(max(int(wait_streak), 0), 2),
    ])
    return DiscreteFeatures(
        FEATURE_ID, state_key, _one_hot(state_key), context.legal_mask.copy())

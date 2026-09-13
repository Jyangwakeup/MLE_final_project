"""Q-learning state with distance and one-step action memory."""

from __future__ import annotations

import numpy as np

from .common import ACTIONS, distance_to_targets, navigation_blocked
from .types import DiscreteFeatures, FeatureSchema


FEATURE_ID = "discrete-q-v2"
STATE_FIELDS = (
    "move_up_safety", "move_right_safety", "move_down_safety", "move_left_safety",
    "current_danger", "bomb_safety", "crates_hit_bucket",
    "coin_closer_up", "coin_closer_right", "coin_closer_down", "coin_closer_left",
    "nearest_opponent_direction", "opponent_distance_bucket", "opponent_in_blast",
    "nearest_coin_distance_bucket", "previous_move",
)
CATEGORY_COUNTS = (3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 5, 4, 2, 5, 5)
PREVIOUS_MOVE = {None: 0, "UP": 1, "RIGHT": 2, "DOWN": 3, "LEFT": 4}


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="discrete",
        action_order=ACTIONS,
        state_fields=STATE_FIELDS,
        category_counts=CATEGORY_COUNTS,
        state_shape=(16,),
        vector_shape=(50,),
        theoretical_state_count=34_992_000,
        normalization={
            "vector": "one-hot categorical state",
            "nearest_coin_distance_bucket": "0=none/unreachable, 1=1-2, 2=3-5, 3=6-9, 4=10+",
            "previous_move": "0=none/non-move, 1=up, 2=right, 3=down, 4=left",
        },
    )


def nearest_coin_distance(game_state: dict) -> float:
    """Return the current static path distance to the nearest reachable coin."""
    blocked = navigation_blocked(game_state)
    distances = distance_to_targets(blocked, tuple(game_state["coins"]))
    return float(distances[game_state["self"][3]])


def _distance_bucket(distance: float) -> int:
    if not np.isfinite(distance) or distance <= 0:
        return 0
    if distance <= 2:
        return 1
    if distance <= 5:
        return 2
    if distance <= 9:
        return 3
    return 4


def _one_hot(state_key: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(sum(CATEGORY_COUNTS), dtype=np.float32)
    offset = 0
    for value, count in zip(state_key, CATEGORY_COUNTS):
        vector[offset + value] = 1.0
        offset += count
    return vector


def extract(game_state: dict, previous_action: str | None = None):
    if game_state is None:
        return None
    # Keep the compatibility baseline lazy so importing team_agent.features
    # directly does not cycle through the registry back into this module.
    from ..features import _extract_features_with_coin_distance
    base, coin_distance = _extract_features_with_coin_distance(game_state)
    previous_move = PREVIOUS_MOVE.get(previous_action, 0)
    state_key = base.state_key + (
        _distance_bucket(coin_distance), previous_move)
    return DiscreteFeatures(
        feature_id=FEATURE_ID,
        state_key=state_key,
        vector=_one_hot(state_key),
        legal_mask=base.legal_mask.copy(),
        action_transform=base.action_transform,
    )

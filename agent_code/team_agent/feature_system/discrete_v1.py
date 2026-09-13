"""Frozen adapter for the original 14-field / 40-vector representation."""

from __future__ import annotations

from .common import ACTIONS
from .types import FeatureSchema


FEATURE_ID = "discrete-v1"
STATE_FIELDS = (
    "move_up_safety", "move_right_safety", "move_down_safety", "move_left_safety",
    "current_danger", "bomb_safety", "crates_hit_bucket",
    "coin_closer_up", "coin_closer_right", "coin_closer_down", "coin_closer_left",
    "nearest_opponent_direction", "opponent_distance_bucket", "opponent_in_blast",
)
CATEGORY_COUNTS = (3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 5, 4, 2)


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="discrete",
        action_order=ACTIONS,
        state_fields=STATE_FIELDS,
        category_counts=CATEGORY_COUNTS,
        state_shape=(14,),
        vector_shape=(40,),
        theoretical_state_count=1_399_680,
        normalization={"vector": "one-hot encoding of the frozen categorical state"},
    )


def extract(game_state: dict):
    # Lazy import avoids making the legacy module depend on the registry.
    from ..features import extract_features
    return extract_features(game_state)

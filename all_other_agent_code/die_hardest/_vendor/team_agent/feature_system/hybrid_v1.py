"""Composed spatial and vector representation without duplicated extraction."""

from __future__ import annotations

from . import continuous_v1
from . import board_v1
from .common import ACTIONS, build_context
from .types import FeatureSchema, HybridFeatures


FEATURE_ID = "hybrid-v1"


def schema(board_shape=None) -> FeatureSchema:
    shape = (12, None, None) if board_shape is None else (12, *tuple(board_shape))
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="hybrid",
        action_order=ACTIONS,
        vector_fields=continuous_v1.VECTOR_FIELDS,
        board_channels=board_v1.BOARD_CHANNELS,
        vector_shape=(70,),
        board_shape=shape,
        normalization={
            "board": "identical to board-v1",
            "vector": "identical to continuous-v1",
        },
    )


def extract(game_state: dict) -> HybridFeatures:
    if game_state is None:
        return None
    context = build_context(game_state)
    board = board_v1.build_from_context(game_state, context)
    vector = continuous_v1.build_from_context(game_state, context)
    return HybridFeatures(
        FEATURE_ID, board.board, vector.vector, context.legal_mask.copy())

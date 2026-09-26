"""Raw spatial board channels for convolutional models."""

from __future__ import annotations

import numpy as np

import settings as s
from ..danger import predict_danger

from .common import ACTIONS, FeatureContext, physical_legal_mask
from .types import BoardFeatures, FeatureSchema


FEATURE_ID = "board-v1"
BOARD_CHANNELS = (
    "stone_wall", "crate", "coin", "self", "opponent", "bomb_presence",
    "bomb_timer", "current_explosion", "danger_t1", "danger_t2", "danger_t3",
    "danger_t2_or_later",
)


def schema(board_shape=None) -> FeatureSchema:
    shape = (12, None, None) if board_shape is None else (12, *tuple(board_shape))
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="board",
        action_order=ACTIONS,
        board_channels=BOARD_CHANNELS,
        board_shape=shape,
        normalization={
            "binary_channels": "0 or 1",
            "bomb_timer": "clip(timer/BOMB_TIMER, 0, 1)",
            "coordinates": "board[channel,x,y] corresponds to field[x,y]",
        },
    )


def build_from_context(game_state: dict, context: FeatureContext) -> BoardFeatures:
    return _build_board(
        game_state, context.field, context.position, context.legal_mask,
        context.normal_danger,
    )


def _build_board(game_state, field, position, legal_mask, normal_danger):
    """Build spatial channels from only the objective facts they require."""
    width, height = field.shape
    board = np.zeros((12, width, height), dtype=np.float32)
    board[0] = field == -1
    board[1] = field == 1
    for position in game_state["coins"]:
        board[2, position[0], position[1]] = 1.0
    board[3, position[0], position[1]] = 1.0
    for _, _, _, position in game_state["others"]:
        board[4, position[0], position[1]] = 1.0
    for position, timer in game_state["bombs"]:
        board[5, position[0], position[1]] = 1.0
        board[6, position[0], position[1]] = np.clip(
            timer / float(s.BOMB_TIMER), 0.0, 1.0)
    board[7] = np.asarray(game_state["explosion_map"] > 0, dtype=np.float32)
    board[8] = normal_danger[1]
    board[9] = normal_danger[2]
    board[10] = normal_danger[3]
    board[11] = np.any(normal_danger[2:], axis=0)
    return BoardFeatures(FEATURE_ID, board, legal_mask.copy())


def extract(game_state: dict) -> BoardFeatures:
    if game_state is None:
        return None
    # board-v1 deliberately contains no path or escape summaries, so avoid
    # computing the full navigation context used by continuous/hybrid-v1.
    return _build_board(
        game_state,
        game_state["field"],
        game_state["self"][3],
        physical_legal_mask(game_state),
        predict_danger(game_state).danger,
    )

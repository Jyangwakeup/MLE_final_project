"""Agent-local 17-channel board-path-history-v2 encoder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import settings as s

from agent_code.team_agent.feature_system import ACTIONS, extract_features
from agent_code.team_agent.feature_system.common import build_context


FEATURE_ID = "board-path-history-v2"
CHANNELS = (
    "stone_wall", "crate", "coin", "self", "opponent", "bomb_presence",
    "bomb_timer", "current_explosion", "danger_t1", "danger_t2", "danger_t3",
    "danger_t2_or_later", "nearest_coin_distance",
    "nearest_crate_frontier_distance", "nearest_opponent_distance",
    "previous_position", "recent_visit_frequency",
)
BOARD_SHAPE = (17, s.COLS, s.ROWS)
FEATURE_SCHEMA = {
    "feature_id": FEATURE_ID, "output_kind": "board",
    "action_order": list(ACTIONS), "state_fields": None,
    "category_counts": None, "vector_fields": None,
    "board_channels": list(CHANNELS), "state_shape": None,
    "vector_shape": None, "board_shape": list(BOARD_SHAPE),
    "theoretical_state_count": None,
    "normalization": {
        "base": "board-v1 semantics",
        "objective_distance": "min(shortest_path,32)/32; unavailable=1",
        "previous_position": "binary",
        "recent_visit_frequency": "count in previous 16 decisions / 16",
    },
}
HISTORY_LENGTH = 16
DISTANCE_SCALE = float(s.COLS + s.ROWS - 2)


@dataclass(frozen=True)
class Features:
    board: np.ndarray
    legal_mask: np.ndarray


def initialize_history(owner):
    owner.cnn_distilled_history = deque(maxlen=HISTORY_LENGTH)
    owner.cnn_distilled_history_round = None


def reset_history(owner):
    initialize_history(owner)


def record_position(owner, game_state):
    if game_state is None:
        return
    round_index = game_state.get("round")
    if getattr(owner, "cnn_distilled_history_round", None) != round_index:
        initialize_history(owner)
        owner.cnn_distilled_history_round = round_index
    owner.cnn_distilled_history.append(tuple(game_state["self"][3]))


def _distance(values):
    result = np.ones(values.shape, dtype=np.float32)
    finite = np.isfinite(values)
    result[finite] = np.minimum(values[finite], DISTANCE_SCALE) / DISTANCE_SCALE
    return result


def features_for_state(owner, game_state):
    if game_state is None:
        return None
    round_index = game_state.get("round")
    if getattr(owner, "cnn_distilled_history_round", None) != round_index:
        initialize_history(owner)
        owner.cnn_distilled_history_round = round_index
    base = extract_features(game_state, "board-v1")
    context = build_context(game_state)
    board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    board[:12] = base.board
    # Local correction for the shared board-v1 variable-shadowing bug.
    board[3] = 0.0
    self_x, self_y = game_state["self"][3]
    board[3, self_x, self_y] = 1.0
    board[12] = _distance(context.coin_distance)
    board[13] = _distance(context.crate_frontier_distance)
    board[14] = _distance(context.opponent_distance)
    history = tuple(getattr(owner, "cnn_distilled_history", ()))
    if history:
        previous_x, previous_y = history[-1]
        board[15, previous_x, previous_y] = 1.0
        for x, y in history:
            board[16, x, y] += 1.0 / HISTORY_LENGTH
    return Features(board, base.legal_mask.copy())

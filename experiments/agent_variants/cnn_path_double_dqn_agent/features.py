from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import settings as s

from agent_code.team_agent.feature_system import ACTIONS, extract_features
from agent_code.team_agent.feature_system.common import build_context


FEATURE_ID = "board-path-history-v2"
BASE_CHANNELS = (
    "stone_wall", "crate", "coin", "self", "opponent", "bomb_presence",
    "bomb_timer", "current_explosion", "danger_t1", "danger_t2", "danger_t3",
    "danger_t2_or_later",
)
EXTRA_CHANNELS = (
    "nearest_coin_distance", "nearest_crate_frontier_distance",
    "nearest_opponent_distance", "previous_position", "recent_visit_frequency",
)
CHANNELS = BASE_CHANNELS + EXTRA_CHANNELS
BOARD_SHAPE = (len(CHANNELS), s.COLS, s.ROWS)
FEATURE_SCHEMA = {
    "feature_id": FEATURE_ID,
    "output_kind": "board",
    "action_order": list(ACTIONS),
    "state_fields": None,
    "category_counts": None,
    "vector_fields": None,
    "board_channels": list(CHANNELS),
    "state_shape": None,
    "vector_shape": None,
    "board_shape": list(BOARD_SHAPE),
    "theoretical_state_count": None,
    "normalization": {
        "base": "board-v1 semantics",
        "objective_distance": "min(shortest_path,32)/32; unavailable=1",
        "previous_position": "binary",
        "recent_visit_frequency": "count in previous 16 decisions / 16",
    },
}
DISTANCE_SCALE = float(s.COLS + s.ROWS - 2)
HISTORY_LENGTH = 16


@dataclass(frozen=True)
class PathBoardFeatures:
    feature_id: str
    board: np.ndarray
    legal_mask: np.ndarray


def initialize_history(owner) -> None:
    owner.cnn_path_history = deque(maxlen=HISTORY_LENGTH)
    owner.cnn_path_history_round = None


def reset_history(owner) -> None:
    initialize_history(owner)


def record_position(owner, game_state) -> None:
    if game_state is None:
        return
    round_index = game_state.get("round")
    if getattr(owner, "cnn_path_history_round", None) != round_index:
        owner.cnn_path_history = deque(maxlen=HISTORY_LENGTH)
        owner.cnn_path_history_round = round_index
    owner.cnn_path_history.append(tuple(game_state["self"][3]))


def _normalized_distance(values: np.ndarray) -> np.ndarray:
    result = np.ones(values.shape, dtype=np.float32)
    finite = np.isfinite(values)
    result[finite] = np.minimum(values[finite], DISTANCE_SCALE) / DISTANCE_SCALE
    return result


def features_for_state(owner, game_state) -> PathBoardFeatures | None:
    if game_state is None:
        return None
    if getattr(owner, "cnn_path_history_round", None) != game_state.get("round"):
        initialize_history(owner)
        owner.cnn_path_history_round = game_state.get("round")
    base = extract_features(game_state, "board-v1")
    context = build_context(game_state)
    board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    board[:12] = base.board
    board[3] = 0.0
    x, y = game_state["self"][3]
    board[3, x, y] = 1.0
    board[12] = _normalized_distance(context.coin_distance)
    board[13] = _normalized_distance(context.crate_frontier_distance)
    board[14] = _normalized_distance(context.opponent_distance)
    history = tuple(getattr(owner, "cnn_path_history", ()))
    if history:
        previous = history[-1]
        board[15, previous[0], previous[1]] = 1.0
        for x, y in history:
            board[16, x, y] += 1.0 / HISTORY_LENGTH
    return PathBoardFeatures(FEATURE_ID, board, base.legal_mask.copy())

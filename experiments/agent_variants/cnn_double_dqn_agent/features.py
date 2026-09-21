import numpy as np
import settings as s

from agent_code.team_agent.feature_system import (
    ACTIONS, BoardFeatures, extract_features, feature_schema_contract,
)

FEATURE_ID = "board-v1"
BOARD_SHAPE = (12, s.COLS, s.ROWS)
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID, (s.COLS, s.ROWS))


def features_for_state(game_state):
    result = extract_features(game_state, FEATURE_ID)
    if result is None:
        return None
    if not isinstance(result, BoardFeatures):
        raise TypeError("board-v1 did not return BoardFeatures")
    if result.board.shape != BOARD_SHAPE or result.board.dtype != np.float32:
        raise ValueError(f"board-v1 returned {result.board.shape}; expected {BOARD_SHAPE}")
    if result.legal_mask.shape != (len(ACTIONS),) or result.legal_mask.dtype != np.bool_:
        raise ValueError("board-v1 returned an incompatible legal mask")
    return result

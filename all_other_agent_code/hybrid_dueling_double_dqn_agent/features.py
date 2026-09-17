import numpy as np
import settings as s

from agent_code.team_agent.feature_system import (
    ACTIONS, HybridFeatures, extract_features, feature_schema_contract,
)

FEATURE_ID = "hybrid-v1"
BOARD_SHAPE = (12, s.COLS, s.ROWS)
VECTOR_SHAPE = (70,)
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID, (s.COLS, s.ROWS))


def features_for_state(game_state):
    result = extract_features(game_state, FEATURE_ID)
    if result is None:
        return None
    if not isinstance(result, HybridFeatures):
        raise TypeError("hybrid-v1 did not return HybridFeatures")
    if result.board.shape != BOARD_SHAPE or result.board.dtype != np.float32:
        raise ValueError("hybrid-v1 returned an incompatible board")
    if result.vector.shape != VECTOR_SHAPE or result.vector.dtype != np.float32:
        raise ValueError("hybrid-v1 returned an incompatible vector")
    if result.legal_mask.shape != (len(ACTIONS),) or result.legal_mask.dtype != np.bool_:
        raise ValueError("hybrid-v1 returned an incompatible legal mask")
    return result

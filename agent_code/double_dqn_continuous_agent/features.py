import numpy as np

from agent_code.team_agent.feature_system import (
    ACTIONS, VectorFeatures, extract_features, feature_schema_contract,
)

FEATURE_ID = "continuous-v1"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)
FEATURE_DIM = 70


def features_for_state(game_state):
    result = extract_features(game_state, FEATURE_ID)
    if result is None:
        return None
    if not isinstance(result, VectorFeatures):
        raise TypeError("continuous-v1 did not return VectorFeatures")
    if result.vector.shape != (FEATURE_DIM,) or result.vector.dtype != np.float32:
        raise ValueError("continuous-v1 returned an incompatible vector")
    if result.legal_mask.shape != (len(ACTIONS),) or result.legal_mask.dtype != np.bool_:
        raise ValueError("continuous-v1 returned an incompatible legal mask")
    return result

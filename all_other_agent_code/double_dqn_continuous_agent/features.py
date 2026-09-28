import numpy as np

from agent_code.team_agent.feature_system import ACTIONS, VectorFeatures, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import extract

FEATURE_ID = "continuous-v2"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)
FEATURE_DIM = 84


def features_for_state(game_state, **history):
    result = extract(game_state, **history)
    if result is None:
        return None
    if not isinstance(result, VectorFeatures):
        raise TypeError("continuous-v2 did not return VectorFeatures")
    if result.vector.shape != (FEATURE_DIM,) or result.vector.dtype != np.float32:
        raise ValueError("continuous-v2 returned an incompatible vector")
    if result.legal_mask.shape != (len(ACTIONS),) or result.legal_mask.dtype != np.bool_:
        raise ValueError("continuous-v2 returned an incompatible legal mask")
    return result

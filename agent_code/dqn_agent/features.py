"""Compatibility wrappers around selectable shared discrete features."""

import numpy as np

from agent_code.team_agent.feature_system import (
    get_feature_extractor, get_feature_schema, normalize_feature_id,
)
from agent_code.team_agent.feature_system.common import ACTIONS

FEATURE_ID = "discrete-q-v2"
FEATURE_DIM = get_feature_schema(FEATURE_ID).vector_shape[0]
FEATURE_VERSION = None


def features_for_state(game_state: dict, previous_action=None, feature_id=FEATURE_ID):
    resolved = normalize_feature_id(feature_id)
    extractor = get_feature_extractor(resolved)
    if resolved == "discrete-q-v2":
        return extractor(game_state, previous_action)
    return extractor(game_state)


def legal_actions(game_state: dict, feature_id: str = FEATURE_ID) -> np.ndarray:
    features = features_for_state(game_state, feature_id=feature_id)
    if features is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    return features.legal_mask.copy()


def state_to_features(game_state: dict, feature_id: str = FEATURE_ID):
    features = features_for_state(game_state, feature_id=feature_id)
    return None if features is None else features.vector

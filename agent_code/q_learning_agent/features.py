"""Compatibility wrappers around selectable shared discrete features."""

import numpy as np

from agent_code.team_agent.feature_system import get_feature_extractor, normalize_feature_id
from agent_code.team_agent.feature_system.common import ACTIONS

FEATURE_ID = "discrete-q-v2"
FEATURE_VERSION = None


def features_for_state(
    game_state: dict, previous_action=None, feature_id=FEATURE_ID, wait_streak=0,
):
    resolved = normalize_feature_id(feature_id)
    extractor = get_feature_extractor(resolved)
    if resolved == "discrete-q-v2":
        return extractor(game_state, previous_action)
    if resolved == "discrete-objective-v1":
        return extractor(game_state, previous_action, wait_streak)
    return extractor(game_state)


def legal_actions(
    game_state: dict, allow_bomb: bool = True, feature_id: str = FEATURE_ID,
) -> np.ndarray:
    features = features_for_state(game_state, feature_id=feature_id)
    if features is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    legal = features.legal_mask.copy()
    if not allow_bomb:
        legal[ACTIONS.index("BOMB")] = False
    return legal


def state_to_features(game_state: dict, feature_id: str = FEATURE_ID):
    features = features_for_state(game_state, feature_id=feature_id)
    return None if features is None else features.state_key

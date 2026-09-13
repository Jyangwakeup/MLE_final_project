"""Compatibility wrappers around the shared team feature interface."""

import numpy as np

from agent_code.team_agent.feature_system.common import ACTIONS
from agent_code.team_agent.feature_system.discrete_q_v2 import (
    FEATURE_ID, extract as extract_features,
)

FEATURE_VERSION = None


def features_for_state(game_state: dict, previous_action=None):
    """Return all shared features so callers can reuse one extraction."""
    return extract_features(game_state, previous_action)


def legal_actions(game_state: dict, allow_bomb: bool = True) -> np.ndarray:
    """Return the team-agent legal mask, optionally disabling bombs."""
    features = features_for_state(game_state)
    if features is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    legal = features.legal_mask.copy()
    if not allow_bomb:
        legal[ACTIONS.index("BOMB")] = False
    return legal


def state_to_features(game_state: dict):
    """Return the shared team-agent discrete state key for tabular Q-learning."""
    features = features_for_state(game_state)
    return None if features is None else features.state_key

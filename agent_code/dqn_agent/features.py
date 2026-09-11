"""Compatibility wrappers around the shared team feature interface."""

import numpy as np

from agent_code.team_agent.features import (
    ACTIONS,
    FEATURE_DIM,
    FEATURE_VERSION,
    extract_features,
)


def features_for_state(game_state: dict):
    """Return the complete shared representation for one game state."""
    return extract_features(game_state)


def legal_actions(game_state: dict) -> np.ndarray:
    """Return the physical legal-action mask from the shared extractor."""
    features = features_for_state(game_state)
    if features is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    return features.legal_mask.copy()


def state_to_features(game_state: dict):
    """Return the shared 40-dimensional vector used by the DQN."""
    features = features_for_state(game_state)
    return None if features is None else features.vector

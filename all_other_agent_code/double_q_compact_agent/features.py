"""Thin adapter for the public discrete-compact-v1 extractor."""

import numpy as np

from agent_code.team_agent.feature_system import (
    ACTIONS, DiscreteFeatures, extract_features, feature_schema_contract,
)

FEATURE_ID = "discrete-compact-v1"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(game_state):
    result = extract_features(game_state, FEATURE_ID)
    if result is None:
        return None
    if not isinstance(result, DiscreteFeatures):
        raise TypeError("discrete-compact-v1 did not return DiscreteFeatures")
    if len(result.state_key) != 12 or result.vector.shape != (38,):
        raise ValueError("discrete-compact-v1 returned an incompatible shape")
    if result.vector.dtype != np.float32 or result.legal_mask.dtype != np.bool_:
        raise TypeError("discrete-compact-v1 returned an incompatible dtype")
    return result


def canonical_legal_mask(features, *, allow_bomb=True):
    world = features.legal_mask.copy()
    if not allow_bomb:
        world[ACTIONS.index("BOMB")] = False
    canonical = np.zeros(len(ACTIONS), dtype=bool)
    for world_index, canonical_index in enumerate(
        features.action_transform.world_to_canonical
    ):
        canonical[canonical_index] = world[world_index]
    return canonical

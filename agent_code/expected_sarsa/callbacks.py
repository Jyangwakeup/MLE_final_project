"""Continuous-v4 Expected SARSA(lambda) callbacks."""

from pathlib import Path

import numpy as np

from agent_code.expected_sarsa._vendor.learning_common import linear_agent
from agent_code.expected_sarsa._vendor.learning_common.tile_coding import TraceControl
from agent_code.expected_sarsa._vendor.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.expected_sarsa._vendor.team_agent.feature_system.continuous_v4 import VECTOR_FIELDS
from .features import features_for_state


ALGORITHM = "expected_sarsa_lambda"
FEATURE_ID = "continuous-v4"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)
MODEL_FILE = Path(__file__).with_name("final.pkl")
ACTION_FEATURE_INDICES = [
    list(range(action * 10, action * 10 + 10))
    + list(range(60, 78))
    + [78 + action]
    + list(range(84 + action * 3, 87 + action * 3))
    + list(range(102, 108))
    + list(range(108 + action * 3, 111 + action * 3))
    for action in range(len(ACTIONS))
]
HYPERPARAMETERS = {
    "gamma": 0.95,
    "learning_rate": 0.08,
    "lambda": 0.8,
    "tilings": 8,
    "bins": 8,
    "memory_size": 32768,
    "signed_indices": [
        i for i, name in enumerate(VECTOR_FIELDS) if "distance_delta" in name
    ],
    "ternary_indices": [
        i
        for i, name in enumerate(VECTOR_FIELDS)
        if name.endswith("coin_distance_delta")
    ],
    "action_feature_indices": ACTION_FEATURE_INDICES,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM,
    "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name,
    "network_spec": None,
    "hyperparameters": HYPERPARAMETERS,
}


def make_model(seed):
    return TraceControl(
        126,
        len(ACTIONS),
        seed=seed,
        hyperparameters=HYPERPARAMETERS,
        algorithm=ALGORITHM,
    )


def setup(self):
    linear_agent.setup(self, __import__(__name__, fromlist=["x"]))


def act(self, game_state):
    return linear_agent.act(self, game_state)


def state_to_features(game_state):
    value = features_for_state(type("FeatureOwner", (), {})(), game_state)
    return None if value is None else value.vector


def legal_actions(game_state):
    value = features_for_state(type("FeatureOwner", (), {})(), game_state)
    if value is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    return value.legal_mask.copy()

"""Callbacks for grouped tile-coded Double Q(lambda)."""

from pathlib import Path

import numpy as np

from agent_code.learning_common import linear_agent
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import VECTOR_FIELDS

from .features import features_for_state
from .model import GroupedTraceControl


ALGORITHM = "double_q_lambda"
FEATURE_ID = "continuous-v2"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)
MODEL_FILE = Path(__file__).with_name("final.pkl")
BASE_GROUPS = [
    list(range(action * 10, action * 10 + 10)) + list(range(60, 77))
    for action in range(len(ACTIONS))
]
HISTORY_GROUPS = [
    [60, 61, 63, 77, 78 + action] for action in range(len(ACTIONS))
]
GROUP_ACTION_FEATURE_INDICES = [
    [BASE_GROUPS[action], HISTORY_GROUPS[action]]
    for action in range(len(ACTIONS))
]
HYPERPARAMETERS = {
    "gamma": 0.95,
    "learning_rate": 0.08,
    "learning_rate_final": 0.02,
    "learning_rate_decay_steps": 200000,
    "lambda": 0.8,
    "tilings": 8,
    "bins": 8,
    "memory_size": 32768,
    "signed_indices": [i for i, name in enumerate(VECTOR_FIELDS) if "distance_delta" in name],
    "ternary_indices": [i for i, name in enumerate(VECTOR_FIELDS) if name.endswith("coin_distance_delta")],
    "group_action_feature_indices": GROUP_ACTION_FEATURE_INDICES,
    "group_tilings": [4, 4],
    "group_memory_sizes": [16384, 16384],
    "watkins_trace_cut": True,
    "history_input_version": 1,
    "grouped_tile_encoding_version": 1,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": None,
    "hyperparameters": HYPERPARAMETERS,
}


def make_model(seed):
    return GroupedTraceControl(
        84, len(ACTIONS), seed=seed, hyperparameters=HYPERPARAMETERS,
        algorithm=ALGORITHM)


def setup(self):
    linear_agent.setup(self, __import__(__name__, fromlist=["unused"]))


def act(self, game_state):
    return linear_agent.act(self, game_state)


def state_to_features(game_state):
    value = features_for_state(type("FeatureOwner", (), {})(), game_state)
    return None if value is None else value.vector


def legal_actions(game_state):
    value = features_for_state(type("FeatureOwner", (), {})(), game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if value is None else value.legal_mask.copy()

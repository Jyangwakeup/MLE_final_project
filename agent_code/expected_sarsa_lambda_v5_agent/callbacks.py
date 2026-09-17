from pathlib import Path
import numpy as np

from agent_code.learning_common import linear_agent
from agent_code.learning_common.tile_coding import TraceControl
from agent_code.team_agent.feature_system.continuous_v5 import VECTOR_FIELDS
from .features import (
    ACTIONS, FEATURE_ID, FEATURE_SCHEMA, action_history_state,
    features_for_state, init_action_history, load_action_history_state,
    record_selected_action,
)

ALGORITHM = "expected_sarsa_lambda"
MODEL_FILE = Path(__file__).with_name("final.pkl")
ACTION_FEATURE_INDICES = [
    list(range(action * 10, action * 10 + 10))
    + list(range(60, 78)) + [78 + action]
    + list(range(84 + action * 3, 87 + action * 3))
    + list(range(102, 108))
    + list(range(108 + action * 3, 111 + action * 3))
    + list(range(126 + action * 2, 128 + action * 2))
    + [138, 139]
    for action in range(len(ACTIONS))
]
HYPERPARAMETERS = {
    "gamma": .95, "learning_rate": .08, "lambda": .8,
    "tilings": 8, "bins": 8, "memory_size": 32768,
    "signed_indices": [
        i for i, name in enumerate(VECTOR_FIELDS) if "distance_delta" in name
    ],
    "ternary_indices": [
        i for i, name in enumerate(VECTOR_FIELDS)
        if name.endswith("coin_distance_delta")
    ],
    "action_feature_indices": ACTION_FEATURE_INDICES,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": None,
    "hyperparameters": HYPERPARAMETERS,
}

def make_model(seed):
    return TraceControl(
        140, len(ACTIONS), seed=seed, hyperparameters=HYPERPARAMETERS,
        algorithm=ALGORITHM,
    )

def setup(self): linear_agent.setup(self, __import__(__name__, fromlist=["x"]))
def act(self, game_state): return linear_agent.act(self, game_state)
def state_to_features(game_state):
    value = features_for_state(type("FeatureOwner", (), {})(), game_state)
    return None if value is None else value.vector
def legal_actions(game_state):
    value = features_for_state(type("FeatureOwner", (), {})(), game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if value is None else value.legal_mask.copy()

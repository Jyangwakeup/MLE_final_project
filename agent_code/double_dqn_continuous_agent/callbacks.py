from pathlib import Path

import numpy as np

from agent_code.learning_common.neural_agent import (
    act_neural, cached_features, setup_neural_agent,
)
from .features import ACTIONS, FEATURE_ID, FEATURE_SCHEMA, features_for_state
from .model import build_learner


ALGORITHM = "double_dqn"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = {
    "gamma": 0.95, "learning_rate": 3e-4, "batch_size": 64,
    "replay_capacity": 50_000, "warmup": 2_000,
    "target_sync_interval": 1_000, "gradient_clip": 10.0,
    "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_action_steps": 80_000,
}
NETWORK_SPEC = {
    "id": "continuous-mlp-128x2-v1", "input_shape": [70],
    "layers": ["Linear(70,128)", "ReLU", "Linear(128,128)", "ReLU", "Linear(128,6)"],
    "dueling": False, "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": NETWORK_SPEC,
    "hyperparameters": HYPERPARAMETERS,
}


def setup(self):
    setup_neural_agent(
        self, feature_id=FEATURE_ID, feature_schema=FEATURE_SCHEMA, actions=ACTIONS,
        algorithm=ALGORITHM, hyperparameters=HYPERPARAMETERS,
        network_spec=NETWORK_SPEC, default_model=MODEL_FILE,
        learner_factory=lambda seed: build_learner(seed, HYPERPARAMETERS),
    )


def act(self, game_state):
    return act_neural(
        self, game_state, actions=ACTIONS, hyperparameters=HYPERPARAMETERS,
        extractor=features_for_state, state_value=lambda features: features.vector,
    )


def state_to_features(game_state):
    result = features_for_state(game_state)
    return None if result is None else result.vector


def legal_actions(game_state):
    result = features_for_state(game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else result.legal_mask.copy()


def _features_for(self, game_state):
    return cached_features(self, game_state, features_for_state)

from pathlib import Path

import numpy as np

from agent_code.learning_common.neural_agent import (
    act_neural, cached_features, setup_neural_agent,
)
from agent_code.learning_common.temporal_reward import (
    advance_frozen_temporal_state, prepare_frozen_temporal_state,
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
    "id": "continuous-mlp-128x2-v2", "input_shape": [84],
    "layers": ["Linear(84,128)", "ReLU", "Linear(128,128)", "ReLU", "Linear(128,6)"],
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
    if not self.train:
        prepare_frozen_temporal_state(self, game_state)
    extractor = lambda state: features_for_state(
        state, previous_action=getattr(self, "previous_action", None),
        previous_position=getattr(self, "reward_previous_position", None),
        previous_coin_target=getattr(self, "reward_previous_coin_target", None))
    action = act_neural(
        self, game_state, actions=ACTIONS, hyperparameters=HYPERPARAMETERS,
        extractor=extractor, state_value=lambda features: features.vector,
    )
    if not self.train:
        advance_frozen_temporal_state(self, action, game_state)
    return action


def state_to_features(game_state):
    result = features_for_state(game_state)
    return None if result is None else result.vector


def legal_actions(game_state):
    result = features_for_state(game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else result.legal_mask.copy()


def _features_for(self, game_state):
    return cached_features(self, game_state, lambda state: features_for_state(
        state, previous_action=getattr(self, "previous_action", None),
        previous_position=getattr(self, "reward_previous_position", None),
        previous_coin_target=getattr(self, "reward_previous_coin_target", None)))

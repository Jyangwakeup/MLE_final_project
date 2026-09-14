from pathlib import Path

import numpy as np

import os

import torch

from agent_code.learning_common.neural_agent import act_neural
from agent_code.learning_common.runtime import (
    adopt_checkpoint_reward, adopt_checkpoint_safety, load_common_configuration,
)
from agent_code.team_agent.feature_system import ACTIONS

from .features import (
    BOARD_SHAPE, FEATURE_ID, FEATURE_SCHEMA, features_for_state,
    initialize_history, record_position,
)
from .model import build_learner


ALGORITHM = "cnn_double_dqn"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = {
    "gamma": 0.95, "learning_rate": 1e-4, "batch_size": 64,
    "replay_capacity": 20_000, "warmup": 5_000,
    "target_sync_interval": 2_000, "gradient_clip": 10.0,
    "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_action_steps": 80_000,
}
NETWORK_SPEC = {
    "id": "path-spatial-residual-v1", "input_shape": list(BOARD_SHAPE),
    "channels": 64, "dilations": [1, 2, 4, 8],
    "pooling": ["self_location", "global_average", "global_max"],
    "dueling": False, "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "feature_schema": FEATURE_SCHEMA, "checkpoint_name": MODEL_FILE.name,
    "network_spec": NETWORK_SPEC, "hyperparameters": HYPERPARAMETERS,
}


def _extract(owner):
    return lambda state: features_for_state(owner, state)


def setup(self):
    initialize_history(self)
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    self.model = build_learner(self.agent_seed, HYPERPARAMETERS)
    if not self.model_file.exists():
        if not self.train:
            raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")
        return
    checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
    adopt_checkpoint_reward(self, checkpoint)
    adopt_checkpoint_safety(self, checkpoint)
    required = {
        "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
        "feature_schema": FEATURE_SCHEMA, "actions": list(ACTIONS),
        "hyperparameters": HYPERPARAMETERS, "network_spec": NETWORK_SPEC,
    }
    for name, expected in required.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"path CNN checkpoint has incompatible {name}")
    if checkpoint.get("reward_id", checkpoint.get("reward_version")) != self.reward_id:
        raise ValueError("path CNN checkpoint has incompatible reward ID")
    self.model.load_checkpoint(checkpoint, training=self.train)
    self.total_action_steps = int(checkpoint.get("total_action_steps", checkpoint["action_steps"]))
    self.stage_action_steps = int(checkpoint.get("stage_action_steps", self.total_action_steps))
    self.action_steps = self.total_action_steps
    if self.train:
        self.rng.setstate(checkpoint["agent_rng_state"])


def act(self, game_state):
    action = act_neural(
        self, game_state, actions=ACTIONS, hyperparameters=HYPERPARAMETERS,
        extractor=_extract(self), state_value=lambda value: value.board,
    )
    record_position(self, game_state)
    return action


def state_to_features(game_state):
    class EmptyHistory:
        cnn_path_history = ()
    result = features_for_state(EmptyHistory(), game_state)
    return None if result is None else result.board


def legal_actions(game_state):
    class EmptyHistory:
        cnn_path_history = ()
    result = features_for_state(EmptyHistory(), game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else result.legal_mask.copy()

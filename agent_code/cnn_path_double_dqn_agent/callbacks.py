from pathlib import Path

import numpy as np

import os

import torch

from agent_code.learning_common.runtime import (
    adopt_checkpoint_reward, adopt_checkpoint_safety, effective_legal_mask,
    load_common_configuration,
)
from agent_code.learning_common.action_history import record_selected_action
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.safety import (
    mask_for_decision, resolve_safety_spec, survival_diagnostics,
)

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
    if self.train and checkpoint.get("safety_spec") != resolve_safety_spec(self.safety_spec):
        raise ValueError("path CNN checkpoint has incompatible safety specification")
    self.model.load_checkpoint(checkpoint, training=self.train)
    self.total_action_steps = int(checkpoint.get("total_action_steps", checkpoint["action_steps"]))
    self.stage_action_steps = int(checkpoint.get("stage_action_steps", self.total_action_steps))
    self.action_steps = self.total_action_steps
    if self.train:
        self.rng.setstate(checkpoint["agent_rng_state"])


def act(self, game_state):
    features = features_for_state(self, game_state)
    physical = effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    values = self.model.q_values(features.board)
    physical_indices = np.flatnonzero(physical).tolist()
    raw_best = max(float(values[index]) for index in physical_indices)
    raw_index = next(index for index in physical_indices
                     if float(values[index]) == raw_best)
    exploring = False
    if self.train:
        epsilon = epsilon_at(self.stage_action_steps, self.exploration_spec)
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        exploring = self.rng.random() < epsilon
    legal, fallback = mask_for_decision(
        game_state, physical, self.safety_spec,
        allow_bomb=self.curriculum_allows_bomb, exploring=exploring)
    enabled = self.safety_spec["mode"] == "all" or (
        self.safety_spec["mode"] == "exploration" and exploring)
    if enabled:
        self.safety_decisions += 1
        self.safety_fallbacks += int(fallback)
        if exploring:
            self.safe_exploration_decisions += 1
            self.safe_exploration_fallbacks += int(fallback)
    indices = np.flatnonzero(legal).tolist()
    if exploring:
        selected = self.rng.choice(indices)
    else:
        masked = values.copy()
        masked[~legal] = -np.inf
        best = max(float(masked[index]) for index in indices)
        tied = [index for index in indices if float(masked[index]) == best]
        selected = self.rng.choice(tied) if self.train else tied[0]
    intervention = bool(enabled and not fallback and not legal[raw_index])
    self.safety_interventions += int(intervention)
    action = ACTIONS[selected]
    self.last_safety_diagnostic = {
        "raw_action": ACTIONS[raw_index], "selected_action": action,
        "intervened": intervention, "fallback": bool(fallback),
        "physical_mask": physical.astype(bool).tolist(),
        "decision_mask": legal.astype(bool).tolist(),
        **survival_diagnostics(
            game_state, physical, allow_bomb=self.curriculum_allows_bomb),
    }
    record_selected_action(self, game_state, action)
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

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from agent_code.dqn_agent.model import DQN
from agent_code.learning_common.action_history import (
    action_history_for_state, load_action_history_state, record_selected_action,
)
from agent_code.learning_common.runtime import (
    adopt_checkpoint_reward, adopt_checkpoint_safety, effective_legal_mask,
    load_common_configuration, validate_checkpoint,
)
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.safety import mask_for_decision, survival_diagnostics
from .features import ACTIONS, FEATURE_ID, FEATURE_SCHEMA, features_for_state


ALGORITHM = "double_dqn"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = {
    "gamma": 0.95, "learning_rate": 3e-4, "batch_size": 64,
    "replay_capacity": 20_000, "warmup": 2_000,
    "target_sync_interval": 1_000, "gradient_clip": 10.0,
}
NETWORK_SPEC = {
    "id": "continuous-v4-mlp-128x2", "input_shape": [126],
    "layers": [
        "Linear(126,128)", "ReLU", "Linear(128,128)", "ReLU",
        "Linear(128,6)",
    ],
    "dueling": False, "double_dqn": True, "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": NETWORK_SPEC,
    "hyperparameters": HYPERPARAMETERS,
}


def setup(self):
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    self.model = DQN(
        126, len(ACTIONS), seed=self.agent_seed,
        gamma=HYPERPARAMETERS["gamma"],
        learning_rate=HYPERPARAMETERS["learning_rate"],
        batch_size=HYPERPARAMETERS["batch_size"],
        replay_capacity=HYPERPARAMETERS["replay_capacity"],
        warmup=HYPERPARAMETERS["warmup"],
        target_sync_interval=HYPERPARAMETERS["target_sync_interval"],
        device=os.getenv("BOMBERMAN_TORCH_DEVICE", "cpu"),
        training_task=self.training_task, retention_spec=self.retention_spec,
        double_dqn=True, hidden_size=128,
    )
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
        adopt_checkpoint_reward(self, checkpoint)
        adopt_checkpoint_safety(self, checkpoint)
        validate_checkpoint(
            checkpoint, algorithm=ALGORITHM, feature_id=FEATURE_ID,
            feature_schema=FEATURE_SCHEMA, actions=ACTIONS,
            reward_id=self.reward_id, hyperparameters=HYPERPARAMETERS,
            network_spec=NETWORK_SPEC, training=self.train,
            training_task=self.training_task, safety_spec=self.safety_spec,
        )
        self.model.load_checkpoint(
            checkpoint, training=self.train, training_task=self.training_task)
        self.total_action_steps = int(checkpoint.get(
            "total_action_steps", checkpoint.get("action_steps", 0)))
        same_task = checkpoint.get("training_task") == self.training_task
        self.stage_action_steps = int(checkpoint.get(
            "stage_action_steps", self.total_action_steps)) if same_task else 0
        self.action_steps = self.total_action_steps
        self.safe_exploration_decisions = int(checkpoint.get(
            "safe_exploration_decisions", 0))
        self.safe_exploration_fallbacks = int(checkpoint.get(
            "safe_exploration_fallbacks", 0))
        self.safety_decisions = int(checkpoint.get("safety_decisions", 0))
        self.safety_interventions = int(checkpoint.get("safety_interventions", 0))
        self.safety_fallbacks = int(checkpoint.get("safety_fallbacks", 0))
        self._resume_n_step_state = checkpoint.get("n_step_state") if same_task else None
        if self.train:
            self.rng.setstate(checkpoint["agent_rng_state"])
            if same_task:
                load_action_history_state(self, checkpoint.get("action_history_state"))
    elif not self.train:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")


def _features_for(self, game_state):
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        previous_action, wait_streak = action_history_for_state(self, game_state)
        self._feature_cache_key = key
        self._feature_cache_value = features_for_state(
            self, game_state, previous_action, wait_streak)
    return self._feature_cache_value


def act(self, game_state):
    from agent_code.learning_common.action_history import advance_observation
    advance_observation(self, game_state)
    features = _features_for(self, game_state)
    physical = effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    values = self.model.q_values(features.vector)
    physical_indices = np.flatnonzero(physical).tolist()
    raw_best = max(float(values[index]) for index in physical_indices)
    raw_tied = [index for index in physical_indices if float(values[index]) == raw_best]
    raw_index = raw_tied[0]
    exploring = False
    if self.train:
        epsilon = epsilon_at(self.stage_action_steps, self.exploration_spec)
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        exploring = self.rng.random() < epsilon
    decision, fallback = mask_for_decision(
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
    indices = np.flatnonzero(decision).tolist()
    if exploring:
        selected = self.rng.choice(indices)
    else:
        masked = values.copy()
        masked[~decision] = -np.inf
        best = max(float(masked[index]) for index in indices)
        tied = [index for index in indices if float(masked[index]) == best]
        selected = self.rng.choice(tied) if self.train else tied[0]
    intervention = bool(enabled and not fallback and not decision[raw_index])
    self.safety_interventions += int(intervention)
    action = ACTIONS[selected]
    self.last_safety_diagnostic = {
        "raw_action": ACTIONS[raw_index], "selected_action": action,
        "intervened": intervention, "fallback": bool(fallback),
        "physical_mask": physical.astype(bool).tolist(),
        "decision_mask": decision.astype(bool).tolist(),
        **survival_diagnostics(
            game_state, physical, allow_bomb=self.curriculum_allows_bomb),
    }
    record_selected_action(self, game_state, action)
    return action


def state_to_features(game_state):
    return None if game_state is None else features_for_state(
        type("FeatureOwner", (), {})(), game_state).vector


def legal_actions(game_state):
    result = state_to_features(game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else features_for_state(
        type("FeatureOwner", (), {})(), game_state).legal_mask.copy()

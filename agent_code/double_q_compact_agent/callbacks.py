from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from agent_code.learning_common.runtime import (
    CHECKPOINT_SCHEMA, adopt_checkpoint_reward, epsilon_at,
    load_common_configuration, validate_checkpoint,
)
from .features import ACTIONS, FEATURE_ID, FEATURE_SCHEMA, canonical_legal_mask, features_for_state


ALGORITHM = "double_q_learning"
MODEL_FILE = Path(__file__).with_name("final.pkl")
HYPERPARAMETERS = {
    "learning_rate": 0.1, "gamma": 0.95,
    "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_action_steps": 80_000,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": None,
    "hyperparameters": HYPERPARAMETERS,
}


def setup(self):
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    self.q_table_a = {}
    self.q_table_b = {}
    self.unseen_q_states = 0
    self.q_decisions = 0
    self.union_q_states = 0
    if self.model_file.exists():
        with self.model_file.open("rb") as file:
            checkpoint = pickle.load(file)
        adopt_checkpoint_reward(self, checkpoint)
        validate_checkpoint(
            checkpoint, algorithm=ALGORITHM, feature_id=FEATURE_ID,
            feature_schema=FEATURE_SCHEMA, actions=ACTIONS, reward_id=self.reward_id,
            hyperparameters=HYPERPARAMETERS, network_spec=None, training=self.train,
            training_task=self.training_task,
        )
        self.q_table_a = checkpoint["q_table_a"]
        self.q_table_b = checkpoint["q_table_b"]
        self.union_q_states = len(self.q_table_a.keys() | self.q_table_b.keys())
        self.action_steps = int(checkpoint["training_steps"])
        if self.train:
            self.rng.setstate(checkpoint["rng_state"])
        self.logger.info("Loaded Double Q checkpoint from %s", self.model_file)
    elif self.train:
        self.logger.info("Starting a new Double Q table")
    else:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")


def _features_for(self, game_state):
    if game_state is None:
        return None
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        self._feature_cache_key = key
        self._feature_cache_value = features_for_state(game_state)
    return self._feature_cache_value


def _values(table, state_key):
    return table.get(state_key, np.zeros(len(ACTIONS), dtype=np.float32))


def act(self, game_state: dict) -> str:
    features = _features_for(self, game_state)
    legal = canonical_legal_mask(
        features, allow_bomb=self.curriculum_allows_bomb)
    legal_indices = np.flatnonzero(legal).tolist()
    if not legal_indices:
        raise ValueError("compact agent has no legal action")
    known = features.state_key in self.q_table_a or features.state_key in self.q_table_b
    self.q_decisions += 1
    if not known:
        self.unseen_q_states += 1
    if self.train:
        epsilon = epsilon_at(self.action_steps, HYPERPARAMETERS)
        self.action_steps += 1
        if self.rng.random() < epsilon:
            canonical_index = self.rng.choice(legal_indices)
            world_index = features.action_transform.canonical_to_world[canonical_index]
            return ACTIONS[world_index]
    values = _values(self.q_table_a, features.state_key) + _values(
        self.q_table_b, features.state_key)
    best = max(float(values[index]) for index in legal_indices)
    tied = [index for index in legal_indices if float(values[index]) == best]
    canonical_index = self.rng.choice(tied) if self.train else tied[0]
    world_index = features.action_transform.canonical_to_world[canonical_index]
    return ACTIONS[world_index]


def state_to_features(game_state):
    result = features_for_state(game_state)
    return None if result is None else result.state_key


def legal_actions(game_state):
    result = features_for_state(game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else result.legal_mask.copy()

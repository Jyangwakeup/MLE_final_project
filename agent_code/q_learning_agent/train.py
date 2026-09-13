"""Training callbacks for one-step tabular Q-learning."""

import csv
import os
from pathlib import Path
import pickle
from typing import List

import numpy as np

from agent_code.learning_common.temporal_reward import (
    reset_temporal_reward_state, temporal_reward_context,
)
from agent_code.team_agent.exploration import epsilon_at, resolve_exploration_spec
from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.rewards import (
    REWARD_VERSION, resolve_reward_spec, reward_from_events,
)
from .callbacks import ACTIONS, CHECKPOINT_SCHEMA_VERSION, MODEL_FILE, _features_for

LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95
TRAINING_FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "epsilon", "q_states", "loss", "updates", "checkpoint",
)


def setup_training(self):
    self.round_reward = 0.0
    reset_temporal_reward_state(self)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTIONS:
        return
    old_features = _features_for(self, old_game_state)
    reward_context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, terminal=False, **reward_context)
    new_features = _features_for(self, new_game_state)
    next_legal = new_features.legal_mask.copy()
    if not self.allow_bomb:
        next_legal[ACTIONS.index("BOMB")] = False
    _q_update(self, old_features.state_key, ACTIONS.index(self_action),
              reward, new_features.state_key, next_legal, terminal=False)
    self.round_reward += reward


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action in ACTIONS:
        reward = reward_from_events(
            events, self.reward_id, old_game_state=last_game_state,
            new_game_state=None, terminal=True)
        _q_update(self, _features_for(self, last_game_state).state_key,
                  ACTIONS.index(last_action), reward, None, None, terminal=True)
        self.round_reward += reward
    reset_temporal_reward_state(self)
    feature_version = "v1" if self.feature_id == "discrete-v1" else None
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "q_learning",
        "actions": list(ACTIONS),
        "agent_seed": self.agent_seed,
        "agent_rng_state": self.rng.getstate(),
        "exploration_spec": self.exploration_spec,
        "feature_version": feature_version,
        "feature_id": self.feature_id,
        "feature_schema": feature_schema_contract(self.feature_id),
        "reward_id": self.reward_id,
        "reward_version": self.reward_id,
        "reward_spec": self.reward_spec,
        "network_spec": None,
        "hyperparameters": {},
        "q_table": self.q_table,
        "training_steps": self.training_steps,
        "training_task": self.training_task,
        "training_device_name": None,
        "training_device_type": "cpu",
    }
    self.model_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.model_file.with_name(self.model_file.name + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(self.model_file)
    _append_training_metrics(self, last_game_state)
    self.logger.info("Round reward %.2f; Q table contains %d states",
                     self.round_reward, len(self.q_table))
    self.round_reward = 0.0


def _append_training_metrics(self, last_game_state) -> None:
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    path = Path(run_dir) / "training.csv"
    record = {
        "schema_version": "training-v1",
        "algorithm": "q_learning",
        "round": "" if last_game_state is None else last_game_state.get("round", ""),
        "reward": self.round_reward,
        "action_steps": self.training_steps,
        "epsilon": epsilon_at(self.training_steps, self.exploration_spec),
        "q_states": len(self.q_table),
        "loss": "",
        "updates": "",
        "checkpoint": str(self.model_file),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _q_update(self, state, action, reward, next_state, next_legal, terminal):
    values = self.q_table.setdefault(state, np.zeros(len(ACTIONS), dtype=np.float32))
    future = 0.0
    if not terminal and next_state is not None:
        next_values = self.q_table.setdefault(
            next_state, np.zeros(len(ACTIONS), dtype=np.float32))
        legal_indices = np.flatnonzero(next_legal)
        if legal_indices.size:
            future = float(np.max(next_values[legal_indices]))
    target = reward + DISCOUNT_FACTOR * future
    values[action] += LEARNING_RATE * (target - float(values[action]))

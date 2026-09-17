from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from agent_code.learning_common.action_history import (
    action_history_for_state, load_action_history_state, record_selected_action,
)
from agent_code.learning_common.runtime import (
    CHECKPOINT_SCHEMA, adopt_checkpoint_reward, adopt_checkpoint_safety,
    load_common_configuration, validate_checkpoint,
)
from agent_code.team_agent.exploration import epsilon_at, survivable_exploration_mask
from .features import ACTIONS, FEATURE_ID, FEATURE_SCHEMA, features_for_state


ALGORITHM = "double_q_learning"
MODEL_FILE = Path(__file__).with_name("final.pkl")
HYPERPARAMETERS = {"learning_rate": 0.1, "gamma": 0.95}
AGENT_METADATA = {
    "algorithm": ALGORITHM,
    "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name,
    "network_spec": None,
    "hyperparameters": HYPERPARAMETERS,
}


def setup(self):
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    self.q_table_a = {}
    self.q_table_b = {}
    self.unseen_q_states = 0
    self.q_decisions = 0
    if self.model_file.exists():
        with self.model_file.open("rb") as file:
            checkpoint = pickle.load(file)
        adopt_checkpoint_reward(self, checkpoint)
        adopt_checkpoint_safety(self, checkpoint)
        validate_checkpoint(
            checkpoint, algorithm=ALGORITHM, feature_id=FEATURE_ID,
            feature_schema=FEATURE_SCHEMA, actions=ACTIONS,
            reward_id=self.reward_id, hyperparameters=HYPERPARAMETERS,
            network_spec=None, training=self.train,
            training_task=self.training_task, safety_spec=self.safety_spec,
        )
        self.q_table_a = checkpoint["q_table_a"]
        self.q_table_b = checkpoint["q_table_b"]
        self.total_action_steps = int(checkpoint.get(
            "total_action_steps", checkpoint.get("training_steps", 0)))
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
        self._resume_n_step_state = (
            checkpoint.get("n_step_state") if same_task else None)
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
            game_state, previous_action, wait_streak)
    return self._feature_cache_value


def _values(table, state):
    return table.get(state, np.zeros(len(ACTIONS), dtype=np.float32))


def act(self, game_state):
    from agent_code.learning_common.action_history import advance_observation
    advance_observation(self, game_state)
    features = _features_for(self, game_state)
    legal = features.legal_mask.copy()
    if not self.curriculum_allows_bomb:
        legal[ACTIONS.index("BOMB")] = False
    indices = np.flatnonzero(legal).tolist()
    known = features.state_key in self.q_table_a or features.state_key in self.q_table_b
    self.q_decisions += 1
    self.unseen_q_states += int(not known)
    explore = False
    if self.train:
        epsilon = epsilon_at(self.stage_action_steps, self.exploration_spec)
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        explore = self.rng.random() < epsilon or not known
    elif not known:
        explore = True
    if explore:
        if self.train and self.safe_exploration:
            legal, fallback = survivable_exploration_mask(
                game_state, legal, allow_bomb=self.curriculum_allows_bomb)
            indices = np.flatnonzero(legal).tolist()
            self.safe_exploration_decisions += 1
            self.safe_exploration_fallbacks += int(fallback)
        action = ACTIONS[self.rng.choice(indices)]
    else:
        values = _values(self.q_table_a, features.state_key) + _values(
            self.q_table_b, features.state_key)
        best = max(float(values[index]) for index in indices)
        tied = [index for index in indices if float(values[index]) == best]
        action = ACTIONS[self.rng.choice(tied) if self.train else tied[0]]
    record_selected_action(self, game_state, action)
    return action


def state_to_features(game_state):
    result = features_for_state(game_state)
    return None if result is None else result.state_key


def legal_actions(game_state):
    result = features_for_state(game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else result.legal_mask.copy()

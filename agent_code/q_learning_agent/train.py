"""Training callbacks for one-step tabular Q-learning."""

import pickle
from typing import List

import numpy as np

import events as e
import settings as s
from .callbacks import ACTIONS, MODEL_FILE, legal_actions, state_to_features


LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95


def setup_training(self):
    self.round_reward = 0.0


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTIONS:
        return
    reward = reward_from_events(events)
    _q_update(self, state_to_features(old_game_state), ACTIONS.index(self_action),
              reward, state_to_features(new_game_state),
              legal_actions(new_game_state), terminal=False)
    self.round_reward += reward


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action in ACTIONS:
        reward = reward_from_events(events)
        _q_update(self, state_to_features(last_game_state), ACTIONS.index(last_action),
                  reward, None, None, terminal=True)
        self.round_reward += reward

    payload = {"q_table": self.q_table, "training_steps": self.training_steps}
    with MODEL_FILE.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    self.logger.info("Round reward %.2f; Q table contains %d states",
                     self.round_reward, len(self.q_table))
    self.round_reward = 0.0


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


def reward_from_events(events: List[str]) -> float:
    """Translate environment outcomes into scalar reinforcement."""
    reward = -0.01
    reward += events.count(e.COIN_COLLECTED) * s.REWARD_COIN
    reward += events.count(e.KILLED_OPPONENT) * s.REWARD_KILL
    reward += events.count(e.CRATE_DESTROYED) * 0.2
    reward += events.count(e.SURVIVED_ROUND) * 1.0
    reward -= events.count(e.INVALID_ACTION) * 0.2
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 10.0
    return float(reward)

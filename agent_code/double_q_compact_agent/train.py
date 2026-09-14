from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA, epsilon_at, save_checkpoint_atomic
from agent_code.team_agent.rewards import reward_from_events
from agent_code.learning_common.temporal_reward import (
    DIAGNOSTIC_COUNT_FIELDS, accumulate_reward_diagnostics,
    observed_terminal_state, reset_reward_diagnostics,
    reset_temporal_reward_state, temporal_reward_context,
)
from .callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, _features_for,
)
from .features import canonical_legal_mask


class Transition(NamedTuple):
    state: tuple[int, ...]
    action: int
    reward: float
    next_state: tuple[int, ...] | None
    done: bool
    next_legal: np.ndarray | None


TRAINING_FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps", "epsilon",
    "q_states", "q_states_a", "q_states_b", "union_q_states", "unseen_q_states",
    "unseen_q_state_rate", "loss", "updates", "checkpoint",
    *DIAGNOSTIC_COUNT_FIELDS, "conditional_loop_reward", "avoidable_wait_reward",
)


def setup_training(self):
    self.round_reward = 0.0
    self.pending = None
    self.ended_key = None
    reset_temporal_reward_state(self)
    reset_reward_diagnostics(self)


def _argmax(values, legal, rng):
    indices = np.flatnonzero(legal).tolist()
    if not indices:
        raise ValueError("Double Q next state has no legal action")
    best = max(float(values[index]) for index in indices)
    return rng.choice([index for index in indices if float(values[index]) == best])


def update_double_q(self, transition: Transition) -> None:
    update_a = self.rng.random() < 0.5
    selected = self.q_table_a if update_a else self.q_table_b
    evaluator = self.q_table_b if update_a else self.q_table_a
    if transition.state not in self.q_table_a and transition.state not in self.q_table_b:
        self.union_q_states = getattr(self, "union_q_states", 0) + 1
    current = selected.setdefault(
        transition.state, np.zeros(len(ACTIONS), dtype=np.float32))
    target = transition.reward
    if not transition.done:
        select_values = selected.get(
            transition.next_state, np.zeros(len(ACTIONS), dtype=np.float32))
        evaluate_values = evaluator.get(
            transition.next_state, np.zeros(len(ACTIONS), dtype=np.float32))
        next_action = _argmax(select_values, transition.next_legal, self.rng)
        target += HYPERPARAMETERS["gamma"] * float(evaluate_values[next_action])
    action = transition.action
    current[action] += HYPERPARAMETERS["learning_rate"] * (target - current[action])


def _make_transition(self, old_state, action, reward, new_state, done):
    old_features = _features_for(self, old_state)
    world_index = ACTIONS.index(action)
    canonical_action = old_features.action_transform.world_to_canonical[world_index]
    if done:
        return Transition(old_features.state_key, canonical_action, reward, None, True, None)
    new_features = _features_for(self, new_state)
    return Transition(
        old_features.state_key, canonical_action, reward, new_features.state_key, False,
        canonical_legal_mask(new_features, allow_bomb=self.curriculum_allows_bomb),
    )


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if key == self.ended_key:
        return
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    _features_for(self, old_game_state)
    reward_context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events,
        reward_id=self.reward_id)
    diagnostic = reward_context.pop("diagnostic")
    if diagnostic:
        accumulate_reward_diagnostics(self, diagnostic, self.reward_spec)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, terminal=False, **reward_context)
    self.pending = (key, _make_transition(
        self, old_game_state, self_action, reward, new_game_state, False))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if key != self.ended_key:
            if self.pending is not None and self.pending[0] != key:
                _submit(self, self.pending[1])
            reward_context = {}
            if self.reward_id == "r5_conditional_loop":
                _features_for(self, last_game_state)
                reward_context = temporal_reward_context(
                    self, last_action, last_game_state,
                    observed_terminal_state(last_game_state, last_action, events),
                    events, reward_id=self.reward_id)
                diagnostic = reward_context.pop("diagnostic")
                accumulate_reward_diagnostics(self, diagnostic, self.reward_spec)
            reward = reward_from_events(
                events, self.reward_id, old_game_state=last_game_state,
                new_game_state=None, terminal=True, **reward_context)
            _submit(self, _make_transition(
                self, last_game_state, last_action, reward, None, True))
            self.ended_key = key
    elif self.pending is not None:
        _submit(self, self.pending[1])
    self.pending = None
    reset_temporal_reward_state(self)
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA, "algorithm": ALGORITHM,
        "actions": list(ACTIONS), "feature_id": FEATURE_ID,
        "feature_schema": FEATURE_SCHEMA, "reward_id": self.reward_id,
        "reward_version": self.reward_id, "reward_spec": self.reward_spec,
        "hyperparameters": HYPERPARAMETERS, "network_spec": None,
        "q_table_a": self.q_table_a, "q_table_b": self.q_table_b,
        "training_steps": self.action_steps, "action_steps": self.action_steps,
        "rng_state": self.rng.getstate(), "agent_rng_state": self.rng.getstate(),
        "agent_seed": getattr(self, "agent_seed", 0),
        "exploration_spec": getattr(self, "exploration_spec", {}),
        "training_device_name": None, "training_device_type": "cpu",
        "training_task": self.training_task,
    }
    save_checkpoint_atomic(self.model_file, checkpoint, _save_pickle)
    _append_metrics(self, last_game_state)
    self.round_reward = 0.0


def _save_pickle(payload, path):
    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def _submit(self, transition):
    update_double_q(self, transition)
    self.round_reward += transition.reward


def _append_metrics(self, last_game_state):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        reset_reward_diagnostics(self)
        return
    path = Path(run_dir) / "training.csv"
    union = self.union_q_states
    record = {
        "schema_version": "training-v1", "algorithm": ALGORITHM,
        "round": "" if last_game_state is None else last_game_state.get("round", ""),
        "reward": self.round_reward, "action_steps": self.action_steps,
        "epsilon": epsilon_at(self.action_steps, HYPERPARAMETERS), "q_states": union,
        "q_states_a": len(self.q_table_a), "q_states_b": len(self.q_table_b),
        "union_q_states": union, "unseen_q_states": self.unseen_q_states,
        "unseen_q_state_rate": self.unseen_q_states / max(self.q_decisions, 1),
        "loss": "", "updates": "", "checkpoint": str(self.model_file),
        **self.reward_diagnostics,
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    reset_reward_diagnostics(self)

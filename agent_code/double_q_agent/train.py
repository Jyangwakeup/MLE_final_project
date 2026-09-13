from __future__ import annotations

import csv
import os
import pickle
from typing import NamedTuple

import numpy as np

from agent_code.learning_common.action_history import action_history_state, init_action_history
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA, save_checkpoint_atomic
from agent_code.learning_common.temporal_reward import reset_temporal_reward_state, temporal_reward_context
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.rewards import reward_from_events
from .callbacks import ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, _features_for


class Transition(NamedTuple):
    state: tuple[int, ...]
    action: int
    reward: float
    next_state: tuple[int, ...] | None
    done: bool
    next_legal: np.ndarray | None
    steps: int = 1


FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "stage_action_steps", "epsilon", "q_states", "unseen_q_state_rate",
    "loss", "updates", "safe_exploration_decisions",
    "safe_exploration_fallbacks", "checkpoint",
)


def setup_training(self):
    self.round_reward = 0.0
    self.accumulator = NStepAccumulator(self.n_step, HYPERPARAMETERS["gamma"])
    self.accumulator.load_state_dict(getattr(self, "_resume_n_step_state", None))
    reset_temporal_reward_state(self)


def _argmax(values, legal, rng):
    indices = np.flatnonzero(legal).tolist()
    best = max(float(values[index]) for index in indices)
    return rng.choice([index for index in indices if float(values[index]) == best])


def _update(self, transition):
    update_a = self.rng.random() < 0.5
    selected = self.q_table_a if update_a else self.q_table_b
    evaluator = self.q_table_b if update_a else self.q_table_a
    current = selected.setdefault(
        transition.state, np.zeros(len(ACTIONS), dtype=np.float32))
    target = transition.reward
    if not transition.done:
        selected_next = selected.get(
            transition.next_state, np.zeros(len(ACTIONS), dtype=np.float32))
        evaluator_next = evaluator.get(
            transition.next_state, np.zeros(len(ACTIONS), dtype=np.float32))
        action = _argmax(selected_next, transition.next_legal, self.rng)
        target += (HYPERPARAMETERS["gamma"] ** transition.steps) * float(
            evaluator_next[action])
    current[transition.action] += HYPERPARAMETERS["learning_rate"] * (
        target - float(current[transition.action]))


def _submit(self, transition):
    for aggregate in self.accumulator.append(transition):
        _update(self, aggregate)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in ACTIONS:
        return
    old = _features_for(self, old_game_state)
    context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, action=self_action, **context)
    new = _features_for(self, new_game_state)
    legal = new.legal_mask.copy()
    if not self.curriculum_allows_bomb:
        legal[ACTIONS.index("BOMB")] = False
    _submit(self, Transition(
        old.state_key, ACTIONS.index(self_action), reward,
        new.state_key, False, legal))
    self.round_reward += reward


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None and last_action in ACTIONS:
        reward = reward_from_events(
            events, self.reward_id, old_game_state=last_game_state,
            terminal=True, action=last_action)
        _submit(self, Transition(
            _features_for(self, last_game_state).state_key,
            ACTIONS.index(last_action), reward, None, True, None))
        self.round_reward += reward
    reset_temporal_reward_state(self)
    init_action_history(self)
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "algorithm": ALGORITHM,
        "actions": list(ACTIONS),
        "feature_id": FEATURE_ID,
        "feature_schema": FEATURE_SCHEMA,
        "feature_version": None,
        "reward_id": self.reward_id,
        "reward_version": self.reward_id,
        "reward_spec": self.reward_spec,
        "hyperparameters": HYPERPARAMETERS,
        "network_spec": None,
        "q_table_a": self.q_table_a,
        "q_table_b": self.q_table_b,
        "training_steps": self.total_action_steps,
        "action_steps": self.total_action_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "agent_rng_state": self.rng.getstate(),
        "rng_state": self.rng.getstate(),
        "agent_seed": self.agent_seed,
        "exploration_spec": self.exploration_spec,
        "safe_exploration": self.safe_exploration,
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "action_history_state": action_history_state(self),
        "n_step": self.n_step,
        "n_step_state": self.accumulator.state_dict(),
        "retention_spec": self.retention_spec,
        "training_budget": self.training_budget,
        "training_task": self.training_task,
        "training_device_name": None,
        "training_device_type": "cpu",
    }
    save_checkpoint_atomic(self.model_file, payload, _save)
    _append_metrics(self, last_game_state)
    self.round_reward = 0.0


def _save(payload, path):
    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def _append_metrics(self, state):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    path = os.path.join(run_dir, "training.csv")
    states = len(self.q_table_a.keys() | self.q_table_b.keys())
    row = {
        "schema_version": "training-v2", "algorithm": ALGORITHM,
        "round": "" if state is None else state.get("round", ""),
        "reward": self.round_reward, "action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "epsilon": epsilon_at(self.stage_action_steps, self.exploration_spec),
        "q_states": states,
        "unseen_q_state_rate": self.unseen_q_states / max(self.q_decisions, 1),
        "loss": "", "updates": "",
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "checkpoint": str(self.model_file),
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

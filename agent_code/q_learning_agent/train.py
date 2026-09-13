"""Training callbacks for one-step tabular Q-learning."""

import csv
import os
from pathlib import Path
import pickle
from typing import List, NamedTuple

import numpy as np

from agent_code.learning_common.temporal_reward import (
    reset_temporal_reward_state, temporal_reward_context,
)
from agent_code.learning_common.action_history import (
    action_history_state, init_action_history,
)
from agent_code.learning_common.n_step import NStepAccumulator
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
    "stage_action_steps", "safe_exploration_decisions",
    "safe_exploration_fallbacks",
)


class Transition(NamedTuple):
    state: tuple[int, ...]
    action: int
    reward: float
    next_state: tuple[int, ...] | None
    done: bool
    next_legal: np.ndarray | None
    steps: int = 1


def setup_training(self):
    self.round_reward = 0.0
    self.pending = None
    self.ended_key = None
    self.n_step = int(getattr(self, "n_step", 1))
    self.n_step_accumulator = NStepAccumulator(self.n_step, DISCOUNT_FACTOR)
    self.n_step_accumulator.load_state_dict(
        getattr(self, "_resume_n_step_state", None))
    reset_temporal_reward_state(self)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if key == self.ended_key:
        return
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    old_features = _features_for(self, old_game_state)
    reward_context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, terminal=False, action=self_action,
        **reward_context)
    new_features = _features_for(self, new_game_state)
    next_legal = new_features.legal_mask.copy()
    if not self.allow_bomb:
        next_legal[ACTIONS.index("BOMB")] = False
    self.pending = (key, Transition(
        old_features.state_key, ACTIONS.index(self_action), reward,
        new_features.state_key, False, next_legal))
    self.round_reward += reward


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if key != self.ended_key:
            if self.pending is not None and self.pending[0] != key:
                _submit(self, self.pending[1])
            elif self.pending is not None:
                # The framework reports the final action once as an ordinary
                # step and then again with terminal events. Replace, do not
                # double count, that provisional reward.
                self.round_reward -= float(self.pending[1].reward)
            reward = reward_from_events(
                events, self.reward_id, old_game_state=last_game_state,
                new_game_state=None, terminal=True, action=last_action)
            _submit(self, Transition(
                _features_for(self, last_game_state).state_key,
                ACTIONS.index(last_action), reward, None, True, None))
            self.round_reward += reward
            self.ended_key = key
    elif self.pending is not None:
        _submit(self, self.pending[1])
    self.pending = None
    reset_temporal_reward_state(self)
    init_action_history(self)
    self.total_action_steps = int(getattr(
        self, "total_action_steps", getattr(self, "training_steps", 0)))
    self.stage_action_steps = int(getattr(
        self, "stage_action_steps", self.total_action_steps))
    self.safe_exploration = bool(getattr(self, "safe_exploration", False))
    self.safe_exploration_decisions = int(getattr(
        self, "safe_exploration_decisions", 0))
    self.safe_exploration_fallbacks = int(getattr(
        self, "safe_exploration_fallbacks", 0))
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
        "network_spec": self.network_spec,
        "hyperparameters": self.hyperparameters,
        "q_table": self.q_table,
        "training_steps": self.training_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "safe_exploration": self.safe_exploration,
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "action_history_state": action_history_state(self),
        "n_step": self.n_step,
        "n_step_state": self.n_step_accumulator.state_dict(),
        "retention_spec": getattr(self, "retention_spec", {}),
        "training_budget": getattr(self, "training_budget", {
            "target_stage_action_steps": None, "min_rounds": 1}),
        "training_task": getattr(self, "training_task", None),
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
        "action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "epsilon": epsilon_at(self.stage_action_steps, self.exploration_spec),
        "q_states": len(self.q_table),
        "loss": "",
        "updates": "",
        "checkpoint": str(self.model_file),
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _submit(self, transition: Transition) -> None:
    for aggregated in self.n_step_accumulator.append(transition):
        _q_update(self, aggregated)


def _q_update(self, transition: Transition):
    values = self.q_table.setdefault(
        transition.state, np.zeros(len(ACTIONS), dtype=np.float32))
    future = 0.0
    if not transition.done and transition.next_state is not None:
        next_values = self.q_table.setdefault(
            transition.next_state, np.zeros(len(ACTIONS), dtype=np.float32))
        legal_indices = np.flatnonzero(transition.next_legal)
        if legal_indices.size:
            future = float(np.max(next_values[legal_indices]))
    target = transition.reward + (DISCOUNT_FACTOR ** transition.steps) * future
    values[transition.action] += LEARNING_RATE * (
        target - float(values[transition.action]))

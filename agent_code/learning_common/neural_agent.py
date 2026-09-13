"""Callback helpers shared by the three neural learning agents."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.exploration import epsilon_at, survivable_exploration_mask
from .action_history import (
    action_history_state, init_action_history, load_action_history_state,
    record_selected_action,
)
from .neural import Transition
from .runtime import (
    CHECKPOINT_SCHEMA, adopt_checkpoint_reward, effective_legal_mask,
    load_common_configuration, save_checkpoint_atomic, validate_checkpoint,
)
from .temporal_reward import reset_temporal_reward_state, temporal_reward_context


TRAINING_FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "stage_action_steps", "epsilon", "q_states", "loss", "updates",
    "replay_size", "safe_exploration_decisions", "safe_exploration_fallbacks",
    "checkpoint",
)


def setup_neural_agent(
    self, *, feature_id, feature_schema, actions, algorithm, hyperparameters,
    network_spec, default_model, learner_factory,
):
    load_common_configuration(self, feature_id=feature_id, default_model=default_model)
    self.model = learner_factory(self.agent_seed)
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
        adopt_checkpoint_reward(self, checkpoint)
        validate_checkpoint(
            checkpoint, algorithm=algorithm, feature_id=feature_id,
            feature_schema=feature_schema, actions=actions, reward_id=self.reward_id,
            hyperparameters=hyperparameters, network_spec=network_spec,
            training=self.train, training_task=self.training_task,
        )
        self.model.load_checkpoint(checkpoint, training=self.train)
        self.total_action_steps = int(checkpoint.get(
            "total_action_steps", checkpoint["action_steps"]))
        same_task = checkpoint.get("training_task") == self.training_task
        self.stage_action_steps = int(checkpoint.get(
            "stage_action_steps", self.total_action_steps)) if same_task else 0
        self.action_steps = self.total_action_steps
        self.safe_exploration_decisions = int(checkpoint.get(
            "safe_exploration_decisions", 0))
        self.safe_exploration_fallbacks = int(checkpoint.get(
            "safe_exploration_fallbacks", 0))
        if self.train:
            self.rng.setstate(checkpoint["agent_rng_state"])
            if same_task:
                load_action_history_state(self, checkpoint.get("action_history_state"))
        self.logger.info("Loaded %s checkpoint from %s", algorithm, self.model_file)
    elif self.train:
        self.logger.info("Starting a new %s model", algorithm)
    else:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")


def cached_features(self, game_state, extractor):
    if game_state is None:
        return None
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        self._feature_cache_key = key
        self._feature_cache_value = extractor(game_state)
    return self._feature_cache_value


def act_neural(self, game_state, *, actions, hyperparameters, extractor, state_value):
    features = cached_features(self, game_state, extractor)
    legal = effective_legal_mask(
        features.legal_mask, actions, self.curriculum_allows_bomb)
    legal_indices = np.flatnonzero(legal).tolist()
    if self.train:
        stage_steps = int(getattr(
            self, "stage_action_steps", getattr(self, "action_steps", 0)))
        total_steps = int(getattr(
            self, "total_action_steps", getattr(self, "action_steps", 0)))
        epsilon = epsilon_at(
            stage_steps, getattr(self, "exploration_spec", None))
        self.stage_action_steps = stage_steps
        self.total_action_steps = total_steps
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        if self.rng.random() < epsilon:
            if getattr(self, "safe_exploration", False):
                legal, fallback = survivable_exploration_mask(
                    game_state, legal,
                    allow_bomb=self.curriculum_allows_bomb)
                legal_indices = np.flatnonzero(legal).tolist()
                self.safe_exploration_decisions = int(getattr(
                    self, "safe_exploration_decisions", 0)) + 1
                self.safe_exploration_fallbacks = int(getattr(
                    self, "safe_exploration_fallbacks", 0)) + int(fallback)
            action = actions[self.rng.choice(legal_indices)]
            record_selected_action(self, game_state, action)
            return action
    q_values = self.model.q_values(state_value(features))
    q_values[~legal] = -np.inf
    best = max(float(q_values[index]) for index in legal_indices)
    tied = [index for index in legal_indices if float(q_values[index]) == best]
    action = actions[self.rng.choice(tied) if self.train else tied[0]]
    record_selected_action(self, game_state, action)
    return action


def setup_neural_training(self):
    self.round_reward = 0.0
    self.last_loss = None
    self.pending = None
    self.ended_key = None


def _transition(self, old_state, action, reward, new_state, done, *,
                actions, extractor, state_value):
    old_features = cached_features(self, old_state, extractor)
    if done:
        return Transition(
            state_value(old_features), actions.index(action), reward, None, True, None)
    new_features = cached_features(self, new_state, extractor)
    next_legal = effective_legal_mask(
        new_features.legal_mask, actions, self.curriculum_allows_bomb)
    return Transition(
        state_value(old_features), actions.index(action), reward,
        state_value(new_features), False, next_legal,
    )


def neural_game_events(self, old_game_state, self_action, new_game_state, events, *,
                       actions, extractor, state_value):
    if old_game_state is None or self_action not in actions:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if key == self.ended_key:
        return
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    # Preserve the pre-action temporal feature before advancing shared history.
    cached_features(self, old_game_state, extractor)
    reward_context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, terminal=False, action=self_action,
        **reward_context)
    self.pending = (key, _transition(
        self, old_game_state, self_action, reward, new_game_state, False,
        actions=actions, extractor=extractor, state_value=state_value))


def neural_end_round(self, last_game_state, last_action, events, *, actions,
                     extractor, state_value, algorithm, feature_id,
                     feature_schema, hyperparameters, network_spec):
    if last_game_state is not None and last_action in actions:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if key != self.ended_key:
            if self.pending is not None and self.pending[0] != key:
                _submit(self, self.pending[1])
            reward = reward_from_events(
                events, self.reward_id, old_game_state=last_game_state,
                new_game_state=None, terminal=True, action=last_action)
            _submit(self, _transition(
                self, last_game_state, last_action, reward, None, True,
                actions=actions, extractor=extractor, state_value=state_value))
            self.ended_key = key
    elif self.pending is not None:
        _submit(self, self.pending[1])
    self.pending = None
    reset_temporal_reward_state(self)
    init_action_history(self)
    self.total_action_steps = int(getattr(
        self, "total_action_steps", getattr(self, "action_steps", 0)))
    self.stage_action_steps = int(getattr(
        self, "stage_action_steps", self.total_action_steps))
    self.safe_exploration = bool(getattr(self, "safe_exploration", False))
    self.safe_exploration_decisions = int(getattr(
        self, "safe_exploration_decisions", 0))
    self.safe_exploration_fallbacks = int(getattr(
        self, "safe_exploration_fallbacks", 0))
    checkpoint = self.model.checkpoint()
    checkpoint.update({
        "checkpoint_schema": CHECKPOINT_SCHEMA, "algorithm": algorithm,
        "actions": list(actions), "feature_id": feature_id,
        "feature_schema": feature_schema, "reward_id": self.reward_id,
        "reward_version": self.reward_id, "reward_spec": self.reward_spec,
        "hyperparameters": hyperparameters, "network_spec": network_spec,
        "action_steps": self.total_action_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "agent_rng_state": self.rng.getstate(),
        "agent_seed": getattr(self, "agent_seed", 0),
        "exploration_spec": getattr(self, "exploration_spec", {}),
        "safe_exploration": self.safe_exploration,
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "action_history_state": action_history_state(self),
        "n_step": int(getattr(self, "n_step", 1)),
        "n_step_state": {
            "n_step": int(getattr(self, "n_step", 1)),
            "gamma": hyperparameters["gamma"], "pending": []},
        "retention_spec": getattr(self, "retention_spec", {}),
        "training_budget": getattr(self, "training_budget", {
            "target_stage_action_steps": None, "min_rounds": 1}),
        "training_task": getattr(self, "training_task", None),
    })
    save_checkpoint_atomic(self.model_file, checkpoint, torch.save)
    _append_metrics(self, last_game_state, algorithm, hyperparameters)
    self.round_reward = 0.0


def _submit(self, transition):
    self.last_loss = self.model.observe(transition)
    self.round_reward += transition.reward


def _append_metrics(self, last_game_state, algorithm, hyperparameters):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    path = Path(run_dir) / "training.csv"
    record = {
        "schema_version": "training-v1", "algorithm": algorithm,
        "round": "" if last_game_state is None else last_game_state.get("round", ""),
        "reward": self.round_reward, "action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "epsilon": epsilon_at(self.stage_action_steps, self.exploration_spec), "q_states": "",
        "loss": "" if self.last_loss is None else self.last_loss,
        "updates": self.model.updates, "replay_size": len(self.model.replay),
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "checkpoint": str(self.model_file),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

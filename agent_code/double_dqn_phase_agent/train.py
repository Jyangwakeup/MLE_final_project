from __future__ import annotations

import csv
import os

import torch

from agent_code.dqn_agent.model import Transition
from agent_code.learning_common.action_history import action_history_state, init_action_history
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA, effective_legal_mask
from agent_code.learning_common.temporal_reward import reset_temporal_reward_state, temporal_reward_context
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.phase import (
    init_phase_history, phase_facts_for_owner, phase_history_state,
)
from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.safety import avoidable_fatal_action
from .callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS,
    NETWORK_SPEC, _decision_mask, _features_for,
)


FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "stage_action_steps", "epsilon", "q_states", "loss", "updates",
    "replay_size", "safe_exploration_decisions", "safe_exploration_fallbacks",
    "safety_decisions", "safety_interventions", "safety_fallbacks", "checkpoint",
)


def setup_training(self):
    self.round_reward = 0.0
    self.last_loss = None
    self.pending = None
    self.ended_key = None
    self.accumulator = NStepAccumulator(self.n_step, HYPERPARAMETERS["gamma"])
    self.accumulator.load_state_dict(getattr(self, "_resume_n_step_state", None))
    reset_temporal_reward_state(self)


def _physical(self, state, features):
    return effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)


def _transition(self, old_state, action, reward, new_state, done):
    old = _features_for(self, old_state)
    old_mask, _ = _decision_mask(self, old_state, _physical(self, old_state, old), False)
    if done:
        return Transition(old.vector.copy(), ACTIONS.index(action), reward, None,
                          True, None, old_mask, self.training_task, 1)
    new = _features_for(self, new_state)
    next_mask, _ = _decision_mask(self, new_state, _physical(self, new_state, new), False)
    return Transition(old.vector.copy(), ACTIONS.index(action), reward,
                      new.vector.copy(), False, next_mask, old_mask,
                      self.training_task, 1)


def _submit(self, transition):
    for aggregate in self.accumulator.append(transition):
        loss = self.model.observe(aggregate)
        if loss is not None:
            self.last_loss = loss


def _reward(self, old_state, action, new_state, events, terminal=False):
    old_features = _features_for(self, old_state)
    context = temporal_reward_context(self, action, old_state, new_state, events)
    context["avoidable_fatal"] = avoidable_fatal_action(
        old_state, action, _physical(self, old_state, old_features),
        allow_bomb=self.curriculum_allows_bomb)
    if self.reward_id.startswith("r9_phase_"):
        context["old_phase_facts"] = phase_facts_for_owner(self, old_state)
        context["new_phase_facts"] = (
            None if terminal or new_state is None else phase_facts_for_owner(self, new_state))
    return reward_from_events(
        events, self.reward_id, old_game_state=old_state,
        new_game_state=new_state, terminal=terminal, action=action, **context)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    reward = _reward(self, old_game_state, self_action, new_game_state, events)
    self.round_reward += reward
    self.pending = (key, _transition(
        self, old_game_state, self_action, reward, new_game_state, False))


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if self.pending is not None and self.pending[0] != key:
            _submit(self, self.pending[1])
        reward = _reward(self, last_game_state, last_action, None, events, terminal=True)
        self.round_reward += reward
        _submit(self, _transition(
            self, last_game_state, last_action, reward, None, True))
    elif self.pending is not None:
        _submit(self, self.pending[1])
    self.pending = None
    reset_temporal_reward_state(self)
    checkpoint = self.model.checkpoint()
    checkpoint.update({
        "checkpoint_schema": CHECKPOINT_SCHEMA, "algorithm": ALGORITHM,
        "actions": list(ACTIONS), "feature_id": FEATURE_ID,
        "feature_version": None, "feature_schema": FEATURE_SCHEMA,
        "reward_id": self.reward_id, "reward_version": self.reward_id,
        "reward_spec": self.reward_spec, "hyperparameters": HYPERPARAMETERS,
        "network_spec": NETWORK_SPEC, "action_steps": self.total_action_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "agent_seed": self.agent_seed, "agent_rng_state": self.rng.getstate(),
        "exploration_spec": self.exploration_spec,
        "safe_exploration": self.safe_exploration,
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "safety_spec": self.safety_spec,
        "safety_decisions": self.safety_decisions,
        "safety_interventions": self.safety_interventions,
        "safety_fallbacks": self.safety_fallbacks,
        "action_history_state": action_history_state(self),
        "phase_history_state": phase_history_state(self),
        "n_step": self.n_step, "n_step_state": self.accumulator.state_dict(),
        "retention_spec": self.retention_spec,
        "training_budget": self.training_budget,
        "training_task": self.training_task,
        "transfer_contract": getattr(self, "transfer_contract", None),
    })
    temporary = self.model_file.with_name(self.model_file.name + ".tmp")
    self.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, temporary)
    temporary.replace(self.model_file)
    _append_metrics(self, last_game_state)
    self.round_reward = 0.0
    init_action_history(self)
    init_phase_history(self)


def _append_metrics(self, state):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    path = os.path.join(run_dir, "training.csv")
    row = {
        "schema_version": "training-v4", "algorithm": ALGORITHM,
        "round": "" if state is None else state.get("round", ""),
        "reward": self.round_reward, "action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "epsilon": epsilon_at(self.stage_action_steps, self.exploration_spec),
        "q_states": "", "loss": "" if self.last_loss is None else self.last_loss,
        "updates": self.model.updates, "replay_size": len(self.model.replay),
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "safety_decisions": self.safety_decisions,
        "safety_interventions": self.safety_interventions,
        "safety_fallbacks": self.safety_fallbacks,
        "checkpoint": str(self.model_file),
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

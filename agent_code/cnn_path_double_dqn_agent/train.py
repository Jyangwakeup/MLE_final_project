import csv
import os
from pathlib import Path
import shutil

import torch

from agent_code.learning_common.action_history import action_history_state, init_action_history
from agent_code.learning_common.neural import Transition
from agent_code.learning_common.runtime import (
    CHECKPOINT_SCHEMA, effective_legal_mask, save_checkpoint_atomic,
)
from agent_code.learning_common.temporal_reward import (
    accumulate_reward_diagnostics, reset_reward_diagnostics,
    reset_temporal_reward_state, temporal_reward_context,
)
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.safety import avoidable_fatal_action, mask_for_decision

from .callbacks import (
    ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
    cached_features,
)
from .features import reset_history


FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "stage_action_steps", "epsilon", "loss", "updates", "replay_size",
    "safety_decisions", "safety_interventions", "safety_fallbacks", "checkpoint",
)


def setup_training(self):
    self.round_reward = 0.0
    self.last_loss = None
    self.pending = None
    self.ended_key = None
    reset_temporal_reward_state(self)
    reset_reward_diagnostics(self)


def _decision_mask(self, state):
    features = cached_features(self, state)
    physical = effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    legal, _ = mask_for_decision(
        state, physical, self.safety_spec,
        allow_bomb=self.curriculum_allows_bomb, exploring=False)
    return features, physical, legal


def _transition(self, old_state, action, reward, new_state, done):
    old, _, _ = _decision_mask(self, old_state)
    if done:
        return Transition(old.board.copy(), ACTIONS.index(action), reward, None, True, None)
    new, _, next_legal = _decision_mask(self, new_state)
    return Transition(
        old.board.copy(), ACTIONS.index(action), reward, new.board.copy(),
        False, next_legal,
    )


def _submit(self, transition):
    loss = self.model.observe(transition)
    if loss is not None:
        self.last_loss = loss


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if key == self.ended_key:
        return
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    _, physical, _ = _decision_mask(self, old_game_state)
    context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events,
        reward_id=self.reward_id)
    diagnostic = context.pop("diagnostic")
    if diagnostic:
        accumulate_reward_diagnostics(self, diagnostic, self.reward_spec)
    context["avoidable_fatal"] = avoidable_fatal_action(
        old_game_state, self_action, physical,
        allow_bomb=self.curriculum_allows_bomb)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, action=self_action, **context)
    self.round_reward += reward
    self.pending = (key, _transition(
        self, old_game_state, self_action, reward, new_game_state, False))


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if key != self.ended_key:
            if self.pending is not None and self.pending[0] != key:
                _submit(self, self.pending[1])
            _, physical, _ = _decision_mask(self, last_game_state)
            reward = reward_from_events(
                events, self.reward_id, old_game_state=last_game_state,
                terminal=True, action=last_action,
                avoidable_fatal=avoidable_fatal_action(
                    last_game_state, last_action, physical,
                    allow_bomb=self.curriculum_allows_bomb))
            self.round_reward += reward
            _submit(self, _transition(
                self, last_game_state, last_action, reward, None, True))
            self.ended_key = key
    elif self.pending is not None:
        _submit(self, self.pending[1])
    self.pending = None
    reset_temporal_reward_state(self)
    init_action_history(self)
    checkpoint = self.model.checkpoint()
    checkpoint.update({
        "checkpoint_schema": CHECKPOINT_SCHEMA, "algorithm": ALGORITHM,
        "actions": list(ACTIONS), "feature_id": FEATURE_ID,
        "feature_version": None, "feature_schema": FEATURE_SCHEMA,
        "reward_id": self.reward_id, "reward_version": self.reward_id,
        "reward_spec": self.reward_spec, "hyperparameters": HYPERPARAMETERS,
        "network_spec": NETWORK_SPEC, "action_steps": self.total_action_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps, "agent_seed": self.agent_seed,
        "agent_rng_state": self.rng.getstate(), "exploration_spec": self.exploration_spec,
        "safe_exploration": self.safe_exploration,
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "safety_spec": self.safety_spec,
        "safety_decisions": self.safety_decisions,
        "safety_interventions": self.safety_interventions,
        "safety_fallbacks": self.safety_fallbacks,
        "action_history_state": action_history_state(self), "n_step": 1,
        "n_step_state": {"n_step": 1, "gamma": HYPERPARAMETERS["gamma"], "pending": []},
        "retention_spec": self.retention_spec, "training_budget": self.training_budget,
        "training_task": self.training_task,
    })
    save_checkpoint_atomic(self.model_file, checkpoint, torch.save)
    _append_metrics(self, last_game_state)
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    action_steps = int(getattr(self, "stage_action_steps", 0))
    milestone = action_steps // 25_000
    previous = int(getattr(self, "cnn_path_snapshot_milestone", 0))
    if run_dir and milestone > previous:
        snapshots = Path(run_dir) / "checkpoints" / "snapshots"
        snapshots.mkdir(exist_ok=True)
        shutil.copy2(self.model_file, snapshots / f"policy_{milestone * 25_000:06d}.pt")
        self.cnn_path_snapshot_milestone = milestone
    reset_history(self)
    self._feature_cache_key = None
    self._feature_cache_value = None


def _append_metrics(self, last_game_state):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        reset_reward_diagnostics(self)
        self.round_reward = 0.0
        return
    path = Path(run_dir) / "training.csv"
    record = {
        "schema_version": "training-v1", "algorithm": ALGORITHM,
        "round": "" if last_game_state is None else last_game_state.get("round", ""),
        "reward": self.round_reward, "action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "epsilon": epsilon_at(self.stage_action_steps, self.exploration_spec),
        "loss": "" if self.last_loss is None else self.last_loss,
        "updates": self.model.updates, "replay_size": len(self.model.replay),
        "safety_decisions": self.safety_decisions,
        "safety_interventions": self.safety_interventions,
        "safety_fallbacks": self.safety_fallbacks, "checkpoint": str(self.model_file),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    reset_reward_diagnostics(self)
    self.round_reward = 0.0

"""Double-DQN fine-tuning callbacks with round-safe n-step returns."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil

import torch

from agent_code.learning_common.action_history import action_history_state, init_action_history
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.runtime import (
    CHECKPOINT_SCHEMA, effective_legal_mask, save_checkpoint_atomic,
)
from agent_code.learning_common.temporal_reward import (
    reset_temporal_reward_state, temporal_reward_context,
)
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.safety import avoidable_fatal_action, mask_for_decision

from .callbacks import (
    ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
    cached_features,
)
from .features import reset_history
from .learner import Transition
from .task2_shaping import avoidable_task2_wait


def setup_training(self):
    self.round_reward = 0.0
    self.last_losses = None
    self.unfinalized_transition = None
    self.n_step_accumulator = NStepAccumulator(self.n_step, HYPERPARAMETERS["gamma"])
    self.n_step_accumulator.load_state_dict(
        getattr(self, "cnn_distilled_n_step_state", None))
    self.ended_key = None
    _reset_task2_shaping_metrics(self)


def _reset_task2_shaping_metrics(self):
    self.cnn_task2_avoidable_wait_count = 0
    self.cnn_task2_progress_move_count = 0
    self.cnn_task2_useful_bomb_count = 0
    self.cnn_task2_avoidable_wait_reward = 0.0


def _task2_shaping_reward(self, state, action, legal):
    penalty = float(self.cnn_task2_shaping["avoidable_wait_penalty"])
    if penalty == 0.0 or self.training_task != "crate_navigation":
        return 0.0
    facts = avoidable_task2_wait(state, action, legal)
    if not facts["avoidable"]:
        return 0.0
    self.cnn_task2_avoidable_wait_count += 1
    self.cnn_task2_progress_move_count += int(facts["safe_progress_move"])
    self.cnn_task2_useful_bomb_count += int(facts["safe_useful_bomb"])
    self.cnn_task2_avoidable_wait_reward += penalty
    return penalty


def _decision(self, state):
    features = cached_features(self, state)
    physical = effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    legal, _ = mask_for_decision(
        state, physical, self.safety_spec,
        allow_bomb=self.curriculum_allows_bomb, exploring=False)
    return features, physical, legal


def _transition(self, old_state, action, reward, new_state, done):
    old, _, _ = _decision(self, old_state)
    if done:
        return Transition(old.board.copy(), ACTIONS.index(action), reward, None, True, None)
    new, _, next_legal = _decision(self, new_state)
    return Transition(
        old.board.copy(), ACTIONS.index(action), reward,
        new.board.copy(), False, next_legal,
    )


def _submit(self, transition):
    losses = self.model.observe(transition)
    if losses is not None:
        self.last_losses = losses


def _commit_n_step(self, transition):
    for emitted in self.n_step_accumulator.append(transition):
        _submit(self, emitted)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if key == self.ended_key:
        return
    if self.unfinalized_transition is not None:
        previous_key, previous = self.unfinalized_transition
        if previous_key == key:
            return
        _commit_n_step(self, previous)
        self.unfinalized_transition = None
    _, physical, legal = _decision(self, old_game_state)
    context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events,
        reward_id=self.reward_id)
    context.pop("diagnostic", None)
    context["avoidable_fatal"] = avoidable_fatal_action(
        old_game_state, self_action, physical,
        allow_bomb=self.curriculum_allows_bomb)
    reward = reward_from_events(
        events, self.reward_id, old_game_state=old_game_state,
        new_game_state=new_game_state, action=self_action, **context)
    reward += _task2_shaping_reward(
        self, old_game_state, self_action, legal)
    self.round_reward += reward
    self.unfinalized_transition = (key, _transition(
        self, old_game_state, self_action, reward, new_game_state, False))


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if key != self.ended_key:
            if self.unfinalized_transition is not None:
                previous_key, previous = self.unfinalized_transition
                if previous_key != key:
                    _commit_n_step(self, previous)
                self.unfinalized_transition = None
            _, physical, legal = _decision(self, last_game_state)
            reward = reward_from_events(
                events, self.reward_id, old_game_state=last_game_state,
                terminal=True, action=last_action,
                avoidable_fatal=avoidable_fatal_action(
                    last_game_state, last_action, physical,
                    allow_bomb=self.curriculum_allows_bomb))
            reward += _task2_shaping_reward(
                self, last_game_state, last_action, legal)
            self.round_reward += reward
            _commit_n_step(self, _transition(
                self, last_game_state, last_action, reward, None, True))
            self.ended_key = key
    elif self.unfinalized_transition is not None:
        _, previous = self.unfinalized_transition
        _commit_n_step(self, previous)
        self.unfinalized_transition = None
    _save(self)
    _metrics(self, last_game_state)
    reset_temporal_reward_state(self)
    init_action_history(self)
    reset_history(self)
    self.cnn_distilled_observation_cache = {}


def _save(self):
    checkpoint = self.model.checkpoint()
    checkpoint.update({
        "checkpoint_schema": CHECKPOINT_SCHEMA, "algorithm": ALGORITHM,
        "actions": list(ACTIONS), "feature_id": FEATURE_ID,
        "feature_schema": FEATURE_SCHEMA, "reward_id": self.reward_id,
        "reward_version": self.reward_id, "reward_spec": self.reward_spec,
        "hyperparameters": self.cnn_distilled_hyperparameters,
        "network_spec": NETWORK_SPEC,
        "model_architecture": self.model_architecture,
        "action_steps": self.total_action_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "agent_seed": self.agent_seed, "agent_rng_state": self.rng.getstate(),
        "exploration_spec": self.exploration_spec,
        "safety_spec": self.safety_spec, "n_step": self.n_step,
        "n_step_state": self.n_step_accumulator.state_dict(),
        "action_history_state": action_history_state(self),
        "training_task": self.training_task,
        "cnn_task2_shaping": self.cnn_task2_shaping,
    })
    save_checkpoint_atomic(self.model_file, checkpoint, torch.save)
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    milestones = tuple(range(25_000, 200_001, 25_000))
    saved = set(getattr(self, "cnn_distilled_saved_milestones", ()))
    directory = Path(run_dir) / "checkpoints" / "snapshots"
    for milestone in milestones:
        if self.stage_action_steps >= milestone and milestone not in saved:
            directory.mkdir(exist_ok=True)
            shutil.copy2(self.model_file, directory / f"policy_{milestone:06d}.pt")
            saved.add(milestone)
    self.cnn_distilled_saved_milestones = saved


def _metrics(self, state):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        self.round_reward = 0.0
        return
    path = Path(run_dir) / "training.csv"
    fields = (
        "algorithm", "round", "reward", "action_steps", "stage_action_steps",
        "epsilon", "loss", "td_loss", "teacher_kl", "updates", "replay_size",
        "safe_exploration_decisions", "safe_exploration_fallbacks", "checkpoint",
        "task2_avoidable_wait_count", "task2_progress_move_count",
        "task2_useful_bomb_count", "task2_avoidable_wait_reward",
    )
    losses = self.last_losses or {}
    record = {
        "algorithm": ALGORITHM,
        "round": "" if state is None else state.get("round", ""),
        "reward": self.round_reward, "action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "epsilon": epsilon_at(self.stage_action_steps, self.exploration_spec),
        "loss": losses.get("loss", ""), "td_loss": losses.get("td_loss", ""),
        "teacher_kl": losses.get("teacher_kl", ""), "updates": self.model.updates,
        "replay_size": len(self.model.replay),
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "checkpoint": str(self.model_file),
        "task2_avoidable_wait_count": self.cnn_task2_avoidable_wait_count,
        "task2_progress_move_count": self.cnn_task2_progress_move_count,
        "task2_useful_bomb_count": self.cnn_task2_useful_bomb_count,
        "task2_avoidable_wait_reward": self.cnn_task2_avoidable_wait_reward,
    }
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(record)
    self.round_reward = 0.0
    _reset_task2_shaping_metrics(self)

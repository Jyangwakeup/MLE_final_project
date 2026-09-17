from __future__ import annotations

import csv
import os

import torch
import events as e

from agent_code.dqn_agent.model import Transition
from agent_code.learning_common.action_history import (
    action_history_state, init_action_history, project_history, bomb_history,
)
from agent_code.learning_common.n_step import NStepAccumulator
from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA, effective_legal_mask
from agent_code.learning_common.temporal_reward import reset_temporal_reward_state, temporal_reward_context
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.rewards import reward_from_events
from agent_code.team_agent.safety import avoidable_fatal_action, safety_decision
from .callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS,
    NETWORK_SPEC, decision_snapshot, features_from_history, LIFECYCLE_VERSION,
)


FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "stage_action_steps", "epsilon", "q_states", "loss", "updates",
    "replay_size", "safe_exploration_decisions", "safe_exploration_fallbacks",
    "safety_decisions", "safety_interventions", "safety_fallbacks", "checkpoint",
    "robust_safety_interventions", "robust_to_v1_fallbacks",
    "opponent_robust_interventions", "opponent_to_v3_fallbacks",
    "opponent_scenarios_evaluated",
    "robust_guarantee_losses", "robust_search_timeouts",
    "robust_states_evaluated",
    "v1_to_physical_fallbacks", "avoidable_escape_collapses",
    "own_bomb_cycles",
)


def setup_training(self):
    self.round_reward = 0.0
    self.last_loss = None
    self.pending = None
    self.ended_key = None
    self.accumulator = NStepAccumulator(self.n_step, HYPERPARAMETERS["gamma"])
    self.accumulator.load_state_dict(getattr(self, "_resume_n_step_state", None))
    reset_temporal_reward_state(self)


def _transition(self, old_state, action, reward, new_state, done):
    snapshot = decision_snapshot(self, old_state, action)
    safety_class = (
        "own_bomb" if action == "BOMB" or snapshot.before.own_bomb_pending else "ordinary")
    old_legal = snapshot.mask.copy()
    if done:
        return Transition(
            snapshot.vector.copy(), ACTIONS.index(action), reward, None, True, None,
            old_legal, self.training_task, 1, safety_class)
    # The official post-action callback still reports the just-executed step;
    # its physical state is the next observation, whose act increments the clock.
    # Normalize only the new fixed-deadline contract, without mutating inputs.
    from agent_code.team_agent.safety import FIXED_DEADLINE_SAFETY_VERSIONS
    if self.safety_spec['version'] in FIXED_DEADLINE_SAFETY_VERSIONS:
        if new_state.get('round') != old_state.get('round'):
            raise ValueError('Nonterminal transition crosses a round boundary')
        old_step, new_step = int(old_state['step']), int(new_state['step'])
        if new_step == old_step:
            new_state = {**new_state, 'step': old_step + 1}
        elif new_step != old_step + 1:
            raise ValueError('Unexpected next-observation step')
    next_history = project_history(snapshot.after, new_state)
    new = features_from_history(self, new_state, next_history)
    next_legal = effective_legal_mask(
        new.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    next_bomb = bomb_history(next_history, new_state)
    next_legal = safety_decision(
        new_state, next_legal, self.safety_spec,
        allow_bomb=self.curriculum_allows_bomb, exploring=False,
        own_bomb_pending=bool(next_bomb["pending"]),
        own_bomb_state=next_bomb).mask
    return Transition(
        snapshot.vector.copy(), ACTIONS.index(action), reward, new.vector.copy(),
        False, next_legal, old_legal, self.training_task, 1, safety_class)


def _submit(self, transition):
    for aggregate in self.accumulator.append(transition):
        loss = self.model.observe(aggregate)
        if loss is not None:
            self.last_loss = loss


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    snapshot = decision_snapshot(self, old_game_state, self_action)
    context = temporal_reward_context(
        self, self_action, old_game_state, new_game_state, events)
    physical = snapshot.physical
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
        if self.pending is not None and self.pending[0] != key:
            _submit(self, self.pending[1])
        physical = decision_snapshot(self, last_game_state, last_action).physical
        reward = reward_from_events(
            events, self.reward_id, old_game_state=last_game_state,
            terminal=True, action=last_action,
            avoidable_fatal=avoidable_fatal_action(
                last_game_state, last_action, physical,
                allow_bomb=self.curriculum_allows_bomb))
        self.round_reward += reward
        _submit(self, _transition(
            self, last_game_state, last_action, reward, None, True))
        if e.KILLED_SELF in events:
            self.model.replay.mark_recent_own_bomb_fatal(
                self.training_task,
                int(self.safety_replay_spec.get("fatal_prefix_steps", 4)))
    elif self.pending is not None:
        _submit(self, self.pending[1])
    self.pending = None
    reset_temporal_reward_state(self)
    init_action_history(self)
    self._decision_snapshot = None
    self._feature_cache_key = None
    self._feature_cache_value = None
    self._own_bomb_cycle_had_safe_alternative = False
    self._own_bomb_cycle_collapse_recorded = False
    self._own_bomb_placement_certificate = None
    checkpoint = self.model.checkpoint()
    checkpoint.update({
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "lifecycle_version": LIFECYCLE_VERSION,
        "transfer_contract": getattr(self, "transfer_contract", None),
        "algorithm": ALGORITHM,
        "actions": list(ACTIONS),
        "feature_id": FEATURE_ID,
        "feature_version": None,
        "feature_schema": FEATURE_SCHEMA,
        "reward_id": self.reward_id,
        "reward_version": self.reward_id,
        "reward_spec": self.reward_spec,
        "hyperparameters": HYPERPARAMETERS,
        "network_spec": NETWORK_SPEC,
        "action_steps": self.total_action_steps,
        "total_action_steps": self.total_action_steps,
        "stage_action_steps": self.stage_action_steps,
        "agent_seed": self.agent_seed,
        "agent_rng_state": self.rng.getstate(),
        "exploration_spec": self.exploration_spec,
        "safe_exploration": self.safe_exploration,
        "safe_exploration_decisions": self.safe_exploration_decisions,
        "safe_exploration_fallbacks": self.safe_exploration_fallbacks,
        "safety_spec": self.safety_spec,
        "safety_decisions": self.safety_decisions,
        "safety_interventions": self.safety_interventions,
        "safety_fallbacks": self.safety_fallbacks,
        "robust_safety_interventions": int(getattr(
            self, "robust_safety_interventions", 0)),
        "robust_to_v1_fallbacks": int(getattr(
            self, "robust_to_v1_fallbacks", 0)),
        "opponent_robust_interventions": int(getattr(
            self, "opponent_robust_interventions", 0)),
        "opponent_to_v3_fallbacks": int(getattr(
            self, "opponent_to_v3_fallbacks", 0)),
        "opponent_scenarios_evaluated": int(getattr(
            self, "opponent_scenarios_evaluated", 0)),
        "robust_guarantee_losses": int(getattr(
            self, "robust_guarantee_losses", 0)),
        "robust_search_timeouts": int(getattr(
            self, "robust_search_timeouts", 0)),
        "robust_states_evaluated": int(getattr(
            self, "robust_states_evaluated", 0)),
        "v1_to_physical_fallbacks": int(getattr(
            self, "v1_to_physical_fallbacks", 0)),
        "avoidable_escape_collapses": int(getattr(
            self, "avoidable_escape_collapses", 0)),
        "own_bomb_cycles": int(getattr(self, "own_bomb_cycles", 0)),
        "own_bomb_escape_state": {
            "had_safe_alternative": bool(getattr(
                self, "_own_bomb_cycle_had_safe_alternative", False)),
            "collapse_recorded": bool(getattr(
                self, "_own_bomb_cycle_collapse_recorded", False)),
            "placement_certificate": getattr(
                self, "_own_bomb_placement_certificate", None),
        },
        "action_history_state": action_history_state(self),
        "n_step": self.n_step,
        "n_step_state": self.accumulator.state_dict(),
        "retention_spec": self.retention_spec,
        "safety_replay_spec": self.safety_replay_spec,
        "training_budget": self.training_budget,
        "training_task": self.training_task,
    })
    temporary = self.model_file.with_name(self.model_file.name + ".tmp")
    self.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, temporary)
    temporary.replace(self.model_file)
    _append_metrics(self, last_game_state)
    self.round_reward = 0.0


def _append_metrics(self, state):
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    path = os.path.join(run_dir, "training.csv")
    row = {
        "schema_version": "training-v2", "algorithm": ALGORITHM,
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
        "robust_safety_interventions": int(getattr(
            self, "robust_safety_interventions", 0)),
        "robust_to_v1_fallbacks": int(getattr(
            self, "robust_to_v1_fallbacks", 0)),
        "opponent_robust_interventions": int(getattr(
            self, "opponent_robust_interventions", 0)),
        "opponent_to_v3_fallbacks": int(getattr(
            self, "opponent_to_v3_fallbacks", 0)),
        "opponent_scenarios_evaluated": int(getattr(
            self, "opponent_scenarios_evaluated", 0)),
        "robust_guarantee_losses": int(getattr(
            self, "robust_guarantee_losses", 0)),
        "robust_search_timeouts": int(getattr(
            self, "robust_search_timeouts", 0)),
        "robust_states_evaluated": int(getattr(
            self, "robust_states_evaluated", 0)),
        "v1_to_physical_fallbacks": int(getattr(
            self, "v1_to_physical_fallbacks", 0)),
        "avoidable_escape_collapses": int(getattr(
            self, "avoidable_escape_collapses", 0)),
        "own_bomb_cycles": int(getattr(self, "own_bomb_cycles", 0)),
        "checkpoint": str(self.model_file),
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

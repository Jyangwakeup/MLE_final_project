from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agent_code.dqn_agent.model import DQN
from agent_code.learning_common.action_history import (
    action_history_for_state, load_action_history_state, own_bomb_history_for_state,
    record_selected_action, advance_observation, snapshot_history, HistorySnapshot,
    project_history, bomb_history,
)
from agent_code.learning_common.runtime import (
    CHECKPOINT_SCHEMA, adopt_checkpoint_reward, adopt_checkpoint_safety, effective_legal_mask,
    load_common_configuration, validate_checkpoint,
)
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.safety import (
    CERTIFIED_PLACEMENT_SAFETY_VERSION, REARMING_SAFETY_VERSION, FIXED_DEADLINE_SAFETY_VERSION, PROVEN_MOVEMENT_SAFETY_VERSION, CONTROLLABLE_SAFETY_VERSION, OPPONENT_ROBUST_SAFETY_VERSION,
    ROBUST_SAFETY_VERSION,
    safety_decision, survival_diagnostics,
)
from agent_code.team_agent.distillation_capture import maybe_capture_teacher_row
from .features import ACTIONS, FEATURE_ID, FEATURE_SCHEMA, features_for_state


ALGORITHM = "double_dqn"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = {
    "gamma": 0.95,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "replay_capacity": 20_000,
    "warmup": 2_000,
    "target_sync_interval": 1_000,
    "gradient_clip": 10.0,
}
NETWORK_SPEC = {
    "id": "continuous-v2-mlp-128x2",
    "input_shape": [84],
    "layers": ["Linear(84,128)", "ReLU", "Linear(128,128)", "ReLU", "Linear(128,6)"],
    "dueling": False,
    "double_dqn": True,
    "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM,
    "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name,
    "network_spec": NETWORK_SPEC,
    "hyperparameters": HYPERPARAMETERS,
}

LIFECYCLE_VERSION = 'decision-snapshot-v1'
SAFETY_DIAGNOSTIC_VERSION = 'escape-collapse-v2'


def _immutable_array(value):
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DecisionSnapshot:
    key: tuple
    before: HistorySnapshot
    vector: np.ndarray
    physical: np.ndarray
    mask: np.ndarray
    action: str
    after: HistorySnapshot


def decision_snapshot(owner, state, action):
    snapshot = getattr(owner, '_decision_snapshot', None)
    if (snapshot is None or snapshot.key != (state.get('round'), state.get('step'))
            or snapshot.action != action):
        raise ValueError('Training callback has no matching decision snapshot')
    return snapshot


def features_from_history(owner, state, history):
    return features_for_state(
        state, history.previous_action, history.wait_streak,
        previous_position=history.previous_position,
        previous_coin_target=history.previous_coin_target,
        legacy78=getattr(owner, '_legacy78', False))


def setup(self):
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    checkpoint = (
        torch.load(self.model_file, map_location="cpu", weights_only=True)
        if self.model_file.exists() else None
    )
    self.transfer_contract = None if checkpoint is None else checkpoint.get("transfer_contract")
    self._decision_snapshot = None
    if (checkpoint is not None and self.train
            and checkpoint.get('training_task') == 'weak_opponents'
            and checkpoint.get('lifecycle_version') != LIFECYCLE_VERSION):
        raise ValueError('Old Task 3 checkpoints are frozen-evaluation only; transfer from Task 2')
    self._legacy78 = bool(
        checkpoint is not None
        and tuple(checkpoint.get("feature_schema", {}).get("vector_shape", ())) == (78,)
    )
    input_size = 78 if self._legacy78 else 84
    self.model = DQN(
        input_size, len(ACTIONS), seed=self.agent_seed,
        gamma=HYPERPARAMETERS["gamma"],
        learning_rate=HYPERPARAMETERS["learning_rate"],
        batch_size=HYPERPARAMETERS["batch_size"],
        replay_capacity=HYPERPARAMETERS["replay_capacity"],
        warmup=HYPERPARAMETERS["warmup"],
        target_sync_interval=HYPERPARAMETERS["target_sync_interval"],
        device=os.getenv("BOMBERMAN_TORCH_DEVICE", "cpu"),
        training_task=self.training_task,
        retention_spec=self.retention_spec,
        safety_replay_spec=self.safety_replay_spec,
        double_dqn=True,
        hidden_size=128,
    )
    if checkpoint is not None:
        checkpoint_for_validation = checkpoint
        runtime_schema = FEATURE_SCHEMA
        runtime_network = NETWORK_SPEC
        validation_feature_id = FEATURE_ID
        if self._legacy78:
            if self.train:
                raise ValueError("continuous-v2 legacy78 checkpoints are frozen-evaluation only")
            from agent_code.team_agent.feature_system import feature_schema_contract
            validation_feature_id = "continuous-v2-legacy78"
            runtime_schema = feature_schema_contract(validation_feature_id)
            runtime_network = checkpoint["network_spec"]
            checkpoint_for_validation = dict(checkpoint)
            checkpoint_for_validation["feature_id"] = validation_feature_id
            checkpoint_schema = dict(checkpoint["feature_schema"])
            checkpoint_schema["feature_id"] = validation_feature_id
            checkpoint_for_validation["feature_schema"] = checkpoint_schema
        adopt_checkpoint_reward(self, checkpoint)
        adopt_checkpoint_safety(self, checkpoint)
        validate_checkpoint(
            checkpoint_for_validation, algorithm=ALGORITHM,
            feature_id=validation_feature_id,
            feature_schema=runtime_schema, actions=ACTIONS,
            reward_id=self.reward_id, hyperparameters=HYPERPARAMETERS,
            network_spec=runtime_network, training=self.train,
            training_task=self.training_task, safety_spec=self.safety_spec,
            safety_replay_spec=self.safety_replay_spec,
        )
        self.model.load_checkpoint(
            checkpoint, training=self.train, training_task=self.training_task)
        self.total_action_steps = int(checkpoint.get(
            "total_action_steps", checkpoint.get("action_steps", 0)))
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
        self.robust_safety_interventions = int(checkpoint.get(
            "robust_safety_interventions", 0))
        self.robust_to_v1_fallbacks = int(checkpoint.get(
            "robust_to_v1_fallbacks", 0))
        self.opponent_robust_interventions = int(checkpoint.get(
            "opponent_robust_interventions", 0))
        self.opponent_to_v3_fallbacks = int(checkpoint.get(
            "opponent_to_v3_fallbacks", 0))
        self.opponent_scenarios_evaluated = int(checkpoint.get(
            "opponent_scenarios_evaluated", 0))
        self.robust_guarantee_losses = int(checkpoint.get(
            "robust_guarantee_losses", 0))
        self.robust_search_timeouts = int(checkpoint.get(
            "robust_search_timeouts", 0))
        self.robust_states_evaluated = int(checkpoint.get(
            "robust_states_evaluated", 0))
        self.v1_to_physical_fallbacks = int(checkpoint.get(
            "v1_to_physical_fallbacks", 0))
        self.avoidable_escape_collapses = int(checkpoint.get(
            "avoidable_escape_collapses", 0))
        self.own_bomb_cycles = int(checkpoint.get("own_bomb_cycles", 0))
        escape_state = checkpoint.get("own_bomb_escape_state", {})
        self._own_bomb_cycle_had_safe_alternative = bool(
            escape_state.get("had_safe_alternative", False))
        self._own_bomb_cycle_collapse_recorded = bool(
            escape_state.get("collapse_recorded", False))
        self._own_bomb_placement_certificate = escape_state.get(
            "placement_certificate")
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
        self._feature_cache_key = key
        self._feature_cache_value = features_from_history(
            self, game_state, project_history(snapshot_history(self), game_state))
    return self._feature_cache_value


def act(self, game_state):
    before = advance_observation(self, game_state)
    features = _features_for(self, game_state)
    physical = effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    values = self.model.q_values(features.vector)
    maybe_capture_teacher_row(self, game_state, values, physical)
    physical_indices = np.flatnonzero(physical).tolist()
    raw_best = max(float(values[index]) for index in physical_indices)
    raw_tied = [
        index for index in physical_indices if float(values[index]) == raw_best]
    # The diagnostic must not consume learner RNG.  In particular, exploratory
    # actions should use exactly one draw for the epsilon test and one for the
    # selected action, as they did before diagnostics were introduced.
    raw_index = raw_tied[0]
    exploring = False
    if self.train:
        epsilon = epsilon_at(self.stage_action_steps, self.exploration_spec)
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        exploring = self.rng.random() < epsilon
    own_bomb = own_bomb_history_for_state(self, game_state)
    decision = safety_decision(
        game_state, physical, self.safety_spec,
        allow_bomb=self.curriculum_allows_bomb, exploring=exploring,
        own_bomb_pending=bool(own_bomb["pending"]), own_bomb_state=own_bomb)
    legal, fallback = decision.mask, decision.physical_fallback
    enabled = self.safety_spec["mode"] == "all" or (
        self.safety_spec["mode"] == "exploration" and exploring)
    if enabled:
        self.safety_decisions += 1
        self.safety_fallbacks += int(fallback)
        self.robust_to_v1_fallbacks = int(getattr(
            self, "robust_to_v1_fallbacks", 0)) + int(decision.robust_fallback)
        self.opponent_to_v3_fallbacks = int(getattr(
            self, "opponent_to_v3_fallbacks", 0)) + int(
                decision.opponent_fallback)
        self.opponent_scenarios_evaluated = int(getattr(
            self, "opponent_scenarios_evaluated", 0)) + sum(
                decision.opponent_scenario_counts)
        self.robust_guarantee_losses = int(getattr(
            self, "robust_guarantee_losses", 0)) + int(
                decision.robust_guarantee_loss)
        self.robust_search_timeouts = int(getattr(
            self, "robust_search_timeouts", 0)) + int(
                decision.robust_search_timed_out)
        self.robust_states_evaluated = int(getattr(
            self, "robust_states_evaluated", 0)) + int(
                decision.robust_states_evaluated)
        self.v1_to_physical_fallbacks = int(getattr(
            self, "v1_to_physical_fallbacks", 0)) + int(decision.physical_fallback)
        if exploring:
            self.safe_exploration_decisions += 1
            self.safe_exploration_fallbacks += int(fallback)
    indices = np.flatnonzero(legal).tolist()
    if exploring:
        selected = self.rng.choice(indices)
    else:
        masked = values.copy()
        masked[~legal] = -np.inf
        best = max(float(masked[index]) for index in indices)
        tied = [index for index in indices if float(masked[index]) == best]
        selected = self.rng.choice(tied) if self.train else tied[0]
    intervention = bool(enabled and not fallback and not legal[raw_index])
    self.safety_interventions += int(intervention)
    robust_intervention = bool(
        self.safety_spec["version"] in {
            ROBUST_SAFETY_VERSION, OPPONENT_ROBUST_SAFETY_VERSION,
            CONTROLLABLE_SAFETY_VERSION, CERTIFIED_PLACEMENT_SAFETY_VERSION, REARMING_SAFETY_VERSION, FIXED_DEADLINE_SAFETY_VERSION, PROVEN_MOVEMENT_SAFETY_VERSION,
        }
        and not decision.robust_fallback and not decision.physical_fallback
        and decision.v1_mask[raw_index] and not legal[raw_index]
        and decision.route_counts[raw_index] < 2)
    self.robust_safety_interventions = int(getattr(
        self, "robust_safety_interventions", 0)) + int(robust_intervention)
    opponent_intervention = bool(
        self.safety_spec["version"] in {
            OPPONENT_ROBUST_SAFETY_VERSION, CONTROLLABLE_SAFETY_VERSION, CERTIFIED_PLACEMENT_SAFETY_VERSION, REARMING_SAFETY_VERSION, FIXED_DEADLINE_SAFETY_VERSION, PROVEN_MOVEMENT_SAFETY_VERSION,
        }
        and not decision.opponent_fallback
        and decision.v1_mask[raw_index] and not legal[raw_index]
        and decision.opponent_scenario_counts[raw_index] > 0
        and decision.opponent_passing_counts[raw_index]
        < decision.opponent_scenario_counts[raw_index])
    self.opponent_robust_interventions = int(getattr(
        self, "opponent_robust_interventions", 0)) + int(opponent_intervention)
    action = ACTIONS[selected]
    if action == "BOMB":
        self.own_bomb_cycles = int(getattr(self, "own_bomb_cycles", 0)) + 1
    collapse_now = bool(
        own_bomb["pending"] and (
            decision.physical_fallback or decision.robust_guarantee_loss)
        and getattr(self, "_own_bomb_cycle_had_safe_alternative", False)
        and not getattr(self, "_own_bomb_cycle_collapse_recorded", False))
    self.last_safety_diagnostic = {
        "diagnostic_version": SAFETY_DIAGNOSTIC_VERSION,
        "raw_action": ACTIONS[raw_index], "selected_action": action,
        "intervened": intervention, "fallback": bool(fallback),
        "robust_intervened": robust_intervention,
        "robust_to_v1_fallback": bool(decision.robust_fallback),
        "opponent_robust_intervened": opponent_intervention,
        "opponent_to_v3_fallback": bool(decision.opponent_fallback),
        "robust_guarantee_loss": bool(decision.robust_guarantee_loss),
        "robust_search_timed_out": bool(decision.robust_search_timed_out),
        "robust_states_evaluated": int(decision.robust_states_evaluated),
        "v1_to_physical_fallback": bool(decision.physical_fallback),
        "own_bomb_pending": bool(own_bomb["pending"]),
        "own_bomb_visible": bool(own_bomb["visible"]),
        "own_bomb_timer": own_bomb["timer"],
        "own_bomb_placed_step": own_bomb["placed_step"],
        "avoidable_escape_collapse": collapse_now,
        "independent_routes": list(decision.route_counts),
        "escape_slack": list(decision.escape_slack),
        "opponent_scenario_counts": list(decision.opponent_scenario_counts),
        "opponent_passing_counts": list(decision.opponent_passing_counts),
        "opponent_failing_profiles": [
            None if value is None else list(value)
            for value in decision.opponent_failing_profiles
        ],
        "opponent_failing_orders": [
            None if value is None else list(value)
            for value in decision.opponent_failing_orders
        ],
        "physical_mask": physical.astype(bool).tolist(),
        "decision_mask": legal.astype(bool).tolist(),
        **survival_diagnostics(
            game_state, physical, allow_bomb=self.curriculum_allows_bomb),
    }
    self._last_decision_key = (game_state.get("round"), game_state.get("step"))
    self._last_decision_mask = legal.copy()
    self._last_had_safe_alternative = bool(
        own_bomb["pending"] and legal.any() and not decision.physical_fallback)
    placement_safe_alternative = bool(
        enabled and not decision.physical_fallback
        and action == "BOMB"
        and decision.v1_mask[:ACTIONS.index("BOMB")].any())
    if collapse_now:
        self.avoidable_escape_collapses = int(getattr(
            self, "avoidable_escape_collapses", 0)) + 1
        self._own_bomb_cycle_collapse_recorded = True
    if action == "BOMB":
        bomb_index = ACTIONS.index("BOMB")
        self._own_bomb_placement_certificate = {
            "had_safe_non_bomb_alternative": placement_safe_alternative,
            "independent_routes": int(decision.route_counts[bomb_index]),
            "scenario_count": int(decision.opponent_scenario_counts[bomb_index]),
            "passing_scenarios": int(decision.opponent_passing_counts[bomb_index]),
        }
    if action == "BOMB" or own_bomb["pending"]:
        self._own_bomb_cycle_had_safe_alternative = bool(
            getattr(self, "_own_bomb_cycle_had_safe_alternative", False)
            or self._last_had_safe_alternative or placement_safe_alternative)
    else:
        self._own_bomb_cycle_had_safe_alternative = False
        self._own_bomb_cycle_collapse_recorded = False
        self._own_bomb_placement_certificate = None
    record_selected_action(self, game_state, action)
    self._decision_snapshot = DecisionSnapshot(
        (game_state.get('round'), game_state.get('step')), before,
        _immutable_array(features.vector), _immutable_array(physical),
        _immutable_array(legal), action, snapshot_history(self))
    return action


def state_to_features(game_state):
    result = features_for_state(game_state)
    return None if result is None else result.vector


def legal_actions(game_state):
    result = features_for_state(game_state)
    return np.zeros(len(ACTIONS), dtype=bool) if result is None else result.legal_mask.copy()

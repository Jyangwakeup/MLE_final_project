"""Environment, action-mask, and checkpoint contracts shared by new agents."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from agent_code.team_agent.feature_system import validate_checkpoint_feature_contract
from agent_code.team_agent.exploration import (
    exploration_from_environment, safe_exploration_from_environment,
)
from agent_code.team_agent.safety import resolve_safety_spec, safety_from_environment
from agent_code.team_agent.rewards import resolve_reward_spec
from .temporal_reward import init_temporal_reward_state
from .action_history import init_action_history
from .training_spec import (
    n_step_from_environment, retention_from_environment,
    training_budget_from_environment,
)


CHECKPOINT_SCHEMA = "training-resume-v6"
DEFAULT_REWARD_ID = "r1"
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
FEATURE_ID_ENV = "BOMBERMAN_FEATURE_ID"
REWARD_ID_ENV = "BOMBERMAN_REWARD_ID"
REWARD_VERSION_ENV = "BOMBERMAN_REWARD_VERSION"
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"
ALLOW_BOMB_ENV = "BOMBERMAN_ALLOW_BOMB"
AGENT_SEED_ENV = "BOMBERMAN_AGENT_SEED"
TASK_ORDER = ("coin_navigation", "crate_navigation", "weak_opponents", "full_match")


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value; got {value!r}")


def load_common_configuration(self, *, feature_id: str, default_model: Path) -> None:
    configured_feature = os.getenv(FEATURE_ID_ENV, feature_id)
    if configured_feature != feature_id:
        raise ValueError(
            f"agent requires feature ID {feature_id!r}; got {configured_feature!r}"
        )
    reward_id = os.getenv(REWARD_ID_ENV)
    legacy_reward = os.getenv(REWARD_VERSION_ENV)
    if reward_id and legacy_reward and reward_id != legacy_reward:
        raise ValueError("BOMBERMAN_REWARD_ID conflicts with BOMBERMAN_REWARD_VERSION")
    self.feature_id = feature_id
    self.configured_reward_id = reward_id or legacy_reward
    self.reward_id = self.configured_reward_id or DEFAULT_REWARD_ID
    self.reward_version = self.reward_id
    self.reward_spec = resolve_reward_spec(self.reward_id)
    self.training_task = os.getenv(TRAINING_TASK_ENV)
    self.curriculum_allows_bomb = _flag(ALLOW_BOMB_ENV, True)
    self.agent_seed = int(os.getenv(AGENT_SEED_ENV, "0"))
    self.rng = random.Random(self.agent_seed)
    self.exploration_spec = exploration_from_environment()
    self.safety_spec, self.safety_spec_configured = safety_from_environment()
    if not self.safety_spec_configured:
        legacy_safe = safe_exploration_from_environment()
        self.safety_spec = resolve_safety_spec(
            legacy_safe_exploration=legacy_safe)
    self.safe_exploration = self.safety_spec["mode"] in {"exploration", "all"}
    self.n_step = n_step_from_environment()
    self.retention_spec = retention_from_environment()
    self.training_budget = training_budget_from_environment()
    configured_checkpoint = os.getenv(CHECKPOINT_ENV)
    self.model_file = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint else default_model
    )
    self.action_steps = 0
    self.total_action_steps = 0
    self.stage_action_steps = 0
    self.safe_exploration_decisions = 0
    self.safe_exploration_fallbacks = 0
    self.safety_decisions = 0
    self.safety_interventions = 0
    self.safety_fallbacks = 0
    self._feature_cache_key = None
    self._feature_cache_value = None
    init_temporal_reward_state(self)
    init_action_history(self)


def adopt_checkpoint_reward(self, payload: dict[str, Any]) -> None:
    """Use checkpoint reward metadata unless the caller explicitly configured it."""
    checkpoint_reward_id = payload.get("reward_id", payload.get("reward_version"))
    if checkpoint_reward_id is None:
        raise ValueError("checkpoint has no reward ID")
    configured = getattr(self, "configured_reward_id", None)
    if configured is not None and configured != checkpoint_reward_id:
        raise ValueError(
            "configured reward ID does not match checkpoint: "
            f"{configured!r} != {checkpoint_reward_id!r}")
    self.reward_id = checkpoint_reward_id
    self.reward_version = checkpoint_reward_id
    self.reward_spec = resolve_reward_spec(checkpoint_reward_id)


def adopt_checkpoint_safety(self, payload: dict[str, Any]) -> None:
    """Use an embedded v6 policy unless an evaluation explicitly overrides it."""
    embedded = payload.get("safety_spec")
    if embedded is None:
        return
    embedded = resolve_safety_spec(embedded)
    if not getattr(self, "safety_spec_configured", False):
        self.safety_spec = embedded
        self.safe_exploration = embedded["mode"] in {"exploration", "all"}


def effective_legal_mask(legal_mask: np.ndarray, actions: tuple[str, ...],
                         allow_bomb: bool) -> np.ndarray:
    """Apply the explicit curriculum constraint without changing Feature output."""
    result = np.asarray(legal_mask, dtype=bool).copy()
    if not allow_bomb:
        result[actions.index("BOMB")] = False
    if not result.any():
        raise ValueError("effective action mask contains no legal action")
    return result


def epsilon_at(action_steps: int, hyperparameters: dict[str, Any]) -> float:
    start = float(hyperparameters["epsilon_start"])
    end = float(hyperparameters["epsilon_end"])
    decay = int(hyperparameters["epsilon_decay_action_steps"])
    progress = min(max(action_steps, 0) / decay, 1.0)
    if progress >= 1.0:
        return end
    return start + (end - start) * progress


def validate_checkpoint(
    payload: dict[str, Any], *, algorithm: str, feature_id: str,
    feature_schema: dict[str, Any], actions: tuple[str, ...],
    reward_id: str, hyperparameters: dict[str, Any],
    network_spec: dict[str, Any] | None, training: bool,
    training_task: str | None, safety_spec: dict[str, Any] | None = None,
) -> None:
    checkpoint_schema = payload.get("checkpoint_schema")
    frozen_schemas = {
        "training-resume-v1", "training-resume-v2",
        "training-resume-v3", "training-resume-v4", "training-resume-v5",
        CHECKPOINT_SCHEMA,
    }
    if checkpoint_schema not in frozen_schemas:
        raise ValueError("checkpoint uses an incompatible checkpoint schema")
    if training and checkpoint_schema != CHECKPOINT_SCHEMA:
        raise ValueError("legacy checkpoint schemas are frozen-evaluation only")
    if payload.get("algorithm") != algorithm:
        raise ValueError("checkpoint uses an incompatible algorithm")
    board_contract = feature_schema.get("board_shape")
    board_shape = (
        None if board_contract is None or any(value is None for value in board_contract[1:])
        else tuple(board_contract[1:])
    )
    validate_checkpoint_feature_contract(
        payload, feature_id, actions, board_shape=board_shape)
    if payload.get("feature_schema") != feature_schema:
        raise ValueError("checkpoint uses an incompatible feature schema")
    if payload.get("reward_id", payload.get("reward_version")) != reward_id:
        raise ValueError("checkpoint uses an incompatible reward ID")
    if (
        checkpoint_schema == CHECKPOINT_SCHEMA
        and payload.get("reward_spec") != resolve_reward_spec(reward_id)
    ):
        raise ValueError("checkpoint uses an incompatible reward specification")
    if tuple(payload.get("actions", ())) != actions:
        raise ValueError("checkpoint uses an incompatible action order")
    if payload.get("hyperparameters") != hyperparameters:
        raise ValueError("checkpoint uses incompatible hyperparameters")
    if payload.get("network_spec") != network_spec:
        raise ValueError("checkpoint uses an incompatible network architecture")
    if (
        training and payload.get("safety_spec")
        != resolve_safety_spec(safety_spec)
    ):
        raise ValueError("checkpoint uses an incompatible safety specification")
    if training and payload.get("training_task") != training_task:
        try:
            previous = TASK_ORDER.index(payload.get("training_task"))
            current = TASK_ORDER.index(training_task)
        except ValueError as exception:
            raise ValueError("checkpoint uses an unknown curriculum task") from exception
        if current != previous + 1:
            raise ValueError("checkpoint uses an incompatible curriculum task")


def save_checkpoint_atomic(path: Path, payload: dict[str, Any], saver) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    saver(payload, temporary)
    temporary.replace(path)

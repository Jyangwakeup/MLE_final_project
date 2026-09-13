"""Inference callbacks and state representation for the Q-learning agent."""

from pathlib import Path
import json
import os
import pickle
import random

import numpy as np

from agent_code.team_agent.exploration import (
    agent_seed_from_environment, epsilon_at, exploration_from_environment,
)
from agent_code.team_agent.feature_system import (
    normalize_feature_id, validate_checkpoint_feature_contract,
)
from agent_code.team_agent.rewards import REWARD_VERSION, resolve_reward_spec
from .features import ACTIONS, FEATURE_ID, features_for_state

FEATURE_VERSION = None

MODEL_FILE = Path(__file__).with_name("final.pkl")
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
SEED = 0
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"
REWARD_VERSION_ENV = "BOMBERMAN_REWARD_VERSION"
FEATURE_ID_ENV = "BOMBERMAN_FEATURE_ID"
REWARD_ID_ENV = "BOMBERMAN_REWARD_ID"
CHECKPOINT_SCHEMA_VERSION = "training-resume-v4"


class ResumeCompatibilityError(RuntimeError):
    """A full training checkpoint cannot be resumed under this contract."""


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}")


def _training_task():
    value = os.getenv(TRAINING_TASK_ENV)
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError(f"{TRAINING_TASK_ENV} must not be empty")
    return value


def _configured_reward_id() -> str | None:
    reward_id = os.getenv(REWARD_ID_ENV)
    legacy_reward_id = os.getenv(REWARD_VERSION_ENV)
    if reward_id and legacy_reward_id and reward_id != legacy_reward_id:
        raise ValueError("BOMBERMAN_REWARD_ID conflicts with BOMBERMAN_REWARD_VERSION")
    return reward_id or legacy_reward_id


def setup(self):
    """Load a learned Q table, or initialise an empty one."""
    self.agent_seed = agent_seed_from_environment(SEED)
    self.exploration_spec = exploration_from_environment()
    self.rng = random.Random(self.agent_seed)
    if "BOMBERMAN_ALLOW_BOMB" in os.environ:
        self.allow_bomb = _env_flag("BOMBERMAN_ALLOW_BOMB", True)
    else:
        self.allow_bomb = _env_flag("Q_LEARNING_ALLOW_BOMB", True)
    self.training_task = _training_task()
    configured_feature_id = os.getenv(FEATURE_ID_ENV)
    configured_feature_id = (
        None if configured_feature_id is None
        else normalize_feature_id(configured_feature_id)
    )
    configured_reward_id = _configured_reward_id()
    self.feature_id = configured_feature_id or FEATURE_ID
    self.reward_id = configured_reward_id or REWARD_VERSION
    self.reward_version = self.reward_id
    self.reward_spec = resolve_reward_spec(self.reward_id)
    self.q_table = {}
    self.training_steps = 0
    self._feature_cache_key = None
    self._feature_cache_value = None
    self.previous_action = None
    self.move_history = []
    self.stationary_streak = 0

    configured_checkpoint = os.getenv(CHECKPOINT_ENV)
    self.model_file = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint else MODEL_FILE
    )
    if self.model_file.exists():
        try:
            with self.model_file.open("rb") as file:
                payload = pickle.load(file)
            if not isinstance(payload, dict) or "q_table" not in payload:
                if self.train:
                    raise ResumeCompatibilityError(
                        "Cannot resume from a legacy directly pickled Q table")
                self.feature_id = configured_feature_id or "discrete-v1"
                self.q_table = payload
                self.logger.warning("Loaded legacy Q table without a training contract")
                return
            if (
                self.train
                and payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION
                and not configured_checkpoint
            ):
                self.logger.warning(
                    "Ignoring package-local legacy Q checkpoint; starting fresh")
                return
            checkpoint_feature_id = normalize_feature_id(
                payload.get("feature_id"), payload.get("feature_version"))
            self.feature_id = configured_feature_id or checkpoint_feature_id
            validate_checkpoint_feature_contract(payload, self.feature_id, ACTIONS)
            self.q_table = payload["q_table"]
            self.training_steps = int(payload.get("training_steps", 0))
            checkpoint_reward_id = payload.get(
                "reward_id", payload.get("reward_version"))
            if configured_reward_id and checkpoint_reward_id != configured_reward_id:
                raise ResumeCompatibilityError(
                    "Configured reward ID does not match the checkpoint: "
                    f"{configured_reward_id!r} != {checkpoint_reward_id!r}")
            if checkpoint_reward_id is not None:
                self.reward_id = checkpoint_reward_id
                self.reward_version = checkpoint_reward_id
                self.reward_spec = resolve_reward_spec(checkpoint_reward_id)
            self.checkpoint_reward_version = checkpoint_reward_id
            if self.train:
                if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
                    raise ResumeCompatibilityError(
                        "Cannot resume Q-learning from a legacy checkpoint schema")
                if payload.get("reward_spec") != self.reward_spec:
                    raise ResumeCompatibilityError(
                        "Cannot continue Q-learning with a different reward contract")
                self.rng.setstate(payload["agent_rng_state"])
            self.logger.info("Loaded Q table with %d states", len(self.q_table))
        except (OSError, pickle.PickleError, EOFError, TypeError) as exc:
            self.logger.warning("Could not load Q table (%s); starting fresh", exc)
            self.q_table = {}
    elif configured_checkpoint and not self.train:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")
    elif not self.train:
        self.logger.warning("No Q table found; actions will initially be random")
    self.logger.info("Feature ID: %s; bomb actions enabled: %s", self.feature_id, self.allow_bomb)


def act(self, game_state: dict) -> str:
    features = _features_for(self, game_state)
    state = features.state_key
    legal_mask = features.legal_mask.copy()
    if not self.allow_bomb:
        legal_mask[ACTIONS.index("BOMB")] = False
    legal_indices = np.flatnonzero(legal_mask).tolist()
    values = self.q_table.get(state)
    _record_q_diagnostic(self, values is None)
    if self.train:
        epsilon = epsilon_at(self.training_steps, self.exploration_spec)
        self.training_steps += 1
        if self.rng.random() < epsilon:
            return ACTIONS[self.rng.choice(legal_indices)]
    if values is None:
        return ACTIONS[self.rng.choice(legal_indices)]
    best = max(float(values[index]) for index in legal_indices)
    choices = [index for index in legal_indices if float(values[index]) == best]
    return ACTIONS[self.rng.choice(choices)]


def _features_for(self, game_state: dict):
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        self._feature_cache_value = features_for_state(
            game_state, getattr(self, "previous_action", None),
            getattr(self, "feature_id", FEATURE_ID))
        self._feature_cache_key = key
    return self._feature_cache_value


def _record_q_diagnostic(self, unseen: bool) -> None:
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    run_id = os.getenv("BOMBERMAN_RUN_ID") or Path(run_dir).name
    record = {
        "schema_version": "q-diagnostics-v1",
        "run_id": run_id,
        "agent_name": getattr(self, "agent_name", "q_learning_agent"),
        "q_decisions": 1,
        "unseen_q_states": int(unseen),
    }
    diagnostics_path = Path(run_dir) / "q_diagnostics.jsonl"
    with diagnostics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")

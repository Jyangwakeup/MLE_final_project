"""Inference callbacks and state representation for the Q-learning agent."""

from pathlib import Path
import json
import os
import pickle
import random

import numpy as np

from agent_code.team_agent.feature_system import (
    normalize_feature_id,
    validate_checkpoint_feature_contract,
)
from agent_code.team_agent.rewards import REWARD_VERSION, resolve_reward_spec
<<<<<<< HEAD
from agent_code.team_agent.exploration import (
    agent_seed_from_environment,
    epsilon_at,
    exploration_from_environment,
)
from .features import ACTIONS, FEATURE_VERSION, features_for_state
=======
from .features import ACTIONS, FEATURE_ID, FEATURE_VERSION, features_for_state
>>>>>>> e6253fd1 (add more feature id, reward id, and model)
MODEL_FILE = Path(__file__).with_name("final.pkl")
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
SEED = 0
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"
REWARD_VERSION_ENV = "BOMBERMAN_REWARD_VERSION"
FEATURE_ID_ENV = "BOMBERMAN_FEATURE_ID"
REWARD_ID_ENV = "BOMBERMAN_REWARD_ID"


class ResumeCompatibilityError(RuntimeError):
    """A full training checkpoint cannot be resumed under this contract."""


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable or raise on an ambiguous value."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def _training_task():
    """Return the explicitly selected curriculum task, if any."""
    value = os.getenv(TRAINING_TASK_ENV)
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError(f"{TRAINING_TASK_ENV} must not be empty")
    return value


def setup(self):
    """Load a learned Q table, or initialise an empty one."""
    self.agent_seed = agent_seed_from_environment(SEED)
    self.exploration_spec = exploration_from_environment()
    self.rng = random.Random(self.agent_seed)
    # The experiment runner uses the shared flag for curriculum consistency.
    # Keep the old agent-specific variable as a backwards-compatible fallback
    # for direct `main.py` invocations documented by this agent.
    if "BOMBERMAN_ALLOW_BOMB" in os.environ:
        self.allow_bomb = _env_flag("BOMBERMAN_ALLOW_BOMB", True)
    else:
        self.allow_bomb = _env_flag("Q_LEARNING_ALLOW_BOMB", True)
    self.training_task = _training_task()
    configured_feature_id = normalize_feature_id(
        os.getenv(FEATURE_ID_ENV) or FEATURE_ID, FEATURE_VERSION)
    if configured_feature_id != FEATURE_ID:
        raise ValueError(
            f"q_learning_agent only supports feature ID {FEATURE_ID!r}; "
            f"got {configured_feature_id!r}")
    self.feature_id = configured_feature_id
    reward_id = os.getenv(REWARD_ID_ENV)
    legacy_reward_id = os.getenv(REWARD_VERSION_ENV)
    if reward_id and legacy_reward_id and reward_id != legacy_reward_id:
        raise ValueError("BOMBERMAN_REWARD_ID conflicts with BOMBERMAN_REWARD_VERSION")
    configured_reward_id = reward_id or legacy_reward_id
    self.reward_version = configured_reward_id or REWARD_VERSION
    self.reward_id = self.reward_version
    self.reward_spec = resolve_reward_spec(self.reward_version)
    self.q_table = {}
    self.training_steps = 0
    self._feature_cache_key = None
    self._feature_cache_value = None
    self.previous_action = None
    self.move_history = []
    self.stationary_streak = 0
    self.logger.info("Bomb actions enabled: %s", self.allow_bomb)

    configured_checkpoint = os.getenv(CHECKPOINT_ENV)
    self.model_file = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint
        else MODEL_FILE
    )
    if self.model_file.exists():
        try:
            with self.model_file.open("rb") as file:
                payload = pickle.load(file)
            if isinstance(payload, dict) and "q_table" in payload:
                validate_checkpoint_feature_contract(payload, FEATURE_ID, ACTIONS)
                self.q_table = payload["q_table"]
                self.training_steps = int(payload.get("training_steps", 0))
                previous_task = payload.get("training_task")
                checkpoint_reward_id = payload.get(
                    "reward_id", payload.get("reward_version"))
                if checkpoint_reward_id is None:
                    raise ValueError("Q-learning checkpoint has no reward ID")
                if (
                    configured_reward_id is not None
                    and configured_reward_id != checkpoint_reward_id
                ):
                    raise ResumeCompatibilityError(
                        "Configured reward ID does not match the checkpoint: "
                        f"{configured_reward_id!r} != {checkpoint_reward_id!r}")
                self.reward_version = checkpoint_reward_id
                self.reward_id = checkpoint_reward_id
                self.reward_spec = resolve_reward_spec(checkpoint_reward_id)
            else:
                raise ValueError(
                    "Q-learning checkpoint has no explicit feature contract")
            if self.train and "agent_rng_state" in payload:
                self.rng.setstate(payload["agent_rng_state"])
            self.logger.info("Loaded Q table with %d states", len(self.q_table))
        except (OSError, pickle.PickleError, EOFError, TypeError) as exc:
            self.logger.warning("Could not load Q table (%s); starting fresh", exc)
            self.q_table = {}
    elif configured_checkpoint and not self.train:
        raise FileNotFoundError(
            f"Evaluation checkpoint does not exist: {self.model_file}"
        )
    elif not self.train:
        self.logger.warning("No Q table found; actions will initially be random")


def act(self, game_state: dict) -> str:
    """Choose a legal action using epsilon-greedy exploration while training."""
    features = _features_for(self, game_state)
    state = features.state_key
    legal_mask = features.legal_mask.copy()
    if not self.allow_bomb:
        legal_mask[ACTIONS.index("BOMB")] = False
    legal_indices = np.flatnonzero(legal_mask).tolist()
    values = self.q_table.get(state)
    _record_q_diagnostic(self, values is None)

    if self.train:
        epsilon = epsilon_at(
            self.training_steps,
            getattr(self, "exploration_spec", None),
        )
        self.training_steps += 1
        if self.rng.random() < epsilon:
            return ACTIONS[self.rng.choice(legal_indices)]

    if values is None:
        return ACTIONS[self.rng.choice(legal_indices)]
    best = max(float(values[index]) for index in legal_indices)
    choices = [index for index in legal_indices if float(values[index]) == best]
    return ACTIONS[self.rng.choice(choices)]


def _features_for(self, game_state: dict):
    """Extract shared features once for each round-step state."""
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        self._feature_cache_value = features_for_state(
            game_state, getattr(self, "previous_action", None))
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

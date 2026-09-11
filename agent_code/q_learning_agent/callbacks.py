"""Inference callbacks and state representation for the Q-learning agent."""

from pathlib import Path
import json
import os
import pickle
import random

import numpy as np

from .features import ACTIONS, FEATURE_VERSION, features_for_state
MODEL_FILE = Path(__file__).with_name("final.pkl")
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
SEED = 0
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"


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
    self.rng = random.Random(SEED)
    # The experiment runner uses the shared flag for curriculum consistency.
    # Keep the old agent-specific variable as a backwards-compatible fallback
    # for direct `main.py` invocations documented by this agent.
    if "BOMBERMAN_ALLOW_BOMB" in os.environ:
        self.allow_bomb = _env_flag("BOMBERMAN_ALLOW_BOMB", True)
    else:
        self.allow_bomb = _env_flag("Q_LEARNING_ALLOW_BOMB", True)
    self.training_task = _training_task()
    self.q_table = {}
    self.training_steps = 0
    self._feature_cache_key = None
    self._feature_cache_value = None
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
                self.q_table = payload["q_table"]
                self.training_steps = int(payload.get("training_steps", 0))
                previous_task = payload.get("training_task")
                previous_feature_version = payload.get("feature_version")
            else:  # backwards-compatible with a directly pickled Q table
                self.q_table = payload
                previous_task = None
                previous_feature_version = None
            if previous_feature_version != FEATURE_VERSION:
                self.logger.warning(
                    "Ignoring Q table trained with feature version %r; expected %r",
                    previous_feature_version,
                    FEATURE_VERSION,
                )
                self.q_table = {}
                self.training_steps = 0
            if (
                self.train
                and self.training_task is not None
                and self.training_task != previous_task
            ):
                self.logger.info(
                    "Training task changed from %r to %r; resetting exploration progress",
                    previous_task,
                    self.training_task,
                )
                self.training_steps = 0
            self.logger.info("Loaded Q table with %d states", len(self.q_table))
        except (OSError, pickle.PickleError, EOFError, TypeError, ValueError) as exc:
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
        epsilon = max(0.05, 1.0 - 0.95 * min(self.training_steps / 75_000, 1.0))
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
        self._feature_cache_value = features_for_state(game_state)
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

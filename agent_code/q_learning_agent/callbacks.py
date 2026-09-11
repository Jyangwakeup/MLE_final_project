"""Inference callbacks and state representation for the Q-learning agent."""

from pathlib import Path
import os
import pickle
import random

import numpy as np

from .features import legal_actions, state_to_features


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MODEL_FILE = Path(__file__).with_name("q-table.pkl")
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
    self.allow_bomb = _env_flag("Q_LEARNING_ALLOW_BOMB", True)
    self.training_task = _training_task()
    self.q_table = {}
    self.training_steps = 0
    self.logger.info("Bomb actions enabled: %s", self.allow_bomb)

    if MODEL_FILE.exists():
        try:
            with MODEL_FILE.open("rb") as file:
                payload = pickle.load(file)
            if isinstance(payload, dict) and "q_table" in payload:
                self.q_table = payload["q_table"]
                self.training_steps = int(payload.get("training_steps", 0))
                previous_task = payload.get("training_task")
            else:  # backwards-compatible with a directly pickled Q table
                self.q_table = payload
                previous_task = None
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
    elif not self.train:
        self.logger.warning("No Q table found; actions will initially be random")


def act(self, game_state: dict) -> str:
    """Choose a legal action using epsilon-greedy exploration while training."""
    state = state_to_features(game_state)
    legal_indices = np.flatnonzero(
        legal_actions(game_state, allow_bomb=self.allow_bomb)
    ).tolist()

    if self.train:
        epsilon = max(0.05, 1.0 - 0.95 * min(self.training_steps / 75_000, 1.0))
        self.training_steps += 1
        if self.rng.random() < epsilon:
            return ACTIONS[self.rng.choice(legal_indices)]

    values = self.q_table.get(state)
    if values is None:
        return ACTIONS[self.rng.choice(legal_indices)]
    best = max(float(values[index]) for index in legal_indices)
    choices = [index for index in legal_indices if float(values[index]) == best]
    return ACTIONS[self.rng.choice(choices)]

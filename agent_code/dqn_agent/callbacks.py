from pathlib import Path
import os
import random

import numpy as np
import torch

from agent_code.team_agent.rewards import REWARD_VERSION, resolve_reward_spec
from .features import ACTIONS, FEATURE_DIM, FEATURE_VERSION, features_for_state
from .model import DQN


INPUT_SIZE = FEATURE_DIM
MODEL_FILE = Path(__file__).with_name("final.pt")
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
SEED = 0
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"
ALLOW_BOMB_ENV = "BOMBERMAN_ALLOW_BOMB"
REWARD_VERSION_ENV = "BOMBERMAN_REWARD_VERSION"


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
    self.rng = random.Random(SEED)
    self.model = DQN(INPUT_SIZE, len(ACTIONS), seed=SEED)
    self.allow_bomb = _env_flag(ALLOW_BOMB_ENV, True)
    self.training_task = _training_task()
    self.reward_version = os.getenv(REWARD_VERSION_ENV, REWARD_VERSION)
    self.reward_spec = resolve_reward_spec(self.reward_version)
    self.action_steps = 0
    self._feature_cache_key = None
    self._feature_cache_value = None
    configured_checkpoint = os.getenv(CHECKPOINT_ENV)
    self.model_file = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint
        else MODEL_FILE
    )
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
        checkpoint_version = checkpoint.get("feature_version")
        if checkpoint_version != FEATURE_VERSION:
            message = (
                f"DQN checkpoint uses feature version {checkpoint_version!r}; "
                f"expected {FEATURE_VERSION!r}"
            )
            if not self.train:
                raise ValueError(message)
            self.logger.warning("%s; starting with a new network", message)
        else:
            if (
                self.train
                and "checkpoint_schema" in checkpoint
                and checkpoint.get("reward_version") != self.reward_version
            ):
                raise ValueError("Cannot continue DQN with a different reward version")
            self.model.load_checkpoint(checkpoint, training=self.train)
            self.action_steps = int(checkpoint.get("action_steps", 0))
            previous_task = checkpoint.get("training_task")
            if self.train and "agent_rng_state" in checkpoint:
                self.rng.setstate(checkpoint["agent_rng_state"])
            self.logger.info("Loaded DQN checkpoint from %s", self.model_file)
    elif configured_checkpoint and not self.train:
        raise FileNotFoundError(
            f"Evaluation checkpoint does not exist: {self.model_file}"
        )
    elif not self.train:
        self.logger.warning("No DQN checkpoint found; using an untrained network")


def act(self, game_state: dict) -> str:
    extracted = _features_for(self, game_state)
    features = extracted.vector
    legal = extracted.legal_mask
    if not self.allow_bomb:
        legal = legal.copy()
        legal[ACTIONS.index("BOMB")] = False
    legal_indices = np.flatnonzero(legal).tolist()

    if self.train:
        epsilon = max(0.05, 1.0 - 0.95 * min(self.action_steps / 80_000, 1.0))
        self.action_steps += 1
        if self.rng.random() < epsilon:
            return ACTIONS[self.rng.choice(legal_indices)]

    q_values = self.model.q_values(features)
    q_values[~legal] = -np.inf
    best = np.max(q_values[legal_indices])
    choices = [index for index in legal_indices if q_values[index] == best]
    if self.train:
        return ACTIONS[self.rng.choice(choices)]
    return ACTIONS[choices[0]]


def legal_actions(game_state: dict) -> np.ndarray:
    """Compatibility API returning the shared physical legality mask."""
    extracted = features_for_state(game_state)
    if extracted is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    return extracted.legal_mask.copy()


def state_to_features(game_state: dict):
    """Return the shared vector without recommending or scoring actions."""
    extracted = features_for_state(game_state)
    return None if extracted is None else extracted.vector


def _features_for(self, game_state: dict):
    """Extract the relatively expensive shared features once per round-step."""
    if game_state is None:
        return None
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        self._feature_cache_value = features_for_state(game_state)
        self._feature_cache_key = key
    return self._feature_cache_value

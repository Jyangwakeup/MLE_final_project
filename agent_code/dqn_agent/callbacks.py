from pathlib import Path
import os
import random

import numpy as np
import torch

from .model import DQN


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
BOARD_SIZE = 17
CHANNELS = 7
INPUT_SIZE = CHANNELS * BOARD_SIZE * BOARD_SIZE + 1
MODEL_FILE = Path(__file__).with_name("dqn-model.pt")
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
SEED = 0
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"


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
    self.training_task = _training_task()
    self.action_steps = 0
    configured_checkpoint = os.getenv(CHECKPOINT_ENV)
    self.model_file = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint
        else MODEL_FILE
    )
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
        self.model.load_checkpoint(checkpoint, training=self.train)
        self.action_steps = int(checkpoint.get("action_steps", 0))
        previous_task = checkpoint.get("training_task")
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
            self.action_steps = 0
        self.logger.info("Loaded DQN checkpoint from %s", self.model_file)
    elif configured_checkpoint and not self.train:
        raise FileNotFoundError(
            f"Evaluation checkpoint does not exist: {self.model_file}"
        )
    elif not self.train:
        self.logger.warning("No DQN checkpoint found; using an untrained network")


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    legal = legal_actions(game_state)
    legal_indices = np.flatnonzero(legal).tolist()

    if self.train:
        epsilon = max(0.05, 1.0 - 0.95 * min(self.action_steps / 80_000, 1.0))
        self.action_steps += 1
        if self.rng.random() < epsilon:
            return ACTIONS[self.rng.choice(legal_indices)]

    q_values = self.model.q_values(features)
    q_values[~legal] = -np.inf
    return ACTIONS[int(np.argmax(q_values))]


def legal_actions(game_state: dict) -> np.ndarray:
    field = game_state["field"]
    x, y = game_state["self"][3]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(other[3] for other in game_state["others"])
    legal = []
    for dx, dy in DIRECTIONS:
        xx, yy = x + dx, y + dy
        legal.append(
            0 <= xx < field.shape[0] and 0 <= yy < field.shape[1]
            and field[xx, yy] == 0 and (xx, yy) not in blocked
        )
    legal.extend((True, bool(game_state["self"][2])))
    return np.asarray(legal, dtype=bool)


def state_to_features(game_state: dict):
    """Encode observations only; this function never recommends or scores actions."""
    if game_state is None:
        return None
    field = game_state["field"]
    if field.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected a {BOARD_SIZE}x{BOARD_SIZE} board, got {field.shape}")

    channels = np.zeros((CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    channels[0] = field == -1
    channels[1] = field == 1
    for x, y in game_state["coins"]:
        channels[2, x, y] = 1.0
    for (x, y), timer in game_state["bombs"]:
        channels[3, x, y] = (timer + 1) / 5.0
    channels[4] = np.clip(game_state["explosion_map"], 0, 2) / 2.0
    x, y = game_state["self"][3]
    channels[5, x, y] = 1.0
    for _, _, _, (x, y) in game_state["others"]:
        channels[6, x, y] = 1.0
    bomb_available = np.asarray([float(game_state["self"][2])], dtype=np.float32)
    return np.concatenate((channels.reshape(-1), bomb_available))

"""Inference callbacks for the selectable-feature baseline DQN agent."""

from pathlib import Path
import os
import random

import numpy as np
import torch

from agent_code.learning_common.action_history import (
    action_history_for_state, init_action_history, load_action_history_state,
    record_selected_action,
)
from agent_code.learning_common.temporal_reward import init_temporal_reward_state
from agent_code.learning_common.training_spec import (
    n_step_from_environment, retention_from_environment,
    training_budget_from_environment,
)
from agent_code.team_agent.exploration import (
    agent_seed_from_environment, epsilon_at, exploration_from_environment,
    safe_exploration_from_environment, survivable_exploration_mask,
)
from agent_code.team_agent.feature_system import (
    get_feature_schema, normalize_feature_id, validate_checkpoint_feature_contract,
)
from agent_code.team_agent.rewards import REWARD_VERSION, resolve_reward_spec
from .features import ACTIONS, FEATURE_ID, features_for_state
from .model import (
    DEFAULT_BATCH_SIZE, DEFAULT_GAMMA, DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_REPLAY_CAPACITY,
    DEFAULT_TARGET_SYNC_INTERVAL, DEFAULT_WARMUP, DQN,
)

FEATURE_VERSION = None
INPUT_SIZE = get_feature_schema(FEATURE_ID).vector_shape[0]
MODEL_FILE = Path(__file__).with_name("final.pt")
CHECKPOINT_ENV = "BOMBERMAN_CHECKPOINT"
SEED = 0
TRAINING_TASK_ENV = "BOMBERMAN_TRAINING_TASK"
ALLOW_BOMB_ENV = "BOMBERMAN_ALLOW_BOMB"
REWARD_VERSION_ENV = "BOMBERMAN_REWARD_VERSION"
TORCH_DEVICE_ENV = "BOMBERMAN_TORCH_DEVICE"
FEATURE_ID_ENV = "BOMBERMAN_FEATURE_ID"
REWARD_ID_ENV = "BOMBERMAN_REWARD_ID"
CHECKPOINT_SCHEMA_VERSION = "training-resume-v5"


def network_spec(input_size: int) -> dict:
    return {
        "architecture": "mlp",
        "input_shape": [int(input_size)],
        "hidden_sizes": [DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE],
        "output_size": len(ACTIONS),
        "double_dqn": False,
    }


HYPERPARAMETERS = {
    "gamma": DEFAULT_GAMMA,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "batch_size": DEFAULT_BATCH_SIZE,
    "replay_capacity_per_task": DEFAULT_REPLAY_CAPACITY,
    "warmup": DEFAULT_WARMUP,
    "target_sync_interval": DEFAULT_TARGET_SYNC_INTERVAL,
}


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
    self.agent_seed = agent_seed_from_environment(SEED)
    self.exploration_spec = exploration_from_environment()
    self.safe_exploration = safe_exploration_from_environment()
    self.n_step = n_step_from_environment()
    self.retention_spec = retention_from_environment()
    self.training_budget = training_budget_from_environment()
    self.rng = random.Random(self.agent_seed)
    self.allow_bomb = _env_flag(ALLOW_BOMB_ENV, True)
    self.training_task = _training_task()
    configured_feature_id = os.getenv(FEATURE_ID_ENV)
    configured_feature_id = (
        None if configured_feature_id is None
        else normalize_feature_id(configured_feature_id)
    )
    configured_reward_id = _configured_reward_id()
    configured_checkpoint = os.getenv(CHECKPOINT_ENV)
    self.model_file = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint else MODEL_FILE
    )
    checkpoint = None
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
        if (
            self.train
            and checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION
            and not configured_checkpoint
        ):
            self.logger.warning(
                "Ignoring package-local legacy DQN checkpoint; starting fresh")
            checkpoint = None
    if checkpoint is not None:
        checkpoint_feature_id = normalize_feature_id(
            checkpoint.get("feature_id"), checkpoint.get("feature_version"))
        self.feature_id = configured_feature_id or checkpoint_feature_id
        validate_checkpoint_feature_contract(checkpoint, self.feature_id, ACTIONS)
    else:
        self.feature_id = configured_feature_id or FEATURE_ID

    schema = get_feature_schema(self.feature_id)
    if schema.vector_shape is None:
        raise ValueError(f"DQN requires a vector feature schema; got {self.feature_id!r}")
    self.model = DQN(
        schema.vector_shape[0], len(ACTIONS), seed=self.agent_seed,
        device=os.getenv(TORCH_DEVICE_ENV, "cpu"),
        training_task=self.training_task, retention_spec=self.retention_spec,
    )
    self.network_spec = network_spec(schema.vector_shape[0])
    self.hyperparameters = dict(HYPERPARAMETERS)
    self.reward_id = configured_reward_id or REWARD_VERSION
    self.reward_version = self.reward_id
    self.reward_spec = resolve_reward_spec(self.reward_id)
    self.action_steps = 0
    self.total_action_steps = 0
    self.stage_action_steps = 0
    self.safe_exploration_decisions = 0
    self.safe_exploration_fallbacks = 0
    self._feature_cache_key = None
    self._feature_cache_value = None
    init_temporal_reward_state(self)
    init_action_history(self)

    if checkpoint is not None:
        checkpoint_reward_id = checkpoint.get(
            "reward_id", checkpoint.get("reward_version"))
        if configured_reward_id and checkpoint_reward_id != configured_reward_id:
            raise ValueError(
                "Configured reward ID does not match the checkpoint: "
                f"{configured_reward_id!r} != {checkpoint_reward_id!r}")
        if checkpoint_reward_id is not None:
            self.reward_id = checkpoint_reward_id
            self.reward_version = checkpoint_reward_id
            self.reward_spec = resolve_reward_spec(checkpoint_reward_id)
        if self.train:
            if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("Cannot resume DQN from a legacy checkpoint schema")
            if checkpoint.get("reward_spec") != self.reward_spec:
                raise ValueError("Cannot continue DQN with a different reward contract")
            if checkpoint.get("network_spec") != self.network_spec:
                raise ValueError("Cannot continue DQN with a different network contract")
            if checkpoint.get("hyperparameters") != self.hyperparameters:
                raise ValueError(
                    "Cannot continue DQN with different learner hyperparameters")
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
        self._resume_n_step_state = (
            checkpoint.get("n_step_state") if same_task else None)
        if self.train:
            self.rng.setstate(checkpoint["agent_rng_state"])
            if same_task:
                load_action_history_state(self, checkpoint.get("action_history_state"))
        self.logger.info("Loaded DQN checkpoint from %s", self.model_file)
    elif configured_checkpoint and not self.train:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")
    elif not self.train:
        self.logger.warning("No DQN checkpoint found; using an untrained network")


def act(self, game_state: dict) -> str:
    extracted = _features_for(self, game_state)
    features = extracted.vector
    legal = extracted.legal_mask.copy()
    if not self.allow_bomb:
        legal[ACTIONS.index("BOMB")] = False
    legal_indices = np.flatnonzero(legal).tolist()
    if self.train:
        stage_steps = int(getattr(
            self, "stage_action_steps", getattr(self, "action_steps", 0)))
        total_steps = int(getattr(
            self, "total_action_steps", getattr(self, "action_steps", 0)))
        epsilon = epsilon_at(
            stage_steps, getattr(self, "exploration_spec", None))
        self.stage_action_steps = stage_steps
        self.total_action_steps = total_steps
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        if self.rng.random() < epsilon:
            if getattr(self, "safe_exploration", False):
                legal, fallback = survivable_exploration_mask(
                    game_state, legal, allow_bomb=self.allow_bomb)
                legal_indices = np.flatnonzero(legal).tolist()
                self.safe_exploration_decisions = int(getattr(
                    self, "safe_exploration_decisions", 0)) + 1
                self.safe_exploration_fallbacks = int(getattr(
                    self, "safe_exploration_fallbacks", 0)) + int(fallback)
            action = ACTIONS[self.rng.choice(legal_indices)]
            record_selected_action(self, game_state, action)
            return action
    q_values = self.model.q_values(features)
    q_values[~legal] = -np.inf
    best = np.max(q_values[legal_indices])
    choices = [index for index in legal_indices if q_values[index] == best]
    action = ACTIONS[self.rng.choice(choices) if self.train else choices[0]]
    record_selected_action(self, game_state, action)
    return action


def legal_actions(game_state: dict, feature_id: str = FEATURE_ID) -> np.ndarray:
    extracted = features_for_state(game_state, feature_id=feature_id)
    if extracted is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    return extracted.legal_mask.copy()


def state_to_features(game_state: dict, feature_id: str = FEATURE_ID):
    extracted = features_for_state(game_state, feature_id=feature_id)
    return None if extracted is None else extracted.vector


def _features_for(self, game_state: dict):
    if game_state is None:
        return None
    key = (game_state.get("round"), game_state.get("step"))
    if key != self._feature_cache_key:
        previous_action, wait_streak = action_history_for_state(self, game_state)
        self._feature_cache_value = features_for_state(
            game_state, previous_action,
            getattr(self, "feature_id", FEATURE_ID), wait_streak)
        self._feature_cache_key = key
    return self._feature_cache_value

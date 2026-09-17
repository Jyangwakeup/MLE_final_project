"""Runtime callbacks for the distilled action-aligned CNN agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from agent_code.learning_common.action_history import record_selected_action
from agent_code.learning_common.runtime import (
    adopt_checkpoint_reward, adopt_checkpoint_safety, effective_legal_mask,
    load_common_configuration,
)
from agent_code.team_agent.exploration import epsilon_at
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.safety import mask_for_decision, resolve_safety_spec

from .features import (
    BOARD_SHAPE, FEATURE_ID, FEATURE_SCHEMA, features_for_state,
    initialize_history, record_position,
)
from .learner import DistilledDoubleDQNLearner, TeacherDataset
from .model import build_network


ALGORITHM = "cnn_distilled_double_dqn"
MODEL_FILE = Path(__file__).with_name("final.pt")
ARCHITECTURES = ("global", "action_aligned", "action_aligned_d4")
HYPERPARAMETERS = {
    "gamma": 0.95, "learning_rate": 5e-5, "batch_size": 64,
    "replay_capacity": 20_000, "warmup": 5_000,
    "target_sync_interval": 2_000, "gradient_clip": 10.0,
    "teacher_kl_weight": 1.0, "temperature": 1.0,
    "epsilon_start": 0.20, "epsilon_end": 0.05,
    "epsilon_decay_action_steps": 25_000,
}
NETWORK_SPEC = {
    "id": "distilled-spatial-action-family-v1",
    "input_shape": list(BOARD_SHAPE), "channels": 64,
    "dilations": [1, 2, 4, 8], "architectures": list(ARCHITECTURES),
    "dueling": True, "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "feature_schema": FEATURE_SCHEMA, "checkpoint_name": MODEL_FILE.name,
    "network_spec": NETWORK_SPEC, "hyperparameters": HYPERPARAMETERS,
}


def _training_options():
    path = os.getenv("BOMBERMAN_CONFIG")
    if not path:
        return {}
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    options = document.get("cnn_distillation", {})
    if not isinstance(options, dict):
        raise ValueError("config.cnn_distillation must be an object")
    return options


def cached_features(owner, game_state):
    if game_state is None:
        return None
    key = (game_state.get("round"), game_state.get("step"))
    cache = getattr(owner, "cnn_distilled_observation_cache", {})
    if cache and next(iter(cache))[0] != key[0]:
        cache = {}
    if key not in cache:
        cache[key] = features_for_state(owner, game_state)
        if len(cache) > 2:
            del cache[next(iter(cache))]
    owner.cnn_distilled_observation_cache = cache
    return cache[key]


def _validate_checkpoint(checkpoint, architecture, *, expected_hyperparameters=None):
    expected = {
        "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
        "feature_schema": FEATURE_SCHEMA, "actions": list(ACTIONS),
        "network_spec": NETWORK_SPEC,
        "model_architecture": architecture,
    }
    if expected_hyperparameters is not None:
        expected["hyperparameters"] = expected_hyperparameters
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"distilled CNN checkpoint has incompatible {name}")


def _diagnostic_safety_override(safety_spec, options, *, train):
    mode = options.get("diagnostic_safety_override")
    if mode is None:
        return safety_spec
    if train or not options.get("decision_diagnostics", False):
        raise ValueError(
            "diagnostic_safety_override is allowed only for frozen diagnostics")
    if mode not in {"all", "off"}:
        raise ValueError("diagnostic_safety_override must be 'all' or 'off'")
    overridden = dict(safety_spec)
    overridden["mode"] = mode
    return resolve_safety_spec(overridden)


def setup(self):
    initialize_history(self)
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    options = _training_options()
    checkpoint = None
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
    init_path = os.getenv("BOMBERMAN_INIT_CHECKPOINT") if self.train else None
    initial = None if not init_path else torch.load(
        Path(init_path).expanduser().resolve(), map_location="cpu", weights_only=True)
    source = checkpoint or initial
    architecture = (
        source.get("model_architecture") if source is not None
        else options.get("architecture", "action_aligned_d4"))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported distilled CNN architecture: {architecture}")
    teacher_dataset = None
    dataset_path = options.get("dataset")
    if self.train and dataset_path:
        dataset_path = Path(dataset_path).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = Path(__file__).resolve().parents[2] / dataset_path
        teacher_dataset = TeacherDataset(dataset_path, self.agent_seed + 104729)
    runtime_hyperparameters = dict(HYPERPARAMETERS)
    if self.train and "teacher_kl_weight" in options:
        runtime_hyperparameters["teacher_kl_weight"] = float(
            options["teacher_kl_weight"])
    wait_penalty = float(options.get("task2_avoidable_wait_penalty", 0.0))
    if wait_penalty > 0:
        raise ValueError("task2_avoidable_wait_penalty must be non-positive")
    self.cnn_task2_shaping = {
        "version": "cnn-task2-progress-v1",
        "avoidable_wait_penalty": wait_penalty,
    }
    self.cnn_decision_diagnostics = bool(options.get("decision_diagnostics", False))
    self.cnn_decision_diagnostics_file = None
    if self.cnn_decision_diagnostics:
        run_dir = os.getenv("BOMBERMAN_RUN_DIR")
        if not run_dir:
            raise ValueError("decision_diagnostics requires BOMBERMAN_RUN_DIR")
        path = Path(run_dir) / "cnn_decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cnn_decision_diagnostics_file = path.open(
            "a", encoding="utf-8", buffering=1)
    self.model_architecture = architecture
    self.cnn_distilled_hyperparameters = runtime_hyperparameters
    self.model = DistilledDoubleDQNLearner(
        build_network(architecture), build_network(architecture),
        hyperparameters=runtime_hyperparameters, seed=self.agent_seed,
        teacher_dataset=teacher_dataset,
    )
    if source is None:
        if not self.train:
            raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")
        return
    _validate_checkpoint(
        source, architecture,
        expected_hyperparameters=(
            runtime_hyperparameters if checkpoint is not None and self.train else None))
    if checkpoint is not None:
        adopt_checkpoint_reward(self, checkpoint)
        adopt_checkpoint_safety(self, checkpoint)
        self.safety_spec = _diagnostic_safety_override(
            self.safety_spec, options, train=self.train)
        if checkpoint.get("reward_id") != self.reward_id:
            raise ValueError("distilled CNN checkpoint has incompatible reward ID")
        if self.train and checkpoint.get("safety_spec") != resolve_safety_spec(self.safety_spec):
            raise ValueError("distilled CNN checkpoint has incompatible safety spec")
        self.model.load_checkpoint(checkpoint, training=self.train)
        self.cnn_distilled_n_step_state = checkpoint.get("n_step_state")
        self.total_action_steps = int(checkpoint.get("total_action_steps", 0))
        self.stage_action_steps = int(checkpoint.get("stage_action_steps", 0))
        if self.train:
            self.rng.setstate(checkpoint["agent_rng_state"])
    else:
        self.model.load_checkpoint(initial, training=True, warm_start=True)
    self.action_steps = self.total_action_steps


def act(self, game_state):
    features = cached_features(self, game_state)
    physical = effective_legal_mask(
        features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    values = self.model.q_values(features.board)
    exploring = False
    if self.train:
        epsilon = epsilon_at(self.stage_action_steps, self.exploration_spec)
        self.stage_action_steps += 1
        self.total_action_steps += 1
        self.action_steps = self.total_action_steps
        exploring = self.rng.random() < epsilon
    legal, fallback = mask_for_decision(
        game_state, physical, self.safety_spec,
        allow_bomb=self.curriculum_allows_bomb, exploring=exploring)
    indices = np.flatnonzero(legal).tolist()
    if exploring:
        selected = self.rng.choice(indices)
    else:
        masked = values.copy()
        masked[~legal] = -np.inf
        best = max(float(masked[index]) for index in indices)
        tied = [index for index in indices if float(masked[index]) == best]
        selected = self.rng.choice(tied) if self.train else tied[0]
    action = ACTIONS[selected]
    _record_decision_diagnostics(
        self, game_state, values, physical, legal, selected, exploring, fallback)
    record_selected_action(self, game_state, action)
    record_position(self, game_state)
    return action


def _record_decision_diagnostics(
        self, game_state, values, physical, legal, selected, exploring, fallback):
    output = getattr(self, "cnn_decision_diagnostics_file", None)
    if output is None:
        return
    from .task2_shaping import avoidable_task2_wait

    facts = avoidable_task2_wait(game_state, "WAIT", legal)
    legal_values = np.asarray(values, dtype=np.float64).copy()
    legal_values[~np.asarray(legal, dtype=bool)] = -np.inf
    wait_index = ACTIONS.index("WAIT")
    alternatives = [
        float(legal_values[index]) for index in range(len(ACTIONS))
        if index != wait_index and np.isfinite(legal_values[index])
    ]
    best_non_wait = max(alternatives) if alternatives else None
    record = {
        "round": int(game_state.get("round", 0)),
        "step": int(game_state.get("step", 0)),
        "action": ACTIONS[selected],
        "selected_index": int(selected),
        "exploring": bool(exploring),
        "safety_fallback": bool(fallback),
        "q_values": [float(value) for value in values],
        "physical_mask": [bool(value) for value in physical],
        "legal_mask": [bool(value) for value in legal],
        "wait_q_margin": (
            None if best_non_wait is None or not legal[wait_index]
            else float(values[wait_index]) - best_non_wait),
        **facts,
    }
    output.write(json.dumps(record, separators=(",", ":")) + "\n")


def state_to_features(game_state):
    class EmptyHistory:
        cnn_distilled_history = ()
        cnn_distilled_history_round = None
    result = features_for_state(EmptyHistory(), game_state)
    return None if result is None else result.board


def legal_actions(game_state):
    class EmptyHistory:
        cnn_distilled_history = ()
        cnn_distilled_history_round = None
    result = features_for_state(EmptyHistory(), game_state)
    return np.zeros(6, dtype=bool) if result is None else result.legal_mask.copy()

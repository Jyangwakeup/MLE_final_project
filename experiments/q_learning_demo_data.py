"""Capture and encode frozen teacher trajectories for tile-coded Q-learning."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from agent_code.learning_common import linear_agent
from agent_code.learning_common.action_history import (
    advance_observation, init_action_history, record_selected_action,
)
from agent_code.learning_common.runtime import effective_legal_mask
from agent_code.learning_common.temporal_reward import (
    observed_terminal_state, reset_temporal_reward_state, temporal_reward_context,
)
from experiments.agent_variants.optimized_double_q_lambda_demo_agent import callbacks
from agent_code.team_agent.rewards import reward_from_events, resolve_reward_spec
from agent_code.team_agent.safety import avoidable_fatal_action, mask_for_decision


CAPTURE_ENV = "BOMBERMAN_Q_DEMO_CAPTURE"
SCHEMA = "q-learning-demonstration-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_raw_transition(path: Path, *, environment_seed: int, round_index: int,
                          old_state, action, new_state, events, terminal: bool):
    record = {
        "schema_version": SCHEMA,
        "environment_seed": int(environment_seed),
        "round": int(round_index),
        "step": int(old_state["step"]),
        "old_state": old_state,
        "action": str(action),
        "new_state": new_state,
        "events": tuple(str(event) for event in events),
        "terminal": bool(terminal),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as file:
        pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_raw(path: Path):
    with Path(path).open("rb") as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                return


def _owner():
    owner = SimpleNamespace(
        linear_config=callbacks,
        curriculum_allows_bomb=True,
        reward_id="r20_safe_credit_targeted_wait",
        reward_spec=resolve_reward_spec("r20_safe_credit_targeted_wait"),
        safety_spec={
            "version": "survival-mask-v1", "mode": "all", "horizon": 7,
            "fallback": "physical_q",
        },
        _feature_cache_key=None,
        _feature_cache_value=None,
    )
    init_action_history(owner)
    reset_temporal_reward_state(owner)
    return owner


def encode_raw_dataset(raw_path: Path, output: Path) -> dict:
    """Replay capture rows through the unchanged r20 feature/reward lifecycle."""
    records = list(load_raw(raw_path))
    if not records:
        raise ValueError("demonstration capture is empty")
    owner = _owner()
    states, actions, rewards, next_states = [], [], [], []
    legal_masks, next_legal = [], []
    dones, action_admissible, seeds, rounds, steps = [], [], [], [], []
    current_round = None
    last_action_key = None
    for record in records:
        key = (record["environment_seed"], record["round"])
        if current_round != key:
            init_action_history(owner)
            reset_temporal_reward_state(owner)
            owner._feature_cache_key = None
            owner._feature_cache_value = None
            current_round = key
            last_action_key = None
        old_state = record["old_state"]
        new_state = record["new_state"]
        action = record["action"]
        events = list(record["events"])
        terminal = bool(record["terminal"])
        action_key = (*key, int(record["step"]))
        first_callback_for_action = action_key != last_action_key
        if first_callback_for_action:
            advance_observation(owner, old_state)
        old_features = linear_agent.features(owner, old_state)
        old_vector = old_features.vector.copy()
        if first_callback_for_action:
            record_selected_action(owner, old_state, action)
            last_action_key = action_key
        reward_state = (
            observed_terminal_state(old_state, action, events)
            if terminal else new_state
        )
        context = temporal_reward_context(
            owner, action, old_state, reward_state, events,
            reward_id=owner.reward_id,
        )
        physical = effective_legal_mask(
            old_features.legal_mask, callbacks.ACTIONS, True)
        current_legal, _ = mask_for_decision(
            old_state, physical, owner.safety_spec,
            allow_bomb=True, exploring=False)
        context["avoidable_fatal"] = avoidable_fatal_action(
            old_state, action, physical, allow_bomb=True)
        reward = reward_from_events(
            events, owner.reward_id, old_game_state=old_state,
            new_game_state=None if terminal else new_state,
            terminal=terminal, action=action, **context,
        )
        if terminal:
            next_vector = np.zeros(84, dtype=np.float32)
            legal = np.zeros(len(callbacks.ACTIONS), dtype=bool)
        else:
            new_features = linear_agent.features(owner, new_state)
            next_vector = new_features.vector.copy()
            legal = effective_legal_mask(
                new_features.legal_mask, callbacks.ACTIONS, True)
            legal, _ = mask_for_decision(
                new_state, legal, owner.safety_spec,
                allow_bomb=True, exploring=False)
        states.append(old_vector)
        actions.append(callbacks.ACTIONS.index(action))
        rewards.append(reward)
        next_states.append(next_vector)
        legal_masks.append(current_legal)
        next_legal.append(legal)
        dones.append(terminal)
        action_admissible.append(
            bool(current_legal[callbacks.ACTIONS.index(action)]))
        seeds.append(record["environment_seed"])
        rounds.append(record["round"])
        steps.append(record["step"])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int8),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        legal_masks=np.asarray(legal_masks, dtype=bool),
        next_legal=np.asarray(next_legal, dtype=bool),
        dones=np.asarray(dones, dtype=bool),
        action_admissible=np.asarray(action_admissible, dtype=bool),
        environment_seeds=np.asarray(seeds, dtype=np.int32),
        rounds=np.asarray(rounds, dtype=np.int16),
        steps=np.asarray(steps, dtype=np.int16),
        feature_id=np.asarray(callbacks.FEATURE_ID),
        reward_id=np.asarray(owner.reward_id),
        schema_version=np.asarray(SCHEMA),
    )
    return {
        "raw_sha256": sha256(raw_path),
        "dataset_sha256": sha256(output),
        "transitions": len(states),
        "terminal_transitions": int(sum(dones)),
        "inadmissible_teacher_actions": int(
            len(action_admissible) - sum(action_admissible)),
        "environment_seeds": [int(min(seeds)), int(max(seeds))],
    }


def write_manifest(path: Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

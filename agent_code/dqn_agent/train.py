import csv
import os
from pathlib import Path
from typing import List

import torch

from agent_code.team_agent.rewards import REWARD_VERSION, reward_from_events

from .callbacks import ACTIONS, FEATURE_VERSION, _features_for
from .model import Transition


TRAINING_FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "epsilon", "q_states", "loss", "updates", "checkpoint",
)


def setup_training(self):
    self.round_reward = 0.0
    self.last_loss = None
    self.pending = None
    self.ended_key = None


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if key == self.ended_key:
        return
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    reward = reward_from_events(events)
    old_features = _features_for(self, old_game_state)
    new_features = _features_for(self, new_game_state)
    next_legal = new_features.legal_mask.copy()
    if not getattr(self, "allow_bomb", True):
        next_legal[ACTIONS.index("BOMB")] = False
    self.pending = (key, Transition(
        old_features.vector.copy(), ACTIONS.index(self_action), reward,
        new_features.vector.copy(), False, next_legal,
    ))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if key != self.ended_key:
            if self.pending is not None and self.pending[0] != key:
                _submit(self, self.pending[1])
            reward = reward_from_events(events)
            features = _features_for(self, last_game_state)
            transition = Transition(
                features.vector.copy(), ACTIONS.index(last_action), reward,
                None, True, None,
            )
            _submit(self, transition)
            self.ended_key = key
    elif self.pending is not None:
        _submit(self, self.pending[1])

    self.pending = None
    checkpoint = self.model.checkpoint()
    checkpoint.update({
        "feature_version": FEATURE_VERSION,
        "reward_version": REWARD_VERSION,
        "action_steps": self.action_steps,
        "training_task": self.training_task,
    })
    self.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, self.model_file)
    _append_training_metrics(self, last_game_state)
    self.logger.info("Round environment reward: %.2f; last loss: %s",
                     self.round_reward, self.last_loss)
    self.round_reward = 0.0


def _append_training_metrics(self, last_game_state) -> None:
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:
        return
    path = Path(run_dir) / "training.csv"
    epsilon = max(0.05, 1.0 - 0.95 * min(self.action_steps / 80_000, 1.0))
    record = {
        "schema_version": "training-v1",
        "algorithm": "dqn",
        "round": "" if last_game_state is None else last_game_state.get("round", ""),
        "reward": self.round_reward,
        "action_steps": self.action_steps,
        "epsilon": epsilon,
        "q_states": "",
        "loss": "" if self.last_loss is None else self.last_loss,
        "updates": self.model.updates,
        "checkpoint": str(self.model_file),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _submit(self, transition: Transition):
    self.last_loss = self.model.observe(transition)
    self.round_reward += transition.reward

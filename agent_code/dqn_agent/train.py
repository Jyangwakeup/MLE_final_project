import csv
import os
from pathlib import Path
from typing import List

import events as e
import settings as s
import torch

from .callbacks import ACTIONS, MODEL_FILE, legal_actions, state_to_features
from .model import Transition


TRAINING_FIELDS = (
    "schema_version", "algorithm", "round", "reward", "action_steps",
    "epsilon", "q_states", "loss", "updates", "checkpoint",
)


def setup_training(self):
    self.round_reward = 0.0
    self.last_loss = None
    self.pending = None


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTIONS:
        return
    key = (old_game_state.get("round"), old_game_state.get("step"))
    if self.pending is not None and self.pending[0] != key:
        _submit(self, self.pending[1])
    reward = reward_from_events(events)
    self.pending = (key, Transition(
        state_to_features(old_game_state), ACTIONS.index(self_action), reward,
        state_to_features(new_game_state), False, legal_actions(new_game_state),
    ))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action in ACTIONS:
        key = (last_game_state.get("round"), last_game_state.get("step"))
        if self.pending is not None and self.pending[0] != key:
            _submit(self, self.pending[1])
        reward = reward_from_events(events)
        transition = Transition(
            state_to_features(last_game_state), ACTIONS.index(last_action), reward,
            None, True, None,
        )
        _submit(self, transition)
    elif self.pending is not None:
        _submit(self, self.pending[1])

    self.pending = None
    checkpoint = self.model.checkpoint()
    checkpoint.update({
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


def reward_from_events(events: List[str]) -> float:
    """Reward objective outcomes without encoding a human-chosen best action.

    Repeated outcome events are counted individually.  Death is deliberately
    counted once because the environment can report both KILLED_SELF and
    GOT_KILLED for the same death.
    """
    reward = -0.01  # small time cost encourages completing objectives efficiently
    reward += events.count(e.COIN_COLLECTED) * s.REWARD_COIN
    reward += events.count(e.KILLED_OPPONENT) * s.REWARD_KILL
    reward += events.count(e.CRATE_DESTROYED) * 0.2
    reward -= events.count(e.INVALID_ACTION) * 0.1
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 10.0
    return float(reward)

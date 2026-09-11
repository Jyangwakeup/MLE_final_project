from typing import List

import events as e
import settings as s
import torch

from .callbacks import ACTIONS, MODEL_FILE, legal_actions, state_to_features
from .model import Transition


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
    torch.save(checkpoint, MODEL_FILE)
    self.logger.info("Round environment reward: %.2f; last loss: %s",
                     self.round_reward, self.last_loss)
    self.round_reward = 0.0


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

"""Uniformly sample from the same physically legal actions as learned agents."""

from __future__ import annotations

import os
import random

from agent_code.team_agent.exploration import agent_seed_from_environment
from agent_code.team_agent.features import ACTIONS, extract_features


def setup(self):
    self.agent_seed = agent_seed_from_environment(0)
    self.rng = random.Random(self.agent_seed)
    self.allow_bomb = _env_flag("BOMBERMAN_ALLOW_BOMB", True)


def act(self, game_state: dict) -> str:
    features = extract_features(game_state)
    if features is None:
        return "WAIT"
    legal = features.legal_mask.copy()
    if not self.allow_bomb:
        legal[ACTIONS.index("BOMB")] = False
    legal_indices = [index for index, allowed in enumerate(legal) if allowed]
    action = ACTIONS[self.rng.choice(legal_indices)]
    self.logger.info("Uniform legal baseline chose %s", action)
    return action


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")

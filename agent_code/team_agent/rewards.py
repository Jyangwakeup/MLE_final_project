"""Versioned objective rewards shared by all learning agents."""

from __future__ import annotations

from typing import Sequence

import events as e
REWARD_VERSION = "r1"
REWARD_SPECS = {
    "r1": {
        "step": -0.01,
        "coin_collected": 1.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.2,
        "death": -10.0,
        "invalid_action": -0.1,
    },
    "r1_no_crate": {
        "step": -0.01,
        "coin_collected": 1.0,
        "killed_opponent": 5.0,
        "crate_destroyed": 0.0,
        "death": -10.0,
        "invalid_action": -0.1,
    },
}
DEATH_EVENTS = frozenset((e.KILLED_SELF, e.GOT_KILLED))


def resolve_reward_spec(version: str = REWARD_VERSION) -> dict[str, float]:
    """Return an independent copy of a registered reward definition."""
    try:
        return dict(REWARD_SPECS[version])
    except KeyError as exception:
        raise ValueError(f"Unknown reward version: {version!r}") from exception


def reward_from_events(events: Sequence[str], version: str = REWARD_VERSION) -> float:
    """Convert framework events into a versioned scalar reward.

    Repeated objective events are counted separately. Death is counted once
    when either or both framework death events are present.
    """
    spec = resolve_reward_spec(version)
    event_rewards = {
        e.COIN_COLLECTED: spec["coin_collected"],
        e.KILLED_OPPONENT: spec["killed_opponent"],
        e.CRATE_DESTROYED: spec["crate_destroyed"],
        e.INVALID_ACTION: spec["invalid_action"],
    }
    reward = spec["step"]
    for event, event_reward in event_rewards.items():
        reward += events.count(event) * event_reward
    if DEATH_EVENTS.intersection(events):
        reward += spec["death"]
    return float(reward)

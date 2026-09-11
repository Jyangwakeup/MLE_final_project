"""Shared objective reward interface for team learning agents."""

from typing import Sequence

import events as e
import settings as s


REWARD_VERSION = "base-v1"
STEP_REWARD = -0.01
EVENT_REWARDS = {
    e.COIN_COLLECTED: float(s.REWARD_COIN),
    e.KILLED_OPPONENT: float(s.REWARD_KILL),
    e.CRATE_DESTROYED: 0.2,
    e.SURVIVED_ROUND: 1.0,
    e.INVALID_ACTION: -0.2,
}
DEATH_PENALTY = -10.0
DEATH_EVENTS = frozenset((e.KILLED_SELF, e.GOT_KILLED))


def reward_from_events(events: Sequence[str]) -> float:
    """Convert framework events into the shared base-v1 scalar reward.

    Repeated objective events are counted separately. Death is counted once
    when either or both framework death events are present.
    """
    reward = STEP_REWARD
    for event, event_reward in EVENT_REWARDS.items():
        reward += events.count(event) * event_reward
    if DEATH_EVENTS.intersection(events):
        reward += DEATH_PENALTY
    return float(reward)

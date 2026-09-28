"""Finite-horizon adversarial viability for an own-bomb danger interval."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

import settings as s

from .opponent_transitions import (
    ACTIONS, canonical_state_key, enumerate_opponent_transition_scenarios,
)
from .danger import blast_coords
from .temporal_safety_features import temporal_maps


class _BudgetExpired(RuntimeError):
    pass


@dataclass(frozen=True)
class ControllableSurvivalResult:
    """Actions proven safe against every modeled opponent transition."""

    proven_actions: tuple[str, ...]
    scenarios_evaluated: int
    states_evaluated: int
    timed_out: bool
    first_failing_profile: tuple[str, ...] | None = None
    first_failing_order: tuple[int, ...] | None = None


def danger_interval_steps(own_bomb: dict, *, placing_bomb: bool) -> int:
    """Return transitions through the own bomb's last dangerous flame."""
    if placing_bomb:
        # Placement step, four subsequent countdown transitions, and one
        # lingering-danger transition.
        return 6
    timer = own_bomb.get("timer")
    if timer is not None:
        return max(0, int(timer) + 2)
    if own_bomb.get("pending"):
        # The bomb is no longer visible; any remaining dangerous flame is
        # represented by the current explosion map and lasts at most one step
        # in the official configuration.
        return 1
    return 0


def _opponent_reach_masks(blocked, others, steps, consider_opponent_rearming):
    """Exact union of named reach sets as two non-wrapping grid frontiers.

    Every name uses the same transition mask, so neighbour expansion and
    intersection distribute over union. Keep the named initialization to
    preserve existing handling even if a diagnostic input repeats a name.
    """
    reach = np.zeros_like(blocked, dtype=bool)
    armed = np.zeros_like(blocked, dtype=bool)
    if not others:
        return reach, armed
    named = {str(other[0]): tuple(other[3]) for other in others}
    armed_names = {str(other[0]) for other in others
                   if consider_opponent_rearming or bool(other[2])}
    for other in others:
        reach[0][tuple(other[3])] = True
        if consider_opponent_rearming or bool(other[2]):
            armed[0][tuple(other[3])] = True
    frontier = np.zeros((2, *blocked.shape[1:]), dtype=bool)
    for name, position in named.items():
        frontier[0][position] = True
        if name in armed_names:
            frontier[1][position] = True
    for time_step in range(1, steps + 1):
        following = frontier.copy()
        following[:, 1:, :] |= frontier[:, :-1, :]
        following[:, :-1, :] |= frontier[:, 1:, :]
        following[:, :, 1:] |= frontier[:, :, :-1]
        following[:, :, :-1] |= frontier[:, :, 1:]
        following &= ~blocked[time_step]
        reach[time_step] = following[0]
        armed[time_step] = following[1]
        frontier = following
    return reach, armed


def controllable_survival_actions(
    game_state: dict,
    candidate_actions: tuple[str, ...],
    *,
    remaining_steps: int,
    budget_ms: int,
    consider_opponent_rearming: bool = False,
    work_stats: dict | None = None,
) -> ControllableSurvivalResult:
    """Solve ``exists own action / forall scenarios`` conservatively.

    The first official transition is enumerated exactly. Remaining histories
    are represented by a sound over-approximation of opponent occupancy and
    bomb danger, then solved by a backward feedback-viability pass. An
    unfinished proof is fail-closed: only fully verified actions are returned.
    """
    deadline = monotonic() + max(0, budget_ms) / 1000.0
    scenarios_evaluated = 0
    states_evaluated = 0
    first_failure = [None, None]

    def check_budget():
        if monotonic() >= deadline:
            raise _BudgetExpired

    from .compact_survival import TransitionContext, ViabilityContext
    stats = {} if work_stats is None else work_stats

    def action_winning(state: dict, action: str, steps: int) -> bool:
        nonlocal scenarios_evaluated, states_evaluated
        for own, others, bombs, alive, profile, order in transitions.scenarios(action, check_budget):
            check_budget()
            scenarios_evaluated += 1
            winning, examined = viability.winning(own, others, bombs) if alive else (False, 0)
            states_evaluated += examined
            if not alive or not winning:
                if first_failure[0] is None:
                    first_failure[:] = profile, order
                return False
        return True

    proven = []
    timed_out = False
    try:
        check_budget()
        transitions = TransitionContext(game_state, stats)
        viability = ViabilityContext(transitions, remaining_steps - 1,
                                     consider_opponent_rearming, check_budget, stats)
        for action in candidate_actions:
            if action not in ACTIONS:
                raise ValueError(f"unsupported candidate action: {action!r}")
            if action_winning(game_state, action, remaining_steps):
                proven.append(action)
    except _BudgetExpired:
        timed_out = True
    return ControllableSurvivalResult(
        tuple(proven), scenarios_evaluated, states_evaluated, timed_out,
        first_failure[0], first_failure[1],
    )

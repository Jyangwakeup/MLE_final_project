"""Finite-horizon adversarial viability for an own-bomb danger interval."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

import settings as s

from tests.reference_compact_transitions import (
    ACTIONS, canonical_state_key, enumerate_opponent_transition_scenarios,
)
from agent_code.team_agent.danger import blast_coords
from agent_code.team_agent.temporal_safety_features import temporal_maps


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

    abstract_memo: dict[tuple, tuple[np.ndarray, int]] = {}

    def abstract_feedback_winning(state: dict, steps: int) -> bool:
        """Sound, conservative viability kernel for the remaining interval.

        Enumerating complete opponent histories is exponential.  Instead we
        over-approximate, at each future time, every tile any opponent can
        occupy and every blast any still-armed opponent can introduce.  A
        backward reachability pass then asks whether the agent has a feedback
        action from every retained position.  Passing this stronger game is a
        valid certificate for all official histories; failing it merely
        rejects an action conservatively.
        """
        nonlocal states_evaluated
        check_budget()
        if steps <= 0:
            return True
        # With hypothetical_bomb=False, the whole-board recurrence depends on
        # terrain, timed hazards and opponents, but not the own starting tile
        # or bomb capacity. Normalize self only in the memo key; evaluate the
        # original state and query the resulting board at the actual position.
        own_position = tuple(state["self"][3])
        environment = {**state, "self": ("", 0, False, (0, 0))}
        key = (steps, canonical_state_key(environment))
        cached = abstract_memo.get(key)
        if cached is not None:
            states_evaluated += cached[1]
            return bool(cached[0][own_position])

        field = np.asarray(state["field"])
        danger, blocked = temporal_maps(
            state, max(steps, 1), hypothetical_bomb=False)
        # The standard map keeps current opponents blocked forever.  The
        # adversarial occupancy sets below replace that static approximation.
        for other in state["others"]:
            position = tuple(other[3])
            if field[position] == 0 and all(
                tuple(bomb_position) != position
                for bomb_position, _ in state["bombs"]
            ):
                blocked[:, position[0], position[1]] = False

        # First-transition capacity remains factual; the optional rearming
        # envelope permits every surviving opponent to recover afterwards.
        opponent_reach, armed_reach = _opponent_reach_masks(
            blocked, state["others"], steps, consider_opponent_rearming)

        # An opponent may place its one available bomb at any reachable tile.
        # Unioning those blasts is deliberately stronger than the real game:
        # it preserves safety while avoiding an exponential bomb-history tree.
        robust_danger = np.asarray(danger, dtype=bool).copy()
        for placement_time in range(1, steps + 1):
            explosion_time = placement_time + int(s.BOMB_TIMER)
            if explosion_time > steps:
                continue
            for position in np.argwhere(armed_reach[placement_time - 1]):
                for coordinate in blast_coords(field, position, s.BOMB_POWER):
                    robust_danger[explosion_time, coordinate[0], coordinate[1]] = True
                    if explosion_time + 1 <= steps:
                        robust_danger[
                            explosion_time + 1, coordinate[0], coordinate[1]
                        ] = True

        viable = np.zeros_like(robust_danger, dtype=bool)
        free_terminal = ~blocked[steps]
        free_terminal &= ~opponent_reach[steps]
        viable[steps] = free_terminal & ~robust_danger[steps]
        examined = int(np.count_nonzero(viable[steps]))
        for time_step in range(steps - 1, -1, -1):
            free_now = ~blocked[time_step]
            if time_step:
                free_now &= ~opponent_reach[time_step]
            allowed_now = free_now & ~robust_danger[time_step]
            # WAIT may remain on a cell even when entry is blocked next step;
            # movement must enter an unblocked next cell. Slice shifts preserve
            # board boundaries, unlike wrapping rolls, and are the exact OR of
            # the scalar five-action recurrence.
            following = viable[time_step + 1]
            enterable = following & ~blocked[time_step + 1]
            reachable = following.copy()
            reachable[1:, :] |= enterable[:-1, :]
            reachable[:-1, :] |= enterable[1:, :]
            reachable[:, 1:] |= enterable[:, :-1]
            reachable[:, :-1] |= enterable[:, 1:]
            viable[time_step] = allowed_now & reachable
            examined += int(np.count_nonzero(allowed_now))
            check_budget()
        board = viable[0].copy()
        abstract_memo[key] = (board, examined)
        states_evaluated += examined
        return bool(board[own_position])

    def action_winning(state: dict, action: str, steps: int) -> bool:
        nonlocal scenarios_evaluated
        check_budget()
        scenarios = enumerate_opponent_transition_scenarios(
            state, action, progress_world=True)
        for scenario in scenarios:
            check_budget()
            scenarios_evaluated += 1
            if not scenario.self_alive or not abstract_feedback_winning(
                scenario.game_state, steps - 1
            ):
                if first_failure[0] is None:
                    first_failure[0] = scenario.action_profile
                    first_failure[1] = scenario.execution_order
                return False
        return True

    proven = []
    timed_out = False
    try:
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

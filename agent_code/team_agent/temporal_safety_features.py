from collections import namedtuple
from dataclasses import dataclass

import numpy as np

import settings as s
from .danger import HORIZON, blast_coords, predict_danger


WALL = -1
CRATE = 1
FREE = 0
ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT')
DIRECTIONS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'WAIT': (0, 0),
}

MoveFeatures = namedtuple('MoveFeatures', (
    'legal', 'safe_next', 'escape_exists', 'escape_area', 'safe_horizon'
))
BombFeatures = namedtuple('BombFeatures', (
    'can_drop_bomb', 'escape_after_bomb', 'escape_area_after_bomb', 'crates_hit'
))
TemporalSafetyAnalysis = namedtuple('TemporalSafetyAnalysis', (
    'move_features', 'bomb_features'
))


@dataclass(frozen=True)
class ReachabilityResult:
    """Result of a time-expanded search after fixing the first action."""

    legal: bool
    safe_next: bool
    survives_horizon: bool
    safe_horizon: int
    reachable_positions: frozenset
    reachable_area: int


@dataclass(frozen=True)
class SafetyMarginResult:
    """Action-conditional escape slack and robust second-step choices."""

    escape_slack: float
    survivable_second_actions: int


def _bomb_explosion_time(timer: int) -> int:
    """Return the future step at which an existing bomb explodes."""
    return timer + 1


def _bomb_schedule(game_state: dict, hypothetical_bomb: bool):
    """Return bomb positions and their predicted explosion times."""
    schedule = [(position, _bomb_explosion_time(timer)) for position, timer in game_state['bombs']]
    if hypothetical_bomb and game_state['self'][2]:
        schedule.append((game_state['self'][3], _bomb_explosion_time(s.BOMB_TIMER)))
    return schedule


def _initial_blocked(field: np.ndarray, game_state: dict, horizon: int):
    """Build the time-indexed occupancy map used before each movement."""
    blocked = np.broadcast_to(field != FREE, (horizon + 1, *field.shape)).copy()

    for _, _, _, position in game_state['others']:
        x, y = position
        blocked[:, x, y] = True

    return blocked


def _mark_destroyed_crates(blocked: np.ndarray, field: np.ndarray, schedule):
    """Clear crates from the step after their first predicted destruction."""
    destroyed_at = {}
    for position, explosion_time in schedule:
        for x, y in blast_coords(field, position, s.BOMB_POWER):
            if field[x, y] == CRATE:
                destroyed_at[(x, y)] = min(destroyed_at.get((x, y), explosion_time), explosion_time)

    for (x, y), explosion_time in destroyed_at.items():
        blocked[explosion_time + 1:, x, y] = False


def _mark_bomb_occupancy(blocked: np.ndarray, schedule):
    """Keep a bomb blocked through its explosion step."""
    horizon = blocked.shape[0] - 1
    for (x, y), explosion_time in schedule:
        blocked[1:min(explosion_time, horizon) + 1, x, y] = True


def _temporal_maps(game_state: dict, horizon: int, hypothetical_bomb: bool):
    """Return predicted danger and movement occupancy for one analysis."""
    prediction = predict_danger(game_state, hypothetical_bomb=hypothetical_bomb, horizon=horizon)
    schedule = _bomb_schedule(game_state, hypothetical_bomb)
    blocked = _initial_blocked(game_state['field'], game_state, horizon)
    _mark_destroyed_crates(blocked, game_state['field'], schedule)
    _mark_bomb_occupancy(blocked, schedule)
    return prediction.danger, blocked


def temporal_maps(game_state: dict, horizon: int, hypothetical_bomb: bool):
    """Public shared danger/occupancy maps for versioned feature extractors."""
    return _temporal_maps(game_state, horizon, hypothetical_bomb)


def _in_bounds(position: tuple, shape: tuple) -> bool:
    """Check whether a board position can be indexed."""
    x, y = position
    return 0 <= x < shape[0] and 0 <= y < shape[1]


def _next_position(position: tuple, action: str) -> tuple:
    """Apply one movement or wait action to a position."""
    dx, dy = DIRECTIONS[action]
    return position[0] + dx, position[1] + dy


def _can_enter(position: tuple, time_step: int, blocked: np.ndarray) -> bool:
    """Check whether a movement can enter a tile before that step's update."""
    return _in_bounds(position, blocked.shape[1:]) and not blocked[time_step, position[0], position[1]]


def _advance(position: tuple, action: str, time_step: int, danger: np.ndarray, blocked: np.ndarray):
    """Return the safe successor position, or None if this action cannot be used."""
    next_position = _next_position(position, action)
    if not _in_bounds(next_position, blocked.shape[1:]):
        return None
    if action != 'WAIT' and not _can_enter(next_position, time_step, blocked):
        return None
    if danger[time_step, next_position[0], next_position[1]]:
        return None
    return next_position


def detailed_reachability_after_first_step(
        position: tuple,
        first_action: str,
        danger: np.ndarray,
        blocked: np.ndarray,
) -> ReachabilityResult:
    """Search safe position-time states and retain the last safe frontier."""
    if first_action not in ACTIONS:
        raise ValueError(f'unsupported first action: {first_action!r}')

    legal = first_action == 'WAIT' or _can_enter(
        _next_position(position, first_action), 1, blocked)
    first_position = _advance(position, first_action, 1, danger, blocked)
    if first_position is None:
        return ReachabilityResult(legal, False, False, 0, frozenset(), 0)

    horizon = danger.shape[0] - 1
    current = {first_position}
    safe_horizon = 1
    for time_step in range(2, horizon + 1):
        following = set()
        for current_position in current:
            for action in ACTIONS:
                next_position = _advance(current_position, action, time_step, danger, blocked)
                if next_position is not None:
                    following.add(next_position)
        if not following:
            positions = frozenset(current)
            return ReachabilityResult(
                legal, True, False, safe_horizon, positions, len(positions))
        current = following
        safe_horizon = time_step

    positions = frozenset(current)
    return ReachabilityResult(
        legal, True, True, safe_horizon, positions, len(positions))


def _survives_from_frontier(
        frontier: set[tuple[int, int]], start_time: int,
        danger: np.ndarray, blocked: np.ndarray,
) -> bool:
    current = set(frontier)
    for time_step in range(start_time, danger.shape[0]):
        following = set()
        for position in current:
            for action in ACTIONS:
                successor = _advance(position, action, time_step, danger, blocked)
                if successor is not None:
                    following.add(successor)
        if not following:
            return False
        current = following
    return bool(current)


def safety_margin_after_first_step(
        position: tuple, first_action: str,
        danger: np.ndarray, blocked: np.ndarray,
) -> SafetyMarginResult:
    """Summarize escape timing after fixing one first action."""
    first_position = _advance(position, first_action, 1, danger, blocked)
    if first_position is None:
        return SafetyMarginResult(-1.0, 0)
    horizon = danger.shape[0] - 1
    deadline = horizon + 1
    for time_step in range(1, horizon + 1):
        if danger[time_step, first_position[0], first_position[1]]:
            deadline = time_step
            break

    frontier = {first_position}
    escape_time = None
    for time_step in range(1, horizon + 1):
        if any(not danger[time_step:, x, y].any() for x, y in frontier):
            escape_time = time_step
            break
        if time_step == horizon:
            break
        following = set()
        for current in frontier:
            for action in ACTIONS:
                successor = _advance(
                    current, action, time_step + 1, danger, blocked)
                if successor is not None:
                    following.add(successor)
        frontier = following
        if not frontier:
            break
    if escape_time is None:
        slack = -1.0
    else:
        slack = float(np.clip((deadline - escape_time) / horizon, -1.0, 1.0))

    second_actions = 0
    if horizon == 1:
        second_actions = len(ACTIONS)
    else:
        for action in ACTIONS:
            second = _advance(first_position, action, 2, danger, blocked)
            if second is not None and (
                horizon == 2 or _survives_from_frontier(
                    {second}, 3, danger, blocked)
            ):
                second_actions += 1
    return SafetyMarginResult(slack, second_actions)


def detailed_reachability_all_first_steps(
        position: tuple,
        danger: np.ndarray,
        blocked: np.ndarray,
) -> dict[str, ReachabilityResult]:
    """Evaluate all five first actions in one equivalent grid propagation.

    The first-step legality rules and the special WAIT occupancy semantics are
    identical to :func:`detailed_reachability_after_first_step`.  Batching only
    removes repeated Python traversal of the same time-indexed board.
    """
    width, height = blocked.shape[1:]
    frontiers = np.zeros((len(ACTIONS), width, height), dtype=bool)
    legal = np.zeros(len(ACTIONS), dtype=bool)
    alive = np.zeros(len(ACTIONS), dtype=bool)
    safe_horizons = np.zeros(len(ACTIONS), dtype=np.int16)
    last_frontiers = np.zeros_like(frontiers)

    for index, action in enumerate(ACTIONS):
        legal[index] = action == 'WAIT' or _can_enter(
            _next_position(position, action), 1, blocked)
        first_position = _advance(position, action, 1, danger, blocked)
        if first_position is not None:
            frontiers[index, first_position[0], first_position[1]] = True
            last_frontiers[index] = frontiers[index]
            alive[index] = True
            safe_horizons[index] = 1

    horizon = danger.shape[0] - 1
    for time_step in range(2, horizon + 1):
        if not alive.any():
            break
        moved = np.zeros_like(frontiers)
        moved[:, :, :-1] |= frontiers[:, :, 1:]
        moved[:, 1:, :] |= frontiers[:, :-1, :]
        moved[:, :, 1:] |= frontiers[:, :, :-1]
        moved[:, :-1, :] |= frontiers[:, 1:, :]
        safe = ~danger[time_step]
        following = (frontiers & safe) | (
            moved & safe & ~blocked[time_step])
        following[~alive] = False
        has_following = following.reshape(len(ACTIONS), -1).any(axis=1)
        survivors = alive & has_following
        last_frontiers[survivors] = following[survivors]
        safe_horizons[survivors] = time_step
        alive &= has_following
        frontiers = following

    results = {}
    for index, action in enumerate(ACTIONS):
        coordinates = frozenset(
            (int(x), int(y)) for x, y in np.argwhere(last_frontiers[index]))
        safe_horizon = int(safe_horizons[index])
        results[action] = ReachabilityResult(
            bool(legal[index]), safe_horizon >= 1,
            safe_horizon == horizon, safe_horizon,
            coordinates, len(coordinates),
        )
    return results


def _reachable_after_first_step(position: tuple, first_action: str, danger: np.ndarray, blocked: np.ndarray):
    """Legacy search result, preserving the original failed-escape semantics."""
    result = detailed_reachability_after_first_step(
        position, first_action, danger, blocked)
    final_positions = set(result.reachable_positions) if result.survives_horizon else set()
    return result.safe_horizon, final_positions


def _move_features(position: tuple, action: str, danger: np.ndarray, blocked: np.ndarray) -> MoveFeatures:
    """Build objective safety features for one fixed first action."""
    legal = action == 'WAIT' or _can_enter(_next_position(position, action), 1, blocked)
    next_position = _advance(position, action, 1, danger, blocked)
    if next_position is None:
        return MoveFeatures(legal, False, False, 0, 0)

    safe_horizon, final_positions = _reachable_after_first_step(position, action, danger, blocked)
    horizon = danger.shape[0] - 1
    escape_exists = safe_horizon == horizon
    return MoveFeatures(
        legal,
        True,
        escape_exists,
        len(final_positions) if escape_exists else 0,
        safe_horizon,
    )


def _bomb_features(game_state: dict, horizon: int) -> BombFeatures:
    """Build objective safety and crate coverage features for a hypothetical bomb."""
    can_drop_bomb = bool(game_state['self'][2])
    if not can_drop_bomb:
        return BombFeatures(False, False, 0, 0)

    field = game_state['field']
    position = game_state['self'][3]
    crates_hit = sum(field[x, y] == CRATE for x, y in blast_coords(field, position, s.BOMB_POWER))
    danger, blocked = _temporal_maps(game_state, horizon, hypothetical_bomb=True)
    safe_horizon, final_positions = _reachable_after_first_step(position, 'WAIT', danger, blocked)
    escape_after_bomb = safe_horizon == horizon
    return BombFeatures(
        True,
        escape_after_bomb,
        len(final_positions) if escape_after_bomb else 0,
        crates_hit,
    )


def _temporal_safety_features_with_danger(game_state: dict, horizon=None):
    """Internal analysis that also exposes the already-computed danger map."""
    if game_state is None:
        return None, None
    if horizon is None:
        horizon = HORIZON
    if horizon <= 0:
        raise ValueError('horizon must be positive')

    position = game_state['self'][3]
    danger, blocked = _temporal_maps(game_state, horizon, hypothetical_bomb=False)
    reachability = detailed_reachability_all_first_steps(
        position, danger, blocked)
    move_features = {
        action: MoveFeatures(
            result.legal,
            result.safe_next,
            result.survives_horizon,
            result.reachable_area if result.survives_horizon else 0,
            result.safe_horizon,
        )
        for action, result in reachability.items()
    }
    return (
        TemporalSafetyAnalysis(move_features, _bomb_features(game_state, horizon)),
        danger,
    )


def temporal_safety_features(game_state: dict, horizon=None):
    """Return objective time-aware movement and bomb safety features."""
    analysis, _ = _temporal_safety_features_with_danger(game_state, horizon)
    return analysis

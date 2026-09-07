from collections import deque, namedtuple

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


def _reachable_after_first_step(position: tuple, first_action: str, danger: np.ndarray, blocked: np.ndarray):
    """Search safe position-time states after a fixed first action."""
    first_position = _advance(position, first_action, 1, danger, blocked)
    if first_position is None:
        return 0, set()

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
            return safe_horizon, set()
        current = following
        safe_horizon = time_step

    return safe_horizon, current


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


def temporal_safety_features(game_state: dict, horizon=None):
    """Return objective time-aware movement and bomb safety features."""
    if game_state is None:
        return None
    if horizon is None:
        horizon = HORIZON
    if horizon <= 0:
        raise ValueError('horizon must be positive')

    position = game_state['self'][3]
    danger, blocked = _temporal_maps(game_state, horizon, hypothetical_bomb=False)
    move_features = {
        action: _move_features(position, action, danger, blocked)
        for action in ACTIONS
    }
    return TemporalSafetyAnalysis(move_features, _bomb_features(game_state, horizon))

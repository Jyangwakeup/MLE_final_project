from collections import deque, namedtuple

import numpy as np

from .danger import predict_danger
from .temporal_safety_features import temporal_safety_features


ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
MOVE_ACTIONS = ACTIONS[:4]
DIRECTIONS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}
FREE = 0

SafetyFeatures = namedtuple('SafetyFeatures', ('state_key', 'vector', 'legal_mask'))
CoinFeatures = namedtuple('CoinFeatures', ('state_key', 'vector'))


def _in_bounds(position: tuple, shape: tuple) -> bool:
    """Check whether a board position can be indexed."""
    x, y = position
    return 0 <= x < shape[0] and 0 <= y < shape[1]


def _legal_mask(game_state: dict) -> np.ndarray:
    """Return physical action legality without applying danger filtering."""
    field = game_state['field']
    x, y = game_state['self'][3]
    occupied = {position for position, _ in game_state['bombs']}
    occupied.update(position for _, _, _, position in game_state['others'])

    legal = []
    for action in MOVE_ACTIONS:
        dx, dy = DIRECTIONS[action]
        position = x + dx, y + dy
        legal.append(
            _in_bounds(position, field.shape)
            and field[position[0], position[1]] == FREE
            and position not in occupied
        )

    legal.extend([True, bool(game_state['self'][2])])
    return np.array(legal, dtype=bool)


def _movement_category(legal: bool, escape_exists: bool) -> int:
    """Encode physical legality and predicted escape as one three-class value."""
    if not legal:
        return 0
    if not escape_exists:
        return 1
    return 2


def _current_danger_category(game_state: dict) -> int:
    """Encode whether the current tile is dangerous now, later, or never."""
    danger = predict_danger(game_state).danger
    x, y = game_state['self'][3]
    if danger[1, x, y]:
        return 1
    if danger[2:, x, y].any():
        return 2
    return 0


def _bomb_category(can_drop_bomb: bool, escape_after_bomb: bool) -> int:
    """Encode bomb availability and whether the hypothetical bomb is escapable."""
    if not can_drop_bomb:
        return 0
    if not escape_after_bomb:
        return 1
    return 2


def _crate_category(crates_hit: int) -> int:
    """Encode the hypothetical bomb's crate coverage."""
    if crates_hit <= 0:
        return 0
    if crates_hit == 1:
        return 1
    return 2


def _one_hot(state_key: tuple) -> np.ndarray:
    """Convert seven three-class feature values into a float32[21] vector."""
    vector = np.zeros(21, dtype=np.float32)
    for index, value in enumerate(state_key):
        vector[index * 3 + value] = 1.0
    return vector


def _navigation_blocked(game_state: dict) -> np.ndarray:
    """Return current static obstacles while allowing the agent's start tile."""
    blocked = game_state['field'] != FREE
    blocked = blocked.copy()
    for position, _ in game_state['bombs']:
        blocked[position[0], position[1]] = True
    for _, _, _, position in game_state['others']:
        blocked[position[0], position[1]] = True

    x, y = game_state['self'][3]
    blocked[x, y] = False
    return blocked


def _distances(origin: tuple, blocked: np.ndarray) -> dict:
    """Return static BFS distances from one traversable origin."""
    if not _in_bounds(origin, blocked.shape) or blocked[origin[0], origin[1]]:
        return {}

    distances = {origin: 0}
    frontier = deque([origin])
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in DIRECTIONS.values():
            position = x + dx, y + dy
            if (
                _in_bounds(position, blocked.shape)
                and not blocked[position[0], position[1]]
                and position not in distances
            ):
                distances[position] = distances[(x, y)] + 1
                frontier.append(position)
    return distances


def _coin_one_hot(state_key: tuple) -> np.ndarray:
    """Convert four binary coin fields into a float32[8] vector."""
    vector = np.zeros(8, dtype=np.float32)
    for index, value in enumerate(state_key):
        vector[index * 2 + value] = 1.0
    return vector


def coin_features(game_state: dict):
    """Return objective static-distance features for the nearest reachable coin."""
    if game_state is None:
        return None

    position = game_state['self'][3]
    blocked = _navigation_blocked(game_state)
    from_self = _distances(position, blocked)
    candidates = [
        (from_self[coin], coin[0], coin[1])
        for coin in game_state['coins']
        if coin in from_self
    ]
    if not candidates:
        state_key = (0, 0, 0, 0)
        return CoinFeatures(state_key, _coin_one_hot(state_key))

    _, target_x, target_y = min(candidates)
    target = target_x, target_y
    to_target = _distances(target, blocked)
    current_distance = to_target[position]
    legal_mask = _legal_mask(game_state)

    state_key = []
    for index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        candidate = position[0] + dx, position[1] + dy
        is_closer = (
            legal_mask[index]
            and candidate in to_target
            and to_target[candidate] < current_distance
        )
        state_key.append(int(is_closer))

    state_key = tuple(state_key)
    return CoinFeatures(state_key, _coin_one_hot(state_key))


def safety_features(game_state: dict):
    """Return objective safety features without selecting an action."""
    if game_state is None:
        return None

    analysis = temporal_safety_features(game_state)
    legal_mask = _legal_mask(game_state)
    movement = tuple(
        _movement_category(legal_mask[index], analysis.move_features[action].escape_exists)
        for index, action in enumerate(MOVE_ACTIONS)
    )
    current_danger = _current_danger_category(game_state)
    bomb = _bomb_category(
        analysis.bomb_features.can_drop_bomb,
        analysis.bomb_features.escape_after_bomb,
    )
    crates = _crate_category(analysis.bomb_features.crates_hit)
    state_key = movement + (current_danger, bomb, crates)

    return SafetyFeatures(state_key, _one_hot(state_key), legal_mask)

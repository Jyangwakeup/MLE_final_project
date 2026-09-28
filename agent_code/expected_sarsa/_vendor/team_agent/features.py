from collections import deque, namedtuple

import numpy as np
import settings as s

from .danger import blast_coords, predict_danger
from .feature_system.types import DiscreteFeatures, IDENTITY_ACTION_TRANSFORM
from .temporal_safety_features import _temporal_safety_features_with_danger


ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
FEATURE_VERSION = 'v1'
FEATURE_ID = 'discrete-v1'
STATE_KEY_SIZE = 14
FEATURE_CATEGORY_COUNTS = (
    3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2,
    5, 4, 2,
)
FEATURE_DIM = sum(FEATURE_CATEGORY_COUNTS)
MOVE_ACTIONS = ACTIONS[:4]
DIRECTIONS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}
FREE = 0

Features = DiscreteFeatures
SafetyFeatures = namedtuple('SafetyFeatures', ('state_key', 'vector', 'legal_mask'))
CoinFeatures = namedtuple('CoinFeatures', ('state_key', 'vector'))
OpponentFeatures = namedtuple('OpponentFeatures', ('state_key', 'vector'))


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
        for position in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
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


def _nearest_opponent(game_state: dict):
    """Return the nearest opponent using the fixed deterministic ordering."""
    x, y = game_state['self'][3]
    if not game_state['others']:
        return None
    return min(
        game_state['others'],
        key=lambda other: (
            abs(other[3][0] - x) + abs(other[3][1] - y),
            other[3],
            other[0],
        ),
    )


def _opponent_direction(origin: tuple, target: tuple) -> int:
    """Encode the dominant displacement axis with fixed tie-breaking."""
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4
    if abs(dy) > abs(dx):
        return 3 if dy > 0 else 1

    if dy < 0:
        return 1
    if dx > 0:
        return 2
    if dy > 0:
        return 3
    return 4


def _opponent_distance(distance: int) -> int:
    """Encode an opponent's Manhattan distance into the fixed four classes."""
    if distance == 1:
        return 1
    if distance <= 4:
        return 2
    return 3


def _opponent_one_hot(state_key: tuple) -> np.ndarray:
    """Convert direction, distance, and blast coverage into float32[11]."""
    vector = np.zeros(11, dtype=np.float32)
    offset = 0
    for value, size in zip(state_key, (5, 4, 2)):
        vector[offset + value] = 1.0
        offset += size
    return vector


def opponent_features(game_state: dict):
    """Return objective spatial features for the currently visible opponents."""
    if game_state is None:
        return None

    nearest = _nearest_opponent(game_state)
    if nearest is None:
        state_key = (0, 0, 0)
        return OpponentFeatures(state_key, _opponent_one_hot(state_key))

    position = game_state['self'][3]
    target = nearest[3]
    distance = abs(target[0] - position[0]) + abs(target[1] - position[1])
    blast = set(blast_coords(game_state['field'], position, s.BOMB_POWER))
    opponent_covered = any(other[3] in blast for other in game_state['others'])
    state_key = (
        _opponent_direction(position, target),
        _opponent_distance(distance),
        int(opponent_covered),
    )
    return OpponentFeatures(state_key, _opponent_one_hot(state_key))


def _coin_features_with_distance(game_state: dict):
    """Return frozen coin features and the distance already found by their BFS."""
    if game_state is None:
        return None, np.inf

    position = game_state['self'][3]
    # Lazy import avoids a package-initialisation cycle with compatibility
    # feature adapters that import this frozen baseline extractor.
    from .feature_system.common import distance_to_targets
    blocked = _navigation_blocked(game_state)
    from_self = distance_to_targets(blocked, (position,))
    candidates = [
        (float(from_self[coin]), coin[0], coin[1])
        for coin in game_state['coins']
        if np.isfinite(from_self[coin])
    ]
    if not candidates:
        state_key = (0, 0, 0, 0)
        return CoinFeatures(state_key, _coin_one_hot(state_key)), np.inf

    nearest_distance, target_x, target_y = min(candidates)
    target = target_x, target_y
    to_target = distance_to_targets(blocked, (target,))
    current_distance = to_target[position]
    legal_mask = _legal_mask(game_state)

    state_key = []
    for index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DIRECTIONS[action]
        candidate = position[0] + dx, position[1] + dy
        is_closer = (
            legal_mask[index]
            and np.isfinite(to_target[candidate])
            and to_target[candidate] < current_distance
        )
        state_key.append(int(is_closer))

    state_key = tuple(state_key)
    return CoinFeatures(state_key, _coin_one_hot(state_key)), nearest_distance


def coin_features(game_state: dict):
    """Return objective static-distance features for the nearest reachable coin."""
    features, _ = _coin_features_with_distance(game_state)
    return features


def safety_features(game_state: dict):
    """Return objective safety features without selecting an action."""
    if game_state is None:
        return None

    analysis, danger = _temporal_safety_features_with_danger(game_state)
    legal_mask = _legal_mask(game_state)
    movement = tuple(
        _movement_category(legal_mask[index], analysis.move_features[action].escape_exists)
        for index, action in enumerate(MOVE_ACTIONS)
    )
    x, y = game_state['self'][3]
    if danger[1, x, y]:
        current_danger = 1
    elif danger[2:, x, y].any():
        current_danger = 2
    else:
        current_danger = 0
    bomb = _bomb_category(
        analysis.bomb_features.can_drop_bomb,
        analysis.bomb_features.escape_after_bomb,
    )
    crates = _crate_category(analysis.bomb_features.crates_hit)
    state_key = movement + (current_danger, bomb, crates)

    return SafetyFeatures(state_key, _one_hot(state_key), legal_mask)


def _extract_features_with_coin_distance(game_state: dict):
    """Build the frozen representation and retain its already-computed distance."""
    if game_state is None:
        return None, np.inf

    safety = safety_features(game_state)
    coins, coin_distance = _coin_features_with_distance(game_state)
    opponents = opponent_features(game_state)
    state_key = safety.state_key + coins.state_key + opponents.state_key
    vector = np.concatenate((safety.vector, coins.vector, opponents.vector))
    return Features(
        feature_id=FEATURE_ID,
        state_key=state_key,
        vector=vector,
        legal_mask=safety.legal_mask,
        action_transform=IDENTITY_ACTION_TRANSFORM,
    ), coin_distance


def extract_features(game_state: dict):
    """Return the formal feature representation composed from all feature groups."""
    features, _ = _extract_features_with_coin_distance(game_state)
    return features

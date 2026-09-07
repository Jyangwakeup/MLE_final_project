from collections import namedtuple

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

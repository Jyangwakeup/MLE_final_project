"""State representation and action-legality helpers for tabular Q-learning."""

import numpy as np

import settings as s


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def legal_actions(game_state: dict, allow_bomb: bool = True) -> np.ndarray:
    """Return a boolean mask in UP, RIGHT, DOWN, LEFT, WAIT, BOMB order."""
    field = game_state["field"]
    x, y = game_state["self"][3]
    occupied = {position for position, _ in game_state["bombs"]}
    occupied.update(other[3] for other in game_state["others"])
    result = []
    for dx, dy in DIRECTIONS:
        xx, yy = x + dx, y + dy
        result.append(
            0 <= xx < field.shape[0] and 0 <= yy < field.shape[1]
            and field[xx, yy] == 0 and (xx, yy) not in occupied
        )
    result.extend((True, allow_bomb and bool(game_state["self"][2])))
    return np.asarray(result, dtype=bool)


def state_to_features(game_state: dict):
    """Convert a game state into a small, hashable state for tabular learning.

    The representation describes adjacent tiles, imminent blast danger at the
    current/adjacent tiles, directions to nearby objectives, and bomb ability.
    Relative directions make experience transferable across board positions.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    x, y = game_state["self"][3]
    bombs = {position for position, _ in game_state["bombs"]}
    opponents = {other[3] for other in game_state["others"]}

    adjacent = []
    positions = [(x + dx, y + dy) for dx, dy in DIRECTIONS]
    for xx, yy in positions:
        if not (0 <= xx < field.shape[0] and 0 <= yy < field.shape[1]):
            adjacent.append(-1)
        elif (xx, yy) in bombs or (xx, yy) in opponents:
            adjacent.append(2)  # temporarily occupied
        else:
            adjacent.append(int(field[xx, yy]))  # -1 wall, 0 free, 1 crate

    danger = _danger_map(game_state)
    danger_features = tuple(
        _danger_bucket(danger[px, py]) for px, py in [(x, y)] + positions
    )
    crates = list(zip(*np.where(field == 1)))
    coin_direction = _direction_to_nearest((x, y), game_state["coins"])
    crate_direction = _direction_to_nearest((x, y), crates)
    opponent_direction = _direction_to_nearest((x, y), list(opponents))

    return (
        *adjacent,
        *danger_features,
        *coin_direction,
        *crate_direction,
        *opponent_direction,
        int(bool(game_state["self"][2])),
    )


def _direction_to_nearest(origin, targets):
    if not targets:
        return (0, 0)
    ox, oy = origin
    tx, ty = min(
        targets,
        key=lambda point: abs(point[0] - ox) + abs(point[1] - oy),
    )
    return (int(np.sign(tx - ox)), int(np.sign(ty - oy)))


def _danger_bucket(value):
    if not np.isfinite(value):
        return 0
    return 1 if value <= 1 else 2 if value <= 2 else 3


def _danger_map(game_state):
    """Earliest bomb timer affecting each tile; walls/crates block blasts."""
    field = game_state["field"]
    danger = np.full(field.shape, np.inf)
    danger[game_state["explosion_map"] > 0] = 0
    for (bx, by), timer in game_state["bombs"]:
        danger[bx, by] = min(danger[bx, by], timer)
        for dx, dy in DIRECTIONS:
            for distance in range(1, s.BOMB_POWER + 1):
                xx, yy = bx + dx * distance, by + dy * distance
                if not (0 <= xx < field.shape[0] and 0 <= yy < field.shape[1]):
                    break
                if field[xx, yy] == -1:
                    break
                danger[xx, yy] = min(danger[xx, yy], timer)
                if field[xx, yy] == 1:
                    break
    return danger

from collections import namedtuple

import numpy as np

import settings as s


WALL = -1
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1

DangerPrediction = namedtuple('DangerPrediction', ('danger',))


def blast_coords(field: np.ndarray, position: tuple, power: int):
    """Return the tiles reached by one bomb."""
    x, y = position
    coords = [(x, y)]

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, power + 1):
            xx = x + distance * dx
            yy = y + distance * dy
            if xx < 0 or xx >= field.shape[0] or yy < 0 or yy >= field.shape[1]:
                break
            if field[xx, yy] == WALL:
                break
            coords.append((xx, yy))

    return coords


def _mark_bomb_danger(danger: np.ndarray, field: np.ndarray, position: tuple, timer: int, horizon: int):
    """Mark the two dangerous steps caused by one bomb."""
    blast_time = timer + 1
    if blast_time > horizon:
        return

    final_time = min(blast_time + s.EXPLOSION_TIMER - 1, horizon)
    for x, y in blast_coords(field, position, s.BOMB_POWER):
        danger[blast_time:final_time + 1, x, y] = True


def _mark_existing_explosions(danger: np.ndarray, explosion_map: np.ndarray, horizon: int):
    """Mark dangerous explosion tiles already present in the game state."""
    for x, y in np.argwhere(explosion_map > 0):
        remaining_steps = min(int(explosion_map[x, y]), horizon)
        danger[1:remaining_steps + 1, x, y] = True


def predict_danger(game_state: dict, hypothetical_bomb=False, horizon=None) -> DangerPrediction:
    """Predict future blast danger without changing the supplied game state."""
    if horizon is None:
        horizon = HORIZON
    if horizon <= 0:
        raise ValueError('horizon must be positive')

    field = game_state['field']
    danger = np.zeros((horizon + 1, *field.shape), dtype=bool)

    _mark_existing_explosions(danger, game_state['explosion_map'], horizon)

    for position, timer in game_state['bombs']:
        _mark_bomb_danger(danger, field, position, timer, horizon)

    if hypothetical_bomb and game_state['self'][2]:
        _mark_bomb_danger(
            danger, field, game_state['self'][3], s.BOMB_TIMER, horizon)

    return DangerPrediction(danger)

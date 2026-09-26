"""Shape-preserving board symmetries and their action permutations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np

from .types import ActionTransform


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
DIRECTION_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}


@dataclass(frozen=True)
class Symmetry:
    name: str
    width: int
    height: int
    coordinate: Callable[[tuple[int, int]], tuple[int, int]]
    array_transform: Callable[[np.ndarray], np.ndarray]
    action_transform: ActionTransform

    def transform_array(self, array: np.ndarray) -> np.ndarray:
        """Apply this symmetry with vectorized NumPy operations.

        Returning an independent contiguous array preserves the old method's
        ownership and byte-signature semantics without its Python pixel loop.
        """
        return np.array(self.array_transform(array), copy=True, order="C")


def _action_transform(
    coordinate: Callable[[tuple[int, int]], tuple[int, int]],
    width: int,
    height: int,
) -> ActionTransform:
    origin = (width // 2, height // 2)
    transformed_origin = coordinate(origin)
    world_to_canonical = []
    for action in ACTIONS:
        if action in {"WAIT", "BOMB"}:
            world_to_canonical.append(ACTIONS.index(action))
            continue
        dx, dy = DIRECTION_DELTAS[action]
        transformed = coordinate((origin[0] + dx, origin[1] + dy))
        transformed_delta = (
            transformed[0] - transformed_origin[0],
            transformed[1] - transformed_origin[1],
        )
        mapped = next(
            candidate for candidate in ACTIONS[:4]
            if DIRECTION_DELTAS[candidate] == transformed_delta
        )
        world_to_canonical.append(ACTIONS.index(mapped))
    inverse = [0] * len(ACTIONS)
    for world_index, canonical_index in enumerate(world_to_canonical):
        inverse[canonical_index] = world_index
    return ActionTransform(tuple(world_to_canonical), tuple(inverse))


@lru_cache(maxsize=8)
def symmetries(width: int, height: int) -> tuple[Symmetry, ...]:
    """Return every board symmetry that preserves the current array shape."""
    transforms = [
        ("identity", lambda p: p, lambda a: a),
        ("rotate_180", lambda p: (width - 1 - p[0], height - 1 - p[1]),
         lambda a: np.rot90(a, 2)),
        ("mirror_x", lambda p: (width - 1 - p[0], p[1]),
         lambda a: np.flip(a, axis=0)),
        ("mirror_y", lambda p: (p[0], height - 1 - p[1]),
         lambda a: np.flip(a, axis=1)),
    ]
    if width == height:
        size = width
        transforms = [
            ("identity", lambda p: p, lambda a: a),
            ("rotate_90", lambda p: (size - 1 - p[1], p[0]),
             lambda a: np.rot90(a, 1)),
            ("rotate_180", lambda p: (size - 1 - p[0], size - 1 - p[1]),
             lambda a: np.rot90(a, 2)),
            ("rotate_270", lambda p: (p[1], size - 1 - p[0]),
             lambda a: np.rot90(a, 3)),
            ("mirror_x", lambda p: (size - 1 - p[0], p[1]),
             lambda a: np.flip(a, axis=0)),
            ("rotate_90_after_mirror_x",
             lambda p: (size - 1 - p[1], size - 1 - p[0]),
             lambda a: np.rot90(np.flip(a, axis=0), 1)),
            ("rotate_180_after_mirror_x", lambda p: (p[0], size - 1 - p[1]),
             lambda a: np.flip(a, axis=1)),
            ("rotate_270_after_mirror_x", lambda p: (p[1], p[0]),
             lambda a: np.swapaxes(a, 0, 1)),
        ]
    return tuple(
        Symmetry(
            name, width, height, coordinate, array_transform,
            _action_transform(coordinate, width, height),
        )
        for name, coordinate, array_transform in transforms
    )


def canonical_symmetry(game_state: dict) -> Symmetry:
    """Select the deterministic lexicographically smallest objective state."""
    field = game_state["field"]
    width, height = field.shape
    coin_map = np.zeros(field.shape, dtype=np.int8)
    self_map = np.zeros(field.shape, dtype=np.int8)
    opponent_map = np.zeros(field.shape, dtype=np.int8)
    bomb_map = np.zeros(field.shape, dtype=np.int16)
    for x, y in game_state["coins"]:
        coin_map[x, y] = 1
    self_map[game_state["self"][3]] = 1
    for _, _, _, position in game_state["others"]:
        opponent_map[position] = 1
    for position, timer in game_state["bombs"]:
        bomb_map[position] = int(timer) + 1
    arrays = (
        np.asarray(field, dtype=np.int8), coin_map, self_map, opponent_map,
        bomb_map, np.asarray(game_state["explosion_map"], dtype=np.int16),
    )
    candidates = symmetries(width, height)

    def signature(transform: Symmetry) -> bytes:
        # ``tobytes(order='C')`` already materialises non-contiguous rotate/flip
        # views.  Avoid a second temporary contiguous array during comparison.
        return b"".join(
            transform.array_transform(array).tobytes(order="C") for array in arrays
        )

    return min(enumerate(candidates), key=lambda item: (signature(item[1]), item[0]))[1]


def transform_positions(
    positions: Iterable[tuple[int, int]], transform: Symmetry,
) -> tuple[tuple[int, int], ...]:
    return tuple(transform.coordinate(position) for position in positions)

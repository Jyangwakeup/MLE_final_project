"""D4 transforms for field[x, y] boards and the six-action contract."""

from __future__ import annotations

import numpy as np


TRANSFORMS = (
    "identity", "rot90", "rot180", "rot270",
    "flip", "flip_rot90", "flip_rot180", "flip_rot270",
)

# Original action index -> transformed action index. WAIT and BOMB are fixed.
_ROT90 = np.asarray((1, 2, 3, 0, 4, 5), dtype=np.int64)
_FLIP = np.asarray((0, 3, 2, 1, 4, 5), dtype=np.int64)
_ROTATIONS = {
    "identity": 0, "rot90": 1, "rot180": 2, "rot270": 3,
    "flip": 0, "flip_rot90": 1, "flip_rot180": 2, "flip_rot270": 3,
}


def action_permutation(name: str) -> np.ndarray:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown D4 transform: {name}")
    reflected = name.startswith("flip")
    rotations = _ROTATIONS[name]
    permutation = np.arange(6, dtype=np.int64)
    if reflected:
        permutation = _FLIP[permutation]
    for _ in range(rotations):
        permutation = _ROT90[permutation]
    return permutation


def transform_board(board: np.ndarray, name: str) -> np.ndarray:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown D4 transform: {name}")
    result = np.asarray(board)
    if name.startswith("flip"):
        result = np.flip(result, axis=-2)
    rotations = _ROTATIONS[name]
    return np.ascontiguousarray(np.rot90(result, rotations, axes=(-2, -1)))


def transform_actions(values: np.ndarray, name: str) -> np.ndarray:
    """Permute the final action axis from the original into transformed frame."""
    values = np.asarray(values)
    if values.shape[-1] != 6:
        raise ValueError("action values must have a final dimension of six")
    result = np.empty_like(values)
    result[..., action_permutation(name)] = values
    return result


def transform_transition(board, action, legal_mask, teacher_q, name):
    permutation = action_permutation(name)
    return (
        transform_board(board, name), int(permutation[int(action)]),
        transform_actions(legal_mask, name), transform_actions(teacher_q, name),
    )

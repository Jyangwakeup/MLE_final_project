"""Replay buffers with exact compact encodings for board observations."""

from __future__ import annotations

from collections import deque
import random
from typing import Any, NamedTuple

import numpy as np
import torch


BINARY_CHANNELS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
TIMER_CHANNEL = 6
CHECKPOINT_FORMAT = "replay-columnar-v1"


class Transition(NamedTuple):
    state: Any
    action: int
    reward: float
    next_state: Any | None
    done: bool
    next_legal: np.ndarray | None


def encode_board(board: np.ndarray) -> dict[str, Any]:
    board = np.asarray(board, dtype=np.float32)
    if board.ndim != 3 or board.shape[0] != 12:
        raise ValueError(f"board must have shape (12,W,H); got {board.shape}")
    binary = board[list(BINARY_CHANNELS)]
    if not np.all((binary == 0.0) | (binary == 1.0)):
        raise ValueError("board binary channels contain non-binary values")
    timer_levels = np.rint(board[TIMER_CHANNEL] * 4.0).astype(np.uint8)
    if not np.allclose(timer_levels.astype(np.float32) / 4.0, board[TIMER_CHANNEL]):
        raise ValueError("bomb timer channel is not representable in quarter steps")
    packed = np.packbits(binary.astype(np.uint8), axis=0)
    return {"packed": packed, "timer": timer_levels, "shape": tuple(board.shape)}


def decode_board(encoded: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in encoded["shape"])
    packed = np.asarray(encoded["packed"], dtype=np.uint8)
    binary = np.unpackbits(packed, axis=0, count=len(BINARY_CHANNELS)).astype(np.float32)
    result = np.zeros(shape, dtype=np.float32)
    result[list(BINARY_CHANNELS)] = binary
    result[TIMER_CHANNEL] = np.asarray(encoded["timer"], dtype=np.float32) / 4.0
    return result


def _encode_state(state: Any, kind: str) -> Any:
    if kind == "vector":
        return np.asarray(state, dtype=np.float32).copy()
    if kind == "board":
        return encode_board(state)
    if kind == "hybrid":
        board, vector = state
        return (encode_board(board), np.asarray(vector, dtype=np.float32).copy())
    raise ValueError(f"unknown replay state kind: {kind!r}")


def _decode_state(state: Any, kind: str) -> Any:
    if kind == "vector":
        return np.asarray(state, dtype=np.float32)
    if kind == "board":
        return decode_board(state)
    if kind == "hybrid":
        return decode_board(state[0]), np.asarray(state[1], dtype=np.float32)
    raise ValueError(f"unknown replay state kind: {kind!r}")


def _stack_encoded(states: list[Any], kind: str) -> Any:
    """Stack encoded states into a few contiguous arrays for checkpoints."""
    if kind == "vector":
        return np.stack(states).astype(np.float32, copy=False)
    if kind == "board":
        return {
            "packed": np.stack([state["packed"] for state in states]),
            "timer": np.stack([state["timer"] for state in states]),
            "shape": tuple(states[0]["shape"]),
        }
    if kind == "hybrid":
        return (
            _stack_encoded([state[0] for state in states], "board"),
            np.stack([state[1] for state in states]).astype(np.float32, copy=False),
        )
    raise ValueError(f"unknown replay state kind: {kind!r}")


def _unstack_encoded(states: Any, index: int, kind: str) -> Any:
    if kind == "vector":
        return np.asarray(states[index], dtype=np.float32).copy()
    if kind == "board":
        return {
            "packed": np.asarray(states["packed"][index], dtype=np.uint8).copy(),
            "timer": np.asarray(states["timer"][index], dtype=np.uint8).copy(),
            "shape": tuple(int(value) for value in states["shape"]),
        }
    if kind == "hybrid":
        return (
            _unstack_encoded(states[0], index, "board"),
            np.asarray(states[1][index], dtype=np.float32).copy(),
        )
    raise ValueError(f"unknown replay state kind: {kind!r}")


def _zero_encoded(state: Any, kind: str) -> Any:
    if kind == "vector":
        return np.zeros_like(state)
    if kind == "board":
        return {
            "packed": np.zeros_like(state["packed"]),
            "timer": np.zeros_like(state["timer"]),
            "shape": tuple(state["shape"]),
        }
    if kind == "hybrid":
        return _zero_encoded(state[0], "board"), np.zeros_like(state[1])
    raise ValueError(f"unknown replay state kind: {kind!r}")


def _decode_board_batch(encoded: list[dict[str, Any]]) -> np.ndarray:
    shape = tuple(int(value) for value in encoded[0]["shape"])
    packed = np.stack([state["packed"] for state in encoded])
    binary = np.unpackbits(
        packed, axis=1, count=len(BINARY_CHANNELS)).astype(np.float32)
    result = np.zeros((len(encoded), *shape), dtype=np.float32)
    result[:, list(BINARY_CHANNELS)] = binary
    result[:, TIMER_CHANNEL] = np.stack(
        [state["timer"] for state in encoded]).astype(np.float32) / 4.0
    return result


def _decode_batch(states: list[Any], kind: str) -> Any:
    if kind == "vector":
        return np.stack(states).astype(np.float32, copy=False)
    if kind == "board":
        return _decode_board_batch(states)
    if kind == "hybrid":
        return (
            _decode_board_batch([state[0] for state in states]),
            np.stack([state[1] for state in states]).astype(np.float32, copy=False),
        )
    raise ValueError(f"unknown replay state kind: {kind!r}")


def _torchify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if isinstance(value, tuple):
        return tuple(_torchify(item) for item in value)
    if isinstance(value, dict):
        return {key: _torchify(item) for key, item in value.items()}
    return value


def _numpyify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, tuple):
        return tuple(_numpyify(item) for item in value)
    if isinstance(value, list):
        return [_numpyify(item) for item in value]
    if isinstance(value, dict):
        return {key: _numpyify(item) for key, item in value.items()}
    return value


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, state_kind: str):
        self.capacity = int(capacity)
        self.state_kind = state_kind
        self.memory = deque(maxlen=self.capacity)
        self.random = random.Random(seed)

    def append(self, transition: Transition) -> None:
        self.memory.append(Transition(
            _encode_state(transition.state, self.state_kind),
            int(transition.action), float(transition.reward),
            None if transition.next_state is None else _encode_state(
                transition.next_state, self.state_kind),
            bool(transition.done),
            None if transition.next_legal is None else np.asarray(
                transition.next_legal, dtype=bool).copy(),
        ))

    def sample(self, size: int) -> list[Transition]:
        return [Transition(
            _decode_state(item.state, self.state_kind), item.action, item.reward,
            None if item.next_state is None else _decode_state(
                item.next_state, self.state_kind),
            item.done, item.next_legal,
        ) for item in self.random.sample(self.memory, size)]

    def sample_batch(self, size: int) -> dict[str, Any]:
        """Sample once and decode directly into contiguous training batches."""
        items = self.random.sample(self.memory, size)
        nonterminal = [item for item in items if not item.done]
        return {
            "states": _decode_batch([item.state for item in items], self.state_kind),
            "actions": np.fromiter(
                (item.action for item in items), dtype=np.int64, count=size),
            "rewards": np.fromiter(
                (item.reward for item in items), dtype=np.float32, count=size),
            "dones": np.fromiter(
                (item.done for item in items), dtype=bool, count=size),
            "nonterminal_indices": np.fromiter(
                (index for index, item in enumerate(items) if not item.done),
                dtype=np.int64, count=len(nonterminal)),
            "next_states": (
                None if not nonterminal else _decode_batch(
                    [item.next_state for item in nonterminal], self.state_kind)
            ),
            "next_legal": (
                None if not nonterminal else np.stack(
                    [item.next_legal for item in nonterminal])
            ),
        }

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> dict[str, Any]:
        items = list(self.memory)
        if not items:
            return {
                "format": CHECKPOINT_FORMAT,
                "capacity": self.capacity,
                "state_kind": self.state_kind,
                "rng_state": self.random.getstate(),
                "count": 0,
            }
        # Terminal transitions have no next observation.  A zero placeholder
        # keeps the checkpoint columnar; ``dones`` controls whether it is read.
        next_states = [
            _zero_encoded(item.state, self.state_kind)
            if item.next_state is None else item.next_state
            for item in items
        ]
        return {
            "format": CHECKPOINT_FORMAT,
            "capacity": self.capacity,
            "state_kind": self.state_kind,
            "rng_state": self.random.getstate(),
            "count": len(items),
            "states": _torchify(_stack_encoded(
                [item.state for item in items], self.state_kind)),
            "next_states": _torchify(_stack_encoded(next_states, self.state_kind)),
            "actions": torch.as_tensor(
                [item.action for item in items], dtype=torch.int64),
            "rewards": torch.as_tensor(
                [item.reward for item in items], dtype=torch.float64),
            "dones": torch.as_tensor(
                [item.done for item in items], dtype=torch.bool),
            "next_legal": torch.as_tensor(np.stack([
                np.zeros(6, dtype=bool) if item.next_legal is None else item.next_legal
                for item in items
            ]), dtype=torch.bool),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity or state["state_kind"] != self.state_kind:
            raise ValueError("replay buffer contract is incompatible")
        self.memory.clear()
        if state.get("format") == CHECKPOINT_FORMAT:
            count = int(state.get("count", 0))
            if count:
                states = _numpyify(state["states"])
                next_states = _numpyify(state["next_states"])
                actions = _numpyify(state["actions"])
                rewards = _numpyify(state["rewards"])
                dones = _numpyify(state["dones"])
                next_legal = _numpyify(state["next_legal"])
                for index in range(count):
                    done = bool(dones[index])
                    self.memory.append(Transition(
                        _unstack_encoded(states, index, self.state_kind),
                        int(actions[index]), float(rewards[index]),
                        None if done else _unstack_encoded(
                            next_states, index, self.state_kind),
                        done,
                        None if done else np.asarray(
                            next_legal[index], dtype=bool).copy(),
                    ))
        else:
            # Read checkpoints produced before the columnar format was added.
            for item in state.get("transitions", ()):
                self.memory.append(Transition(
                    _numpyify(item["state"]), int(item["action"]),
                    float(item["reward"]), _numpyify(item["next_state"]),
                    bool(item["done"]), _numpyify(item["next_legal"]),
                ))
        self.random.setstate(state["rng_state"])

"""Private compact replay and Double-DQN learner for the path CNN agent."""

from __future__ import annotations

from collections import deque, namedtuple
import os
import random

import numpy as np
import torch
from torch import nn


PathTransition = namedtuple(
    "PathTransition", "state action reward next_state done next_legal steps")
PathTransition.__new__.__defaults__ = (1,)
EncodedTransition = namedtuple(
    "EncodedTransition", "state action reward next_state done next_legal steps")
BINARY_CHANNELS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15)
LEVEL_CHANNELS = (6, 12, 13, 14, 16)


def _encode(board):
    board = np.asarray(board, dtype=np.float32)
    if board.shape[0] != 17:
        raise ValueError(f"path board requires 17 channels; got {board.shape}")
    binary = board[list(BINARY_CHANNELS)]
    if not np.all((binary == 0.0) | (binary == 1.0)):
        raise ValueError("path board binary channels are not binary")
    levels = np.empty((len(LEVEL_CHANNELS), *board.shape[1:]), dtype=np.uint8)
    levels[0] = np.rint(np.clip(board[6], 0.0, 1.0) * 4.0)
    levels[1:4] = np.rint(np.clip(board[12:15], 0.0, 1.0) * 32.0)
    levels[4] = np.rint(np.clip(board[16], 0.0, 1.0) * 16.0)
    return (np.packbits(binary.astype(np.uint8), axis=0), levels, tuple(board.shape))


def _decode_batch(items):
    packed = np.stack([item[0] for item in items])
    levels = np.stack([item[1] for item in items])
    shape = items[0][2]
    board = np.zeros((len(items), *shape), dtype=np.float32)
    board[:, list(BINARY_CHANNELS)] = np.unpackbits(
        packed, axis=1, count=len(BINARY_CHANNELS)).astype(np.float32)
    board[:, 6] = levels[:, 0].astype(np.float32) / 4.0
    board[:, 12:15] = levels[:, 1:4].astype(np.float32) / 32.0
    board[:, 16] = levels[:, 4].astype(np.float32) / 16.0
    return board


class PathReplay:
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.memory = deque(maxlen=self.capacity)
        self.random = random.Random(seed)

    def append(self, transition):
        self.memory.append(EncodedTransition(
            _encode(transition.state), int(transition.action), float(transition.reward),
            None if transition.next_state is None else _encode(transition.next_state),
            bool(transition.done),
            None if transition.next_legal is None else np.asarray(
                transition.next_legal, dtype=bool).copy(),
            int(getattr(transition, "steps", 1)),
        ))

    def sample_batch(self, size):
        items = self.random.sample(self.memory, size)
        nonterminal_indices = [index for index, item in enumerate(items) if not item.done]
        nonterminal = [items[index] for index in nonterminal_indices]
        return {
            "states": _decode_batch([item.state for item in items]),
            "actions": np.asarray([item.action for item in items], dtype=np.int64),
            "rewards": np.asarray([item.reward for item in items], dtype=np.float32),
            "steps": np.asarray([item.steps for item in items], dtype=np.int64),
            "nonterminal_indices": np.asarray(nonterminal_indices, dtype=np.int64),
            "next_states": None if not nonterminal else _decode_batch(
                [item.next_state for item in nonterminal]),
            "next_legal": None if not nonterminal else np.stack(
                [item.next_legal for item in nonterminal]),
        }

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        def serialized_state(value):
            if value is None:
                return None
            packed, levels, shape = value
            return {"packed": torch.from_numpy(packed),
                    "levels": torch.from_numpy(levels), "shape": list(shape)}

        return {
            "capacity": self.capacity, "rng_state": self.random.getstate(),
            "transitions": [{
                "state": serialized_state(item.state), "action": item.action,
                "reward": item.reward, "next_state": serialized_state(item.next_state),
                "done": item.done,
                "next_legal": None if item.next_legal is None else torch.from_numpy(item.next_legal),
                "steps": item.steps,
            } for item in self.memory],
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("path replay capacity is incompatible")
        def decoded_state(value):
            if value is None:
                return None
            return (
                np.asarray(value["packed"], dtype=np.uint8),
                np.asarray(value["levels"], dtype=np.uint8),
                tuple(int(item) for item in value["shape"]),
            )

        self.memory = deque((
            EncodedTransition(
                decoded_state(item["state"]), int(item["action"]),
                float(item["reward"]), decoded_state(item["next_state"]),
                bool(item["done"]), None if item["next_legal"] is None else np.asarray(
                    item["next_legal"], dtype=bool),
                int(item.get("steps", 1)),
            ) for item in state["transitions"]), maxlen=self.capacity)
        self.random.setstate(state["rng_state"])


class PathDoubleDQNLearner:
    def __init__(self, policy, target, *, hyperparameters, seed):
        torch.manual_seed(seed)
        requested = os.getenv("BOMBERMAN_TORCH_DEVICE", "cpu")
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for path CNN but is unavailable")
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        else:
            torch.cuda.manual_seed_all(seed)
        self.policy, self.target = policy.to(self.device), target.to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.hyperparameters = dict(hyperparameters)
        self.gamma = float(hyperparameters["gamma"])
        self.batch_size = int(hyperparameters["batch_size"])
        self.warmup = int(hyperparameters["warmup"])
        self.target_sync_interval = int(hyperparameters["target_sync_interval"])
        self.gradient_clip = float(hyperparameters["gradient_clip"])
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=float(hyperparameters["learning_rate"]))
        self.loss_function = nn.SmoothL1Loss()
        self.replay = PathReplay(int(hyperparameters["replay_capacity"]), seed)
        self.updates = 0

    def q_values(self, state):
        with torch.no_grad():
            inputs = torch.as_tensor(state[None], dtype=torch.float32, device=self.device)
            return self.policy(inputs).squeeze(0).cpu().numpy()

    def observe(self, transition):
        self.replay.append(transition)
        if len(self.replay) < max(self.batch_size, self.warmup):
            return None
        batch = self.replay.sample_batch(self.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        steps = torch.as_tensor(batch["steps"], dtype=torch.float32, device=self.device)
        current = self.policy(states).gather(1, actions[:, None]).squeeze(1)
        next_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        if len(batch["nonterminal_indices"]):
            next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32,
                                          device=self.device)
            legal = torch.as_tensor(batch["next_legal"], dtype=torch.bool,
                                    device=self.device)
            with torch.no_grad():
                online = self.policy(next_states).masked_fill(~legal, -torch.inf)
                selected = online.argmax(dim=1, keepdim=True)
                evaluated = self.target(next_states).gather(1, selected).squeeze(1)
            next_values[torch.as_tensor(batch["nonterminal_indices"], device=self.device)] = evaluated
        loss = self.loss_function(current, rewards + self.gamma ** steps * next_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def checkpoint(self):
        return {
            "policy": self.policy.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "updates": self.updates,
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None),
            "training_device_type": self.device.type,
            "training_device_name": (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None),
            "teacher": None,
        }

    def load_checkpoint(self, checkpoint, *, training):
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint["target"])
        self.updates = int(checkpoint.get("updates", 0))
        if training:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for values in self.optimizer.state.values():
                for name, value in values.items():
                    if torch.is_tensor(value):
                        values[name] = value.to(self.device)
            self.replay.load_state_dict(checkpoint["replay"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            cuda_rng = checkpoint.get("torch_cuda_rng_state_all")
            if self.device.type == "cuda" and cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

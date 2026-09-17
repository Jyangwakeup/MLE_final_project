"""Private compact-replay Double DQN with optional teacher-KL retention."""

from __future__ import annotations

from collections import deque, namedtuple
import hashlib
import os
import random

import numpy as np
import torch
from torch import nn

from .distillation import masked_kl


Transition = namedtuple(
    "Transition", "state action reward next_state done next_legal steps", defaults=(1,))
EncodedTransition = namedtuple(
    "EncodedTransition", "state action reward next_state done next_legal steps")
BINARY = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15)
LEVELS = (6, 12, 13, 14, 16)


def _encode(board):
    board = np.asarray(board, dtype=np.float32)
    binary = board[list(BINARY)]
    if board.shape[0] != 17 or not np.all((binary == 0) | (binary == 1)):
        raise ValueError("invalid 17-channel board")
    levels = np.empty((5, *board.shape[1:]), dtype=np.uint8)
    levels[0] = np.rint(np.clip(board[6], 0, 1) * 4)
    levels[1:4] = np.rint(np.clip(board[12:15], 0, 1) * 32)
    levels[4] = np.rint(np.clip(board[16], 0, 1) * 16)
    return np.packbits(binary.astype(np.uint8), axis=0), levels, tuple(board.shape)


def _decode(items):
    packed = np.stack([item[0] for item in items])
    levels = np.stack([item[1] for item in items])
    result = np.zeros((len(items), *items[0][2]), dtype=np.float32)
    result[:, list(BINARY)] = np.unpackbits(
        packed, axis=1, count=len(BINARY)).astype(np.float32)
    result[:, 6] = levels[:, 0] / 4.0
    result[:, 12:15] = levels[:, 1:4] / 32.0
    result[:, 16] = levels[:, 4] / 16.0
    return result


class Replay:
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.memory = deque(maxlen=self.capacity)
        self.random = random.Random(seed)

    def append(self, item):
        self.memory.append(EncodedTransition(
            _encode(item.state), int(item.action), float(item.reward),
            None if item.next_state is None else _encode(item.next_state),
            bool(item.done), None if item.next_legal is None else np.asarray(
                item.next_legal, dtype=bool).copy(), int(item.steps)))

    def sample_batch(self, size):
        items = self.random.sample(self.memory, size)
        indices = [i for i, item in enumerate(items) if not item.done]
        return {
            "states": _decode([item.state for item in items]),
            "actions": np.asarray([item.action for item in items]),
            "rewards": np.asarray([item.reward for item in items], dtype=np.float32),
            "steps": np.asarray([item.steps for item in items], dtype=np.float32),
            "nonterminal_indices": np.asarray(indices),
            "next_states": None if not indices else _decode(
                [items[i].next_state for i in indices]),
            "next_legal": None if not indices else np.stack(
                [items[i].next_legal for i in indices]),
        }

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        def state(value):
            if value is None:
                return None
            packed, levels, shape = value
            return {"packed": torch.from_numpy(packed),
                    "levels": torch.from_numpy(levels), "shape": list(shape)}
        return {
            "capacity": self.capacity, "rng_state": self.random.getstate(),
            "transitions": [{
                "state": state(item.state), "action": item.action,
                "reward": item.reward, "next_state": state(item.next_state),
                "done": item.done,
                "next_legal": None if item.next_legal is None else torch.from_numpy(
                    item.next_legal),
                "steps": item.steps,
            } for item in self.memory],
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity is incompatible")
        def encoded(value):
            if value is None:
                return None
            return (np.asarray(value["packed"], dtype=np.uint8),
                    np.asarray(value["levels"], dtype=np.uint8),
                    tuple(int(item) for item in value["shape"]))
        self.memory = deque((EncodedTransition(
            encoded(item["state"]), int(item["action"]), float(item["reward"]),
            encoded(item["next_state"]), bool(item["done"]),
            None if item["next_legal"] is None else np.asarray(
                item["next_legal"], dtype=bool), int(item.get("steps", 1)),
        ) for item in state["transitions"]), maxlen=self.capacity)
        self.random.setstate(state["rng_state"])


class TeacherDataset:
    def __init__(self, path, seed):
        path = os.path.abspath(os.path.expanduser(path))
        with open(path, "rb") as file:
            self.sha256 = hashlib.sha256(file.read()).hexdigest()
        payload = np.load(path, allow_pickle=False)
        self.boards = np.asarray(payload["boards"], dtype=np.float32)
        self.legal = np.asarray(payload["legal_masks"], dtype=bool)
        self.teacher_q = np.asarray(payload["teacher_q"], dtype=np.float32)
        seeds = np.asarray(payload["environment_seeds"], dtype=np.int64)
        training = (seeds >= 6000) & (seeds <= 6079)
        if len(seeds) != len(self.boards) or not training.any():
            raise ValueError("teacher dataset needs training seeds 6000-6079")
        if self.boards.ndim != 4 or self.boards.shape[1] != 17:
            raise ValueError("teacher dataset has incompatible boards")
        if self.legal.shape != self.teacher_q.shape or self.legal.shape[1] != 6:
            raise ValueError("teacher dataset has incompatible action values")
        if len(self.boards) != len(self.legal) or not self.legal.any(axis=1).all():
            raise ValueError("teacher dataset rows are inconsistent")
        self.boards = self.boards[training]
        self.legal = self.legal[training]
        self.teacher_q = self.teacher_q[training]
        self.path = path
        self.random = np.random.default_rng(seed)

    def sample(self, size):
        indices = self.random.integers(0, len(self.boards), size=size)
        return self.boards[indices], self.legal[indices], self.teacher_q[indices]


class DistilledDoubleDQNLearner:
    def __init__(self, policy, target, *, hyperparameters, seed, teacher_dataset=None):
        requested = os.getenv("BOMBERMAN_TORCH_DEVICE", "cpu")
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.policy, self.target = policy.to(self.device), target.to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.hyperparameters = dict(hyperparameters)
        self.gamma = float(hyperparameters["gamma"])
        self.batch_size = int(hyperparameters["batch_size"])
        self.warmup = int(hyperparameters["warmup"])
        self.target_sync_interval = int(hyperparameters["target_sync_interval"])
        self.gradient_clip = float(hyperparameters["gradient_clip"])
        self.kl_weight = float(hyperparameters.get("teacher_kl_weight", 1.0))
        self.temperature = float(hyperparameters.get("temperature", 1.0))
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(hyperparameters["learning_rate"]))
        self.replay = Replay(int(hyperparameters["replay_capacity"]), seed)
        self.teacher_dataset = teacher_dataset
        self.updates = 0

    def q_values(self, state):
        with torch.no_grad():
            value = self.policy(torch.as_tensor(
                state[None], dtype=torch.float32, device=self.device))
        return value[0].cpu().numpy()

    def _teacher_loss(self):
        if self.teacher_dataset is None or self.kl_weight == 0:
            return torch.zeros((), device=self.device)
        boards, legal, teacher_q = self.teacher_dataset.sample(self.batch_size)
        boards = torch.as_tensor(boards, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal, dtype=torch.bool, device=self.device)
        teacher_q = torch.as_tensor(teacher_q, dtype=torch.float32, device=self.device)
        return masked_kl(
            self.policy(boards), teacher_q, legal, temperature=self.temperature)

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
        next_values = torch.zeros(self.batch_size, device=self.device)
        if len(batch["nonterminal_indices"]):
            next_states = torch.as_tensor(
                batch["next_states"], dtype=torch.float32, device=self.device)
            legal = torch.as_tensor(batch["next_legal"], dtype=torch.bool, device=self.device)
            with torch.no_grad():
                selected = self.policy(next_states).masked_fill(~legal, -torch.inf).argmax(
                    1, keepdim=True)
                evaluated = self.target(next_states).gather(1, selected).squeeze(1)
            next_values[torch.as_tensor(
                batch["nonterminal_indices"], device=self.device)] = evaluated
        td_loss = nn.functional.smooth_l1_loss(
            current, rewards + self.gamma ** steps * next_values)
        kl_loss = self._teacher_loss()
        loss = td_loss + self.kl_weight * kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return {"loss": float(loss.item()), "td_loss": float(td_loss.item()),
                "teacher_kl": float(kl_loss.item())}

    def checkpoint(self):
        return {
            "policy": self.policy.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "updates": self.updates,
            "teacher_dataset": None if self.teacher_dataset is None else {
                "sha256": self.teacher_dataset.sha256,
            },
        }

    def load_checkpoint(self, checkpoint, *, training, warm_start=False):
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(
            checkpoint["policy"] if warm_start else checkpoint["target"])
        self.updates = 0 if warm_start else int(checkpoint.get("updates", 0))
        if training and not warm_start:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.replay.load_state_dict(checkpoint["replay"])
            torch.set_rng_state(checkpoint["torch_rng_state"])

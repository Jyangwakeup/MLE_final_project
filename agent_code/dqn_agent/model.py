from collections import deque, namedtuple
import random

import numpy as np
import torch
from torch import nn


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done", "next_legal"))


class QNetwork(nn.Module):
    def __init__(self, input_size: int, action_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.memory = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def append(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, size: int):
        return self.random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        return {
            "transitions": [
                {
                    "state": torch.as_tensor(item.state, dtype=torch.float32),
                    "action": int(item.action),
                    "reward": float(item.reward),
                    "next_state": (
                        None if item.next_state is None
                        else torch.as_tensor(item.next_state, dtype=torch.float32)
                    ),
                    "done": bool(item.done),
                    "next_legal": (
                        None if item.next_legal is None
                        else torch.as_tensor(item.next_legal, dtype=torch.bool)
                    ),
                }
                for item in self.memory
            ],
            "rng_state": self.random.getstate(),
        }

    def load_state_dict(self, state):
        self.memory.clear()
        self.memory.extend(
            Transition(
                item["state"].cpu().numpy(),
                item["action"],
                item["reward"],
                None if item["next_state"] is None else item["next_state"].cpu().numpy(),
                item["done"],
                None if item["next_legal"] is None else item["next_legal"].cpu().numpy(),
            )
            for item in state.get("transitions", ())
        )
        if "rng_state" in state:
            self.random.setstate(state["rng_state"])


class DQN:
    def __init__(self, input_size, action_count, seed=0, gamma=0.95, learning_rate=3e-4,
                 batch_size=64, replay_capacity=20_000, warmup=2_000,
                 target_sync_interval=1_000):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.action_count = action_count
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_sync_interval = target_sync_interval
        self.updates = 0

        self.policy = QNetwork(input_size, action_count).to(self.device)
        self.target = QNetwork(input_size, action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_capacity, seed)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(tensor).squeeze(0).cpu().numpy()

    def observe(self, transition: Transition):
        self.replay.append(transition)
        if len(self.replay) < max(self.batch_size, self.warmup):
            return None
        return self._learn()

    def _learn(self):
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.bool)

        current_q = self.policy(states).gather(1, actions).squeeze(1)
        next_values = torch.zeros(self.batch_size, dtype=torch.float32)
        nonterminal = ~dones
        if nonterminal.any():
            next_states = torch.as_tensor(
                np.stack([t.next_state for t in batch if not t.done]), dtype=torch.float32
            )
            legal = torch.as_tensor(
                np.stack([t.next_legal for t in batch if not t.done]), dtype=torch.bool
            )
            with torch.no_grad():
                next_q = self.target(next_states).masked_fill(~legal, -torch.inf)
                next_values[nonterminal] = next_q.max(dim=1).values

        targets = rewards + self.gamma * next_values
        loss = self.loss_function(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def checkpoint(self):
        return {
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "updates": self.updates,
        }

    def load_checkpoint(self, checkpoint, training=False):
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["policy"]))
        self.updates = checkpoint.get("updates", 0)
        if training and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if training and "replay" in checkpoint:
            self.replay.load_state_dict(checkpoint["replay"])
        if training and "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])

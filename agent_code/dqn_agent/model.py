from collections import deque, namedtuple
import os
import random

import numpy as np

# Required by CUDA >= 10.2 when deterministic algorithms are enabled. It must
# be present before the first cuBLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done", "next_legal"))
REPLAY_CHECKPOINT_FORMAT = "vector-replay-columnar-v1"


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

    def sample_batch(self, size: int):
        items = self.random.sample(self.memory, size)
        nonterminal = [item for item in items if not item.done]
        return {
            "states": np.stack([item.state for item in items]),
            "actions": np.fromiter(
                (item.action for item in items), dtype=np.int64, count=size),
            "rewards": np.fromiter(
                (item.reward for item in items), dtype=np.float32, count=size),
            "dones": np.fromiter(
                (item.done for item in items), dtype=bool, count=size),
            "next_states": None if not nonterminal else np.stack([
                item.next_state for item in nonterminal]),
            "next_legal": None if not nonterminal else np.stack([
                item.next_legal for item in nonterminal]),
        }

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        items = list(self.memory)
        if not items:
            return {
                "format": REPLAY_CHECKPOINT_FORMAT,
                "count": 0,
                "rng_state": self.random.getstate(),
            }
        states = np.stack([item.state for item in items]).astype(np.float32, copy=False)
        next_states = np.stack([
            np.zeros_like(item.state) if item.next_state is None else item.next_state
            for item in items
        ]).astype(np.float32, copy=False)
        return {
            "format": REPLAY_CHECKPOINT_FORMAT,
            "count": len(items),
            "states": torch.as_tensor(states, dtype=torch.float32),
            "next_states": torch.as_tensor(next_states, dtype=torch.float32),
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
            "rng_state": self.random.getstate(),
        }

    def load_state_dict(self, state):
        self.memory.clear()
        if state.get("format") == REPLAY_CHECKPOINT_FORMAT:
            count = int(state.get("count", 0))
            if count:
                states = state["states"].cpu().numpy()
                next_states = state["next_states"].cpu().numpy()
                actions = state["actions"].cpu().numpy()
                rewards = state["rewards"].cpu().numpy()
                dones = state["dones"].cpu().numpy()
                next_legal = state["next_legal"].cpu().numpy()
                self.memory.extend(
                    Transition(
                        states[index].copy(), int(actions[index]),
                        float(rewards[index]),
                        None if bool(dones[index]) else next_states[index].copy(),
                        bool(dones[index]),
                        None if bool(dones[index]) else next_legal[index].copy(),
                    )
                    for index in range(count)
                )
        else:
            # Backward-compatible loader for existing baseline checkpoints.
            self.memory.extend(
                Transition(
                    item["state"].cpu().numpy(), item["action"], item["reward"],
                    None if item["next_state"] is None
                    else item["next_state"].cpu().numpy(),
                    item["done"],
                    None if item["next_legal"] is None
                    else item["next_legal"].cpu().numpy(),
                )
                for item in state.get("transitions", ())
            )
        if "rng_state" in state:
            self.random.setstate(state["rng_state"])


class DQN:
    def __init__(self, input_size, action_count, seed=0, gamma=0.95, learning_rate=3e-4,
                 batch_size=64, replay_capacity=20_000, warmup=2_000,
                 target_sync_interval=1_000, device="cpu", deterministic=True):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic
        torch.set_num_threads(1)
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
        batch = self.replay.sample_batch(self.batch_size)
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(
            batch["dones"], dtype=torch.bool, device=self.device)

        current_q = self.policy(states).gather(1, actions).squeeze(1)
        next_values = torch.zeros(
            self.batch_size, dtype=torch.float32, device=self.device)
        nonterminal = ~dones
        if nonterminal.any():
            next_states = torch.as_tensor(
                batch["next_states"], dtype=torch.float32, device=self.device)
            legal = torch.as_tensor(
                batch["next_legal"], dtype=torch.bool, device=self.device)
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
        checkpoint = {
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "training_device_name": (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
            ),
            "training_device_type": self.device.type,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" else 0
            ),
            "updates": self.updates,
        }
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return checkpoint

    def load_checkpoint(self, checkpoint, training=False):
        if training and checkpoint.get("training_device_type") not in (None, self.device.type):
            raise ValueError("Cannot resume DQN training on a different training device")
        if (
            training and self.device.type == "cuda"
            and checkpoint.get("training_device_name") not in (
                None, torch.cuda.get_device_name(self.device)
            )
        ):
            raise ValueError("Cannot resume DQN training on a different GPU model")
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["policy"]))
        self.updates = checkpoint.get("updates", 0)
        if training and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        if training and "replay" in checkpoint:
            self.replay.load_state_dict(checkpoint["replay"])
        if training and "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if training and self.device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

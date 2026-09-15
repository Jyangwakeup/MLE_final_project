from collections import deque, namedtuple
import os
import random

import numpy as np

# Required by CUDA >= 10.2 when deterministic algorithms are enabled. It must
# be present before the first cuBLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


Transition = namedtuple(
    "Transition",
    (
        "state", "action", "reward", "next_state", "done", "next_legal",
        "state_legal", "task_id", "steps", "safety_class",
    ),
    defaults=(None, None, 1, "ordinary"),
)
REPLAY_CHECKPOINT_FORMAT = "vector-replay-task-partitioned-v3"
DEFAULT_GAMMA = 0.95
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_REPLAY_CAPACITY = 20_000
DEFAULT_WARMUP = 2_000
DEFAULT_TARGET_SYNC_INTERVAL = 1_000
DEFAULT_HIDDEN_SIZE = 64


class QNetwork(nn.Module):
    def __init__(self, input_size: int, action_count: int, hidden_size: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.partitions = {}
        self.current_task = None
        self.random = random.Random(seed)

    def configure(self, current_task: str | None, per_task_capacity: int | None = None):
        self.current_task = current_task or "unassigned"
        if per_task_capacity is not None:
            self.capacity = int(per_task_capacity)

    def append(self, transition: Transition):
        task_id = transition.task_id or self.current_task or "unassigned"
        if task_id not in self.partitions:
            self.partitions[task_id] = deque(maxlen=self.capacity)
        self.partitions[task_id].append(transition._replace(task_id=task_id))

    def sample(self, size: int):
        return self.random.sample(self._all_items(), size)

    def sample_batch(self, size: int, parent_fraction: float = 0.0,
                     safety_replay_spec=None):
        current = list(self.partitions.get(self.current_task, ()))
        parents = [
            item for task, partition in self.partitions.items()
            if task != self.current_task for item in partition
        ]
        safety = dict(safety_replay_spec or {})
        if safety.get("enabled") and parents and current:
            parent_count = int(safety["parent_samples"])
            safety_count = int(safety["own_bomb_cycle_samples"])
            ordinary_count = int(safety["ordinary_task3_samples"])
            if parent_count + safety_count + ordinary_count != size:
                raise ValueError("safety replay counts do not match batch size")
            safety_pool = [
                item for item in current if item.safety_class != "ordinary"]
            fatal_pool = [
                item for item in safety_pool
                if item.safety_class == "fatal_own_bomb"]
            selected_safety = self.random.sample(
                fatal_pool, min(len(fatal_pool), safety_count))
            remaining = safety_count - len(selected_safety)
            selected_ids = {id(item) for item in selected_safety}
            other_safety = [item for item in safety_pool if id(item) not in selected_ids]
            selected_safety += self.random.sample(
                other_safety, min(len(other_safety), remaining))
            remaining = safety_count - len(selected_safety)
            selected_ids = {id(item) for item in selected_safety}
            ordinary_pool = [item for item in current if id(item) not in selected_ids]
            if remaining:
                selected_safety += self.random.sample(ordinary_pool, remaining)
                selected_ids = {id(item) for item in selected_safety}
                ordinary_pool = [
                    item for item in ordinary_pool if id(item) not in selected_ids]
            items = (
                self.random.sample(parents, parent_count)
                + selected_safety
                + self.random.sample(ordinary_pool, ordinary_count)
            )
            self.random.shuffle(items)
        elif parents and current and parent_fraction > 0.0:
            parent_count = min(len(parents), int(round(size * parent_fraction)))
            current_count = size - parent_count
            if len(current) < current_count:
                current_count = len(current)
                parent_count = size - current_count
            if len(parents) < parent_count:
                parent_count = len(parents)
                current_count = size - parent_count
            items = self.random.sample(parents, parent_count) + self.random.sample(
                current, current_count)
            self.random.shuffle(items)
        else:
            items = self.random.sample(self._all_items(), size)
        nonterminal = [item for item in items if not item.done]
        return {
            "states": np.stack([item.state for item in items]),
            "actions": np.fromiter(
                (item.action for item in items), dtype=np.int64, count=size),
            "rewards": np.fromiter(
                (item.reward for item in items), dtype=np.float32, count=size),
            "dones": np.fromiter(
                (item.done for item in items), dtype=bool, count=size),
            "steps": np.fromiter(
                (item.steps for item in items), dtype=np.int64, count=size),
            "state_legal": np.stack([
                np.ones(6, dtype=bool) if item.state_legal is None else item.state_legal
                for item in items
            ]),
            "is_parent": np.fromiter(
                (item.task_id != self.current_task for item in items),
                dtype=bool, count=size),
            "safety_classes": [item.safety_class for item in items],
            "next_states": None if not nonterminal else np.stack([
                item.next_state for item in nonterminal]),
            "next_legal": None if not nonterminal else np.stack([
                item.next_legal for item in nonterminal]),
        }

    def __len__(self):
        return sum(len(partition) for partition in self.partitions.values())

    def current_size(self):
        return len(self.partitions.get(self.current_task, ()))

    def _all_items(self):
        return [item for partition in self.partitions.values() for item in partition]

    def state_dict(self):
        items = self._all_items()
        if not items:
            return {
                "format": REPLAY_CHECKPOINT_FORMAT,
                "count": 0,
                "capacity": self.capacity,
                "current_task": self.current_task,
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
            "capacity": self.capacity,
            "current_task": self.current_task,
            "states": torch.as_tensor(states, dtype=torch.float32),
            "next_states": torch.as_tensor(next_states, dtype=torch.float32),
            "actions": torch.as_tensor(
                [item.action for item in items], dtype=torch.int64),
            "rewards": torch.as_tensor(
                [item.reward for item in items], dtype=torch.float64),
            "dones": torch.as_tensor(
                [item.done for item in items], dtype=torch.bool),
            "steps": torch.as_tensor(
                [item.steps for item in items], dtype=torch.int64),
            "state_legal": torch.as_tensor(np.stack([
                np.ones(6, dtype=bool) if item.state_legal is None else item.state_legal
                for item in items
            ]), dtype=torch.bool),
            "task_ids": [item.task_id for item in items],
            "safety_classes": [item.safety_class for item in items],
            "next_legal": torch.as_tensor(np.stack([
                np.zeros(6, dtype=bool) if item.next_legal is None else item.next_legal
                for item in items
            ]), dtype=torch.bool),
            "rng_state": self.random.getstate(),
        }

    def load_state_dict(self, state):
        self.partitions.clear()
        if state.get("format") in {
            REPLAY_CHECKPOINT_FORMAT, "vector-replay-task-partitioned-v2",
        }:
            self.capacity = int(state.get("capacity", self.capacity))
            self.current_task = state.get("current_task", self.current_task)
            count = int(state.get("count", 0))
            if count:
                states = state["states"].cpu().numpy()
                next_states = state["next_states"].cpu().numpy()
                actions = state["actions"].cpu().numpy()
                rewards = state["rewards"].cpu().numpy()
                dones = state["dones"].cpu().numpy()
                steps = state["steps"].cpu().numpy()
                state_legal = state["state_legal"].cpu().numpy()
                next_legal = state["next_legal"].cpu().numpy()
                task_ids = list(state["task_ids"])
                safety_classes = list(state.get(
                    "safety_classes", ["ordinary"] * count))
                for index in range(count):
                    transition = Transition(
                        states[index].copy(), int(actions[index]),
                        float(rewards[index]),
                        None if bool(dones[index]) else next_states[index].copy(),
                        bool(dones[index]),
                        None if bool(dones[index]) else next_legal[index].copy(),
                        state_legal[index].copy(), task_ids[index], int(steps[index]),
                        safety_classes[index],
                    )
                    self.append(transition)
        else:
            # Backward-compatible loader for existing baseline checkpoints.
            for item in state.get("transitions", ()):
                self.append(Transition(
                    item["state"].cpu().numpy(), item["action"], item["reward"],
                    None if item["next_state"] is None
                    else item["next_state"].cpu().numpy(),
                    item["done"],
                    None if item["next_legal"] is None
                    else item["next_legal"].cpu().numpy(),
                ))
        if "rng_state" in state:
            self.random.setstate(state["rng_state"])

    def mark_recent_own_bomb_fatal(self, task_id: str, count: int = 4) -> None:
        """Label the fatal transition prefix without duplicating replay rows."""
        partition = self.partitions.get(task_id)
        if not partition:
            return
        marked = 0
        for index in range(len(partition) - 1, -1, -1):
            item = partition[index]
            if item.safety_class in {"own_bomb", "fatal_own_bomb"}:
                partition[index] = item._replace(safety_class="fatal_own_bomb")
                marked += 1
                if marked >= count:
                    break


class DQN:
    def __init__(self, input_size, action_count, seed=0, gamma=DEFAULT_GAMMA,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 batch_size=DEFAULT_BATCH_SIZE,
                 replay_capacity=DEFAULT_REPLAY_CAPACITY,
                 warmup=DEFAULT_WARMUP,
                 target_sync_interval=DEFAULT_TARGET_SYNC_INTERVAL,
                 device="cpu", deterministic=True,
                 training_task=None, retention_spec=None, double_dqn=False,
                 hidden_size=DEFAULT_HIDDEN_SIZE, safety_replay_spec=None):
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.warmup = warmup
        self.target_sync_interval = target_sync_interval
        self.training_task = training_task
        self.retention_spec = dict(retention_spec or {
            "parent_fraction": 0.0, "distillation_weight": 0.0,
            "temperature": 1.0, "per_task_capacity": replay_capacity,
            "current_warmup": warmup,
        })
        self.double_dqn = bool(double_dqn)
        self.safety_replay_spec = dict(safety_replay_spec or {"enabled": False})
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.updates = 0

        self.policy = QNetwork(input_size, action_count, hidden_size).to(self.device)
        self.target = QNetwork(input_size, action_count, hidden_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(
            int(self.retention_spec["per_task_capacity"]), seed)
        self.replay.configure(training_task)
        self.teacher = None

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(tensor).squeeze(0).cpu().numpy()

    def observe(self, transition: Transition):
        self.replay.append(transition)
        required_current = max(
            self.batch_size, int(self.retention_spec["current_warmup"]))
        if self.replay.current_size() < required_current:
            return None
        return self._learn()

    def _learn(self):
        parent_fraction = (
            float(self.retention_spec["parent_fraction"])
            if self.teacher is not None else 0.0
        )
        batch = self.replay.sample_batch(
            self.batch_size, parent_fraction, self.safety_replay_spec)
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(
            batch["dones"], dtype=torch.bool, device=self.device)
        steps = torch.as_tensor(
            batch["steps"], dtype=torch.float32, device=self.device)

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
                target_q = self.target(next_states).masked_fill(~legal, -torch.inf)
                if self.double_dqn:
                    policy_q = self.policy(next_states).masked_fill(~legal, -torch.inf)
                    selected = policy_q.argmax(dim=1, keepdim=True)
                    next_values[nonterminal] = target_q.gather(
                        1, selected).squeeze(1)
                else:
                    next_values[nonterminal] = target_q.max(dim=1).values

        targets = rewards + torch.pow(
            torch.full_like(steps, self.gamma), steps) * next_values
        loss = self.loss_function(current_q, targets)
        parent_rows = torch.as_tensor(
            batch["is_parent"], dtype=torch.bool, device=self.device)
        if self.teacher is not None and parent_rows.any():
            temperature = float(self.retention_spec["temperature"])
            legal = torch.as_tensor(
                batch["state_legal"], dtype=torch.bool, device=self.device)[parent_rows]
            if not bool(legal.any(dim=1).all()):
                raise ValueError("A replay state has no physically legal action")
            student_q = self.policy(states[parent_rows])
            mask_value = torch.finfo(student_q.dtype).min
            student_q = student_q.masked_fill(~legal, mask_value)
            with torch.no_grad():
                teacher_q = self.teacher(states[parent_rows]).masked_fill(
                    ~legal, mask_value)
                teacher_probabilities = torch.softmax(teacher_q / temperature, dim=1)
            distillation = nn.functional.kl_div(
                torch.log_softmax(student_q / temperature, dim=1),
                teacher_probabilities,
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = loss + float(
                self.retention_spec["distillation_weight"]) * distillation
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
            "teacher": None if self.teacher is None else self.teacher.state_dict(),
            "retention_spec": self.retention_spec,
            "double_dqn": self.double_dqn,
            "safety_replay_spec": self.safety_replay_spec,
        }
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return checkpoint

    def load_checkpoint(self, checkpoint, training=False, training_task=None):
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
            parent_task = checkpoint.get("training_task")
            self.training_task = training_task or self.training_task
            self.replay.configure(
                self.training_task, int(self.retention_spec["per_task_capacity"]))
            if parent_task != self.training_task:
                self.teacher = QNetwork(
                    self.input_size, self.action_count, self.hidden_size).to(self.device)
                self.teacher.load_state_dict(checkpoint["policy"])
            elif checkpoint.get("teacher") is not None:
                self.teacher = QNetwork(
                    self.input_size, self.action_count, self.hidden_size).to(self.device)
                self.teacher.load_state_dict(checkpoint["teacher"])
            if self.teacher is not None:
                self.teacher.eval()
                for parameter in self.teacher.parameters():
                    parameter.requires_grad_(False)
        if training and "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if training and self.device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

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
        "state_legal", "task_id", "steps", "safety_class", "observed_kill",
    ),
    defaults=(None, None, 1, "ordinary", False),
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
    def __init__(
        self, input_size: int, action_count: int, hidden_size: int = 64,
        exact_prefix_size: int | None = None,
    ):
        super().__init__()
        self.exact_prefix_size = exact_prefix_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, x):
        if self.exact_prefix_size is not None:
            first = self.layers[0]
            prefix = nn.functional.linear(
                x[..., :self.exact_prefix_size],
                first.weight[:, :self.exact_prefix_size], first.bias)
            suffix = nn.functional.linear(
                x[..., self.exact_prefix_size:],
                first.weight[:, self.exact_prefix_size:], None)
            x = prefix + suffix
            for layer in self.layers[1:]:
                x = layer(x)
            return x
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
                     safety_replay_spec=None, task_samples=None, sampling_version=None):
        current = list(self.partitions.get(self.current_task, ()))
        parents = [
            item for task, partition in self.partitions.items()
            if task != self.current_task for item in partition
        ]
        safety = dict(safety_replay_spec or {})
        if sampling_version is not None:
            if (sampling_version != 'task4-only-v1' or self.current_task != 'full_match'
                    or set(self.partitions) - {'full_match'} or parent_fraction != 0
                    or safety.get('enabled') or task_samples is not None):
                raise ValueError('Foreign task or incompatible Task4-only sampling contract')
            items = self.random.sample(current, size)
        elif task_samples is not None:
            if safety.get('enabled') or sum(task_samples.values()) != size:
                raise ValueError('Task quotas conflict with batch size or safety replay')
            items = []
            for task, count in task_samples.items():
                pool = list(self.partitions.get(task, ()))
                if len(pool) < count:
                    raise ValueError('Replay does not yet satisfy fixed task quotas')
                items.extend(self.random.sample(pool, count))
            self.random.shuffle(items)
        elif safety.get("enabled") and parents and current:
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
        return self.pack_batch(items, size)

    def pack_batch(self, items, size):
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
                        safety_classes[index], bool(state.get("observed_kills", [False] * count)[index]),
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
                 hidden_size=DEFAULT_HIDDEN_SIZE, safety_replay_spec=None, exact_prefix_size=None):
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
        self.exact_prefix_size = exact_prefix_size
        self.updates = 0

        self.policy = QNetwork(
            input_size, action_count, hidden_size, exact_prefix_size).to(self.device)
        self.target = QNetwork(
            input_size, action_count, hidden_size, exact_prefix_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(
            int(self.retention_spec["per_task_capacity"]), seed)
        if self.retention_spec.get("sampling_version") == "task4-kill-replay-v1":
            from agent_code.learning_common.kill_replay import KillReplayBuffer
            self.replay = KillReplayBuffer(20000, seed)
        self.replay.configure(training_task)
        self.teacher = None
        self.distillation_dataset = None
        self.distillation_random = random.Random(seed + 104729)
        self.distillation_dataset_hash = None

    def load_distillation_dataset(self, dataset, dataset_hash=None):
        """Install immutable teacher facts sampled independently of TD replay."""
        if dataset is None:
            self.distillation_dataset = None
            self.distillation_dataset_hash = None
            return
        required = {"states", "legal_masks", "teacher_q"}
        if not required.issubset(dataset):
            raise ValueError("distillation dataset is missing required arrays")
        states = np.asarray(dataset["states"], dtype=np.float32)
        legal = np.asarray(dataset["legal_masks"], dtype=bool)
        teacher_q = np.asarray(dataset["teacher_q"], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.input_size:
            raise ValueError("distillation states have an incompatible shape")
        if legal.shape != (len(states), self.action_count):
            raise ValueError("distillation legal masks have an incompatible shape")
        if teacher_q.shape != legal.shape or not len(states):
            raise ValueError("distillation teacher Q values have an incompatible shape")
        if not legal.any(axis=1).all() or not np.isfinite(teacher_q).all():
            raise ValueError("distillation dataset contains invalid rows")
        self.distillation_dataset = {
            "states": states.copy(), "legal_masks": legal.copy(),
            "teacher_q": teacher_q.copy(),
        }
        self.distillation_dataset_hash = dataset_hash

    def transfer_input_prefix(self, checkpoint, prefix_size: int):
        """Expand a policy while preserving its exact function on a zero suffix."""
        if not 0 < int(prefix_size) < self.input_size:
            raise ValueError("transfer prefix size must be smaller than the new input")
        for destination_network, source_key in (
            (self.policy, "policy"), (self.target, "target"),
        ):
            source = checkpoint[source_key]
            destination = destination_network.state_dict()
            first = "layers.0.weight"
            source_weight = source[first]
            if source_weight.shape[1] != prefix_size:
                raise ValueError("parent first layer has an incompatible input size")
            if source_weight.shape[0] != destination[first].shape[0]:
                raise ValueError("parent hidden size is incompatible")
            destination[first].zero_()
            destination[first][:, :prefix_size].copy_(source_weight)
            for name, value in source.items():
                if name == first:
                    continue
                if name not in destination or destination[name].shape != value.shape:
                    raise ValueError("parent network architecture is incompatible")
                destination[name].copy_(value)
            destination_network.load_state_dict(destination)
        self.updates = 0

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(tensor).squeeze(0).cpu().numpy()

    def observe(self, transition: Transition):
        if self.retention_spec.get('sampling_version') in ('task4-only-v1', 'task4-kill-replay-v1'):
            if transition.task_id != 'full_match' or self.teacher is not None or self.distillation_dataset is not None:
                raise ValueError('Task4-only observation contract violated')
        self.replay.append(transition)
        required_current = max(
            self.batch_size, int(self.retention_spec["current_warmup"]))
        if self.replay.current_size() < required_current:
            return None
        return self._learn()

    def _learn(self):
        task_samples = self.retention_spec.get('task_samples')
        if task_samples and any(len(self.replay.partitions.get(task, ())) < count
                                for task, count in task_samples.items()):
            return None
        parent_fraction = (
            float(self.retention_spec["parent_fraction"])
            if self.teacher is not None else 0.0
        )
        batch = self.replay.sample_batch(
            self.batch_size, parent_fraction, self.safety_replay_spec, task_samples,
            self.retention_spec.get("sampling_version"))
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
        if self.distillation_dataset is not None:
            sample_size = min(self.batch_size, len(self.distillation_dataset["states"]))
            indices = self.distillation_random.sample(
                range(len(self.distillation_dataset["states"])), sample_size)
            teacher_states = torch.as_tensor(
                self.distillation_dataset["states"][indices],
                dtype=torch.float32, device=self.device)
            teacher_legal = torch.as_tensor(
                self.distillation_dataset["legal_masks"][indices],
                dtype=torch.bool, device=self.device)
            teacher_q = torch.as_tensor(
                self.distillation_dataset["teacher_q"][indices],
                dtype=torch.float32, device=self.device)
            temperature = float(self.retention_spec["temperature"])
            minimum = torch.finfo(teacher_q.dtype).min
            student_logits = self.policy(teacher_states).masked_fill(
                ~teacher_legal, minimum)
            teacher_logits = teacher_q.masked_fill(~teacher_legal, minimum)
            teacher_probabilities = torch.softmax(
                teacher_logits / temperature, dim=1)
            distillation = nn.functional.kl_div(
                torch.log_softmax(student_logits / temperature, dim=1),
                teacher_probabilities, reduction="batchmean",
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
            "distillation_dataset": (
                None if self.distillation_dataset is None else {
                    "states": torch.as_tensor(
                        self.distillation_dataset["states"], dtype=torch.float32),
                    "legal_masks": torch.as_tensor(
                        self.distillation_dataset["legal_masks"], dtype=torch.bool),
                    "teacher_q": torch.as_tensor(
                        self.distillation_dataset["teacher_q"], dtype=torch.float32),
                }
            ),
            "distillation_dataset_hash": self.distillation_dataset_hash,
            "distillation_rng_state": self.distillation_random.getstate(),
        }
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return checkpoint

    def load_checkpoint(self, checkpoint, training=False, training_task=None):
        if self.retention_spec.get('sampling_version') in ('task4-only-v1', 'task4-kill-replay-v1'):
            if (checkpoint.get('training_task') != 'full_match'
                    or set(checkpoint['replay'].get('task_ids', [])) - {'full_match'}
                    or checkpoint.get('teacher') is not None
                    or checkpoint.get('distillation_dataset') is not None
                    or checkpoint.get('retention_spec') != self.retention_spec):
                raise ValueError('Task4-only checkpoint contains foreign learning state')
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
        if training and checkpoint.get("distillation_dataset") is not None:
            self.load_distillation_dataset(
                checkpoint["distillation_dataset"],
                checkpoint.get("distillation_dataset_hash"))
            if "distillation_rng_state" in checkpoint:
                self.distillation_random.setstate(checkpoint["distillation_rng_state"])
        if training and "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if training and self.device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    def load_policy_weights(self, checkpoint):
        """Warm-start from policy weights while retaining fresh training state."""
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint["policy"])

"""Dueling Double DQN with proportional prioritized replay."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
import copy
from agent_code.dqn_agent.model import Transition

class DuelingNetwork(nn.Module):
    def __init__(self, input_size, action_count, hidden_size=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_count)
    def forward(self, state):
        hidden = self.trunk(state)
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)

class PrioritizedReplay:
    def __init__(self, capacity, seed, alpha, *, protected_capacity=0,
                 protected_reward_threshold=None):
        self.capacity, self.alpha = int(capacity), float(alpha)
        self.protected_capacity = int(protected_capacity)
        self.protected_reward_threshold = (
            None if protected_reward_threshold is None
            else float(protected_reward_threshold))
        if not 0 <= self.protected_capacity <= self.capacity:
            raise ValueError("protected_capacity must be between zero and capacity")
        self.random = random.Random(seed)
        self.memory, self.priorities, self.protected, self.position = [], [], [], 0

    def _is_protected(self, transition):
        return (
            self.protected_capacity > 0
            and self.protected_reward_threshold is not None
            and float(transition.reward) >= self.protected_reward_threshold
        )

    def _replacement_index(self, protected):
        desired = bool(protected)
        protected_count = sum(self.protected)
        replace_protected = desired and protected_count >= self.protected_capacity
        for offset in range(self.capacity):
            index = (self.position + offset) % self.capacity
            if bool(self.protected[index]) == replace_protected:
                self.position = (index + 1) % self.capacity
                return index
        index = self.position
        self.position = (self.position + 1) % self.capacity
        return index

    def append(self, transition):
        priority = max(self.priorities, default=1.0)
        protected = self._is_protected(transition)
        if len(self.memory) < self.capacity:
            self.memory.append(transition); self.priorities.append(priority); self.protected.append(protected)
        else:
            index = self._replacement_index(protected)
            self.memory[index] = transition; self.priorities[index] = priority; self.protected[index] = protected
        if len(self.memory) < self.capacity:
            self.position = len(self.memory) % self.capacity
    def sample(self, size, beta):
        probabilities = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        probabilities /= probabilities.sum()
        cumulative = np.cumsum(probabilities)
        indices = np.asarray([min(int(np.searchsorted(cumulative, self.random.random(), side="right")), len(self.memory)-1) for _ in range(size)])
        weights = (len(self.memory) * probabilities[indices]) ** (-float(beta)); weights /= weights.max()
        return [self.memory[i] for i in indices], indices, weights.astype(np.float32)
    def update_priorities(self, indices, errors, epsilon):
        for index, error in zip(indices, errors): self.priorities[int(index)] = abs(float(error)) + epsilon
    def __len__(self): return len(self.memory)
    def current_size(self): return len(self.memory)
    def state_dict(self):
        transitions = [{"state":torch.as_tensor(x.state), "action":x.action, "reward":x.reward,
                        "next_state":None if x.next_state is None else torch.as_tensor(x.next_state),
                        "done":x.done, "next_legal":None if x.next_legal is None else torch.as_tensor(x.next_legal),
                        "state_legal":torch.as_tensor(x.state_legal), "task_id":x.task_id, "steps":x.steps}
                       for x in self.memory]
        return {"capacity": self.capacity, "alpha": self.alpha, "rng_state": self.random.getstate(), "transitions": transitions, "priorities": self.priorities, "position": self.position, "protected": self.protected, "protected_capacity": self.protected_capacity, "protected_reward_threshold": self.protected_reward_threshold}
    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity or float(state["alpha"]) != self.alpha: raise ValueError("prioritized replay contract changed")
        self.memory = [Transition(x["state"].cpu().numpy(),x["action"],x["reward"],None if x["next_state"] is None else x["next_state"].cpu().numpy(),x["done"],None if x["next_legal"] is None else x["next_legal"].cpu().numpy(),x["state_legal"].cpu().numpy(),x["task_id"],x["steps"]) for x in state["transitions"]]
        self.priorities, self.position = list(state["priorities"]), int(state["position"])
        self.protected = list(state.get("protected", [False] * len(self.memory)))
        self.random.setstate(state["rng_state"])

class RainbowLite:
    def __init__(self, input_size, action_count, *, seed, hyperparameters, training_task=None, retention_spec=None, device="cpu"):
        self.device = torch.device(device); torch.manual_seed(seed); torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
        self.gamma, self.batch_size, self.warmup = hyperparameters["gamma"], hyperparameters["batch_size"], hyperparameters["warmup"]
        self.target_sync_interval = hyperparameters["target_sync_interval"]
        self.beta_start, self.beta_steps, self.priority_epsilon = hyperparameters["per_beta_start"], hyperparameters["per_beta_steps"], hyperparameters["per_epsilon"]
        self.training_task, self.retention_spec = training_task, dict(retention_spec or {})
        self.policy = DuelingNetwork(input_size, action_count).to(self.device); self.target = DuelingNetwork(input_size, action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict()); self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperparameters["learning_rate"])
        self.replay = PrioritizedReplay(
            hyperparameters["replay_capacity"], seed, hyperparameters["per_alpha"],
            protected_capacity=hyperparameters.get("protected_replay_capacity", 0),
            protected_reward_threshold=hyperparameters.get("protected_reward_threshold"))
        self.updates = 0
        self.observations = 0
        self.teacher = None
        self.train_interval = int(hyperparameters.get("train_interval", 1))
        if self.train_interval < 1:
            raise ValueError("train_interval must be positive")
    def q_values(self, state):
        with torch.no_grad(): return self.policy(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
    def install_teacher_from_policy(self):
        """Freeze the current policy as an immutable anti-forgetting anchor."""
        self.teacher = copy.deepcopy(self.policy).to(self.device)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
    def observe(self, transition):
        self.replay.append(transition)
        self.observations += 1
        if self.observations % self.train_interval:
            return None
        if len(self.replay) < max(self.batch_size, self.warmup): return None
        beta = min(1., self.beta_start + (1.-self.beta_start)*self.updates/self.beta_steps)
        batch, indices, weights = self.replay.sample(self.batch_size, beta)
        states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in batch], device=self.device).unsqueeze(1)
        targets = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        current = self.policy(states).gather(1, actions).squeeze(1)
        rows = [i for i,x in enumerate(batch) if not x.done]
        if rows:
            nxt = torch.as_tensor(np.stack([batch[i].next_state for i in rows]), dtype=torch.float32, device=self.device)
            legal = torch.as_tensor(np.stack([batch[i].next_legal for i in rows]), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                selected = self.policy(nxt).masked_fill(~legal, -torch.inf).argmax(1, keepdim=True)
                future = self.target(nxt).gather(1, selected).squeeze(1)
                powers = torch.as_tensor([self.gamma ** batch[i].steps for i in rows], dtype=torch.float32, device=self.device)
                targets[torch.as_tensor(rows, device=self.device)] += powers * future
        errors = targets-current
        loss = (torch.as_tensor(weights, device=self.device)*nn.functional.smooth_l1_loss(current, targets, reduction="none")).mean()
        distillation_weight = float(self.retention_spec.get("distillation_weight", 0.0))
        if self.teacher is not None and distillation_weight > 0.0:
            legal = torch.as_tensor(
                np.stack([x.state_legal for x in batch]),
                dtype=torch.bool, device=self.device)
            if not bool(legal.any(dim=1).all()):
                raise ValueError("A replay state has no physically legal action")
            temperature = float(self.retention_spec.get("temperature", 1.0))
            minimum = torch.finfo(states.dtype).min
            student_logits = self.policy(states).masked_fill(~legal, minimum)
            with torch.no_grad():
                teacher_logits = self.teacher(states).masked_fill(~legal, minimum)
                teacher_probabilities = torch.softmax(
                    teacher_logits / temperature, dim=1)
            distillation = nn.functional.kl_div(
                torch.log_softmax(student_logits / temperature, dim=1),
                teacher_probabilities, reduction="batchmean") * (temperature ** 2)
            loss = loss + distillation_weight * distillation
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), 10.); self.optimizer.step()
        self.replay.update_priorities(indices, errors.detach().cpu().numpy(), self.priority_epsilon)
        self.updates += 1
        if self.updates % self.target_sync_interval == 0: self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())
    def checkpoint(self):
        return {"policy":self.policy.state_dict(), "target":self.target.state_dict(), "optimizer":self.optimizer.state_dict(), "replay":self.replay.state_dict(), "torch_rng_state":torch.get_rng_state(), "updates":self.updates, "observations":self.observations, "training_device_type":self.device.type, "training_device_name":None, "teacher":None if self.teacher is None else self.teacher.state_dict(), "retention_spec":self.retention_spec}
    def load_checkpoint(self, checkpoint, training=False, training_task=None):
        self.policy.load_state_dict(checkpoint["policy"]); self.target.load_state_dict(checkpoint.get("target",checkpoint["policy"])); self.updates=int(checkpoint.get("updates",0)); self.observations=int(checkpoint.get("observations", self.updates * self.train_interval))
        if training:
            self.optimizer.load_state_dict(checkpoint["optimizer"]); self.replay.load_state_dict(checkpoint["replay"]); torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("teacher") is not None:
                self.teacher = DuelingNetwork(
                    self.policy.trunk[0].in_features,
                    self.policy.advantage.out_features).to(self.device)
                self.teacher.load_state_dict(checkpoint["teacher"])
                self.teacher.eval()
                for parameter in self.teacher.parameters():
                    parameter.requires_grad_(False)

__all__ = ["DuelingNetwork", "PrioritizedReplay", "RainbowLite", "Transition"]

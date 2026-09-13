"""A reusable CPU Double-DQN learner for vector, board, and hybrid inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .replay import ReplayBuffer, Transition


def double_dqn_next_values(policy_q: torch.Tensor, target_q: torch.Tensor,
                           legal: torch.Tensor) -> torch.Tensor:
    """Select with policy Q-values and evaluate with target Q-values."""
    if policy_q.shape != target_q.shape or policy_q.shape != legal.shape:
        raise ValueError("Double DQN tensors must have identical shapes")
    if not legal.any(dim=1).all():
        raise ValueError("nonterminal replay transition has no legal next action")
    actions = policy_q.masked_fill(~legal, -torch.inf).argmax(dim=1, keepdim=True)
    return target_q.gather(1, actions).squeeze(1)


class DoubleDQNLearner:
    def __init__(self, policy: nn.Module, target: nn.Module, *, state_kind: str,
                 hyperparameters: dict[str, Any], seed: int):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.policy = policy.to(self.device)
        self.target = target.to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.state_kind = state_kind
        self.hyperparameters = dict(hyperparameters)
        self.gamma = float(hyperparameters["gamma"])
        self.batch_size = int(hyperparameters["batch_size"])
        self.warmup = int(hyperparameters["warmup"])
        self.target_sync_interval = int(hyperparameters["target_sync_interval"])
        self.gradient_clip = float(hyperparameters["gradient_clip"])
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(hyperparameters["learning_rate"]))
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(
            int(hyperparameters["replay_capacity"]), seed, state_kind)
        self.updates = 0

    def _inputs(self, states: list[Any]):
        if self.state_kind in {"vector", "board"}:
            return torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        boards, vectors = zip(*states)
        return (
            torch.as_tensor(np.stack(boards), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(vectors), dtype=torch.float32, device=self.device),
        )

    @staticmethod
    def _forward(network: nn.Module, inputs):
        return network(*inputs) if isinstance(inputs, tuple) else network(inputs)

    def q_values(self, state: Any) -> np.ndarray:
        inputs = self._inputs([state])
        with torch.no_grad():
            return self._forward(self.policy, inputs).squeeze(0).cpu().numpy()

    def observe(self, transition: Transition) -> float | None:
        self.replay.append(transition)
        if len(self.replay) < max(self.batch_size, self.warmup):
            return None
        return self._learn()

    def _learn(self) -> float:
        batch = self.replay.sample_batch(self.batch_size)
        states = self._tensor_inputs(batch["states"])
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(
            batch["dones"], dtype=torch.bool, device=self.device)
        current = self._forward(self.policy, states).gather(1, actions).squeeze(1)
        next_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        nonterminal_indices = batch["nonterminal_indices"]
        if nonterminal_indices.size:
            next_states = self._tensor_inputs(batch["next_states"])
            legal = torch.as_tensor(
                batch["next_legal"], dtype=torch.bool, device=self.device)
            with torch.no_grad():
                policy_q = self._forward(self.policy, next_states)
                target_q = self._forward(self.target, next_states)
                evaluated = double_dqn_next_values(policy_q, target_q, legal)
                next_values[torch.as_tensor(nonterminal_indices, device=self.device)] = evaluated
        targets = rewards + self.gamma * next_values
        loss = self.loss_function(current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def _tensor_inputs(self, states):
        if isinstance(states, tuple):
            return tuple(torch.as_tensor(
                value, dtype=torch.float32, device=self.device) for value in states)
        return torch.as_tensor(states, dtype=torch.float32, device=self.device)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "updates": self.updates,
            "training_device_name": None,
            "training_device_type": self.device.type,
        }

    def load_checkpoint(self, checkpoint: dict[str, Any], *, training: bool) -> None:
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint["target"])
        self.updates = int(checkpoint.get("updates", 0))
        if training:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.replay.load_state_dict(checkpoint["replay"])
            torch.set_rng_state(checkpoint["torch_rng_state"])


__all__ = ["DoubleDQNLearner", "Transition", "double_dqn_next_values"]

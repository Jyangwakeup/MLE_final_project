"""Spatial CNN heads for the distilled Double-DQN agent."""

from __future__ import annotations

import torch
from torch import nn

from .symmetry import TRANSFORMS, _ROTATIONS, action_permutation


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs):
        return self.activation(inputs + self.layers(inputs))


class SpatialEncoder(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(17, channels, 3, padding=1), nn.ReLU(),
            *(ResidualBlock(channels, dilation) for dilation in (1, 2, 4, 8)),
        )

    def forward(self, board):
        return self.layers(board)


def _self_local(encoded, board):
    mask = board[:, 3:4]
    return (encoded * mask).sum(dim=(2, 3))


class GlobalQNetwork(nn.Module):
    """Control candidate matching the old global six-output head."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.encoder = SpatialEncoder(channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 128), nn.ReLU(), nn.Linear(128, 6),
        )

    def forward(self, board):
        encoded = self.encoder(board)
        context = torch.cat((
            _self_local(encoded, board), encoded.mean((2, 3)), encoded.amax((2, 3))),
            dim=1,
        )
        return self.head(context)


class ActionAlignedQNetwork(nn.Module):
    """Score four neighboring cells with one shared head, plus WAIT/BOMB."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.encoder = SpatialEncoder(channels)
        context_width = channels * 3
        self.value = nn.Sequential(
            nn.Linear(context_width, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.move = nn.Sequential(
            nn.Linear(context_width + channels, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.wait = nn.Sequential(
            nn.Linear(context_width, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.bomb = nn.Sequential(
            nn.Linear(context_width, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, board):
        encoded = self.encoder(board)
        local = _self_local(encoded, board)
        context = torch.cat((local, encoded.mean((2, 3)), encoded.amax((2, 3))), dim=1)
        flat_position = board[:, 3].flatten(1).argmax(1)
        width, height = board.shape[-2:]
        x = torch.div(flat_position, height, rounding_mode="floor")
        y = flat_position.remainder(height)
        deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
        destinations = []
        batch = torch.arange(board.shape[0], device=board.device)
        for dx, dy in deltas:
            nx = (x + dx).clamp(0, width - 1)
            ny = (y + dy).clamp(0, height - 1)
            destinations.append(encoded[batch, :, nx, ny])
        neighbor = torch.stack(destinations, dim=1)
        repeated = context[:, None, :].expand(-1, 4, -1)
        moves = self.move(torch.cat((repeated, neighbor), dim=2)).squeeze(2)
        advantages = torch.cat((moves, self.wait(context), self.bomb(context)), dim=1)
        value = self.value(context)
        return value + advantages - advantages.mean(dim=1, keepdim=True)


def _transform_torch(board, name):
    result = board
    if name.startswith("flip"):
        result = torch.flip(result, dims=(-2,))
    rotations = _ROTATIONS[name]
    return torch.rot90(result, rotations, dims=(-2, -1))


class D4AveragedQNetwork(nn.Module):
    """Orbit-average a base network for exact D4 action equivariance."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, board):
        aligned = []
        for name in TRANSFORMS:
            values = self.base(_transform_torch(board, name))
            permutation = torch.as_tensor(
                action_permutation(name), dtype=torch.long, device=values.device)
            aligned.append(values.index_select(1, permutation))
        return torch.stack(aligned).mean(dim=0)


def build_network(architecture: str, *, channels: int = 64):
    if architecture == "global":
        return GlobalQNetwork(channels)
    if architecture == "action_aligned":
        return ActionAlignedQNetwork(channels)
    if architecture == "action_aligned_d4":
        return D4AveragedQNetwork(ActionAlignedQNetwork(channels))
    raise ValueError(f"unknown CNN architecture: {architecture}")

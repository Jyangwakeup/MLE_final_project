from __future__ import annotations

import torch
from torch import nn

from .learner import PathDoubleDQNLearner


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


class PathBoardQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(17, 64, 3, padding=1), nn.ReLU())
        self.encoder = nn.Sequential(*(
            ResidualBlock(64, dilation) for dilation in (1, 2, 4, 8)
        ))
        self.head = nn.Sequential(
            nn.Linear(64 * 3, 128), nn.ReLU(), nn.Linear(128, 6),
        )

    def forward(self, board):
        encoded = self.encoder(self.stem(board))
        self_mask = board[:, 3:4]
        local = (encoded * self_mask).sum(dim=(2, 3))
        average = encoded.mean(dim=(2, 3))
        maximum = encoded.amax(dim=(2, 3))
        return self.head(torch.cat((local, average, maximum), dim=1))


def build_learner(seed, hyperparameters):
    torch.manual_seed(seed)
    return PathDoubleDQNLearner(
        PathBoardQNetwork(), PathBoardQNetwork(),
        hyperparameters=hyperparameters, seed=seed)

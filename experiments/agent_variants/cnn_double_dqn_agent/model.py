import torch
from torch import nn

from agent_code.learning_common.neural import DoubleDQNLearner


class BoardQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 6),
        )

    def forward(self, board):
        return self.spatial(board)


def build_learner(seed, hyperparameters):
    torch.manual_seed(seed)
    return DoubleDQNLearner(
        BoardQNetwork(), BoardQNetwork(), state_kind="board",
        hyperparameters=hyperparameters, seed=seed,
    )

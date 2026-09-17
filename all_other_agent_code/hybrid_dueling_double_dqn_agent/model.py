import torch
from torch import nn

from agent_code.learning_common.neural import DoubleDQNLearner


class HybridDuelingQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_branch = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
        )
        self.vector_branch = nn.Sequential(
            nn.Linear(70, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(192, 128), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6))

    def forward(self, board, vector):
        fused = self.fusion(torch.cat(
            (self.board_branch(board), self.vector_branch(vector)), dim=1))
        value = self.value(fused)
        advantage = self.advantage(fused)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_learner(seed, hyperparameters):
    torch.manual_seed(seed)
    return DoubleDQNLearner(
        HybridDuelingQNetwork(), HybridDuelingQNetwork(), state_kind="hybrid",
        hyperparameters=hyperparameters, seed=seed,
    )

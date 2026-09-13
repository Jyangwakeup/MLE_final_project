import torch
from torch import nn

from agent_code.learning_common.neural import DoubleDQNLearner


class ContinuousQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(70, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, vector):
        return self.layers(vector)


def build_learner(seed, hyperparameters):
    torch.manual_seed(seed)
    return DoubleDQNLearner(
        ContinuousQNetwork(), ContinuousQNetwork(), state_kind="vector",
        hyperparameters=hyperparameters, seed=seed,
    )

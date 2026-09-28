"""Small board branch added to the existing v6 Rainbow-lite learner."""

import numpy as np
import torch
from torch import nn

from agent_code.rainbow_lite_agent.model import RainbowLite, PrioritizedReplay, Transition

VECTOR_DIM = 160
BOARD_SHAPE = (12, 17, 17)
INPUT_DIM = VECTOR_DIM + 12 * 17 * 17
_BINARY_CHANNELS = tuple(i for i in range(12) if i != 6)


def _encode(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (INPUT_DIM,):
        raise ValueError(f"expected {INPUT_DIM} state values")
    board = state[VECTOR_DIM:].reshape(BOARD_SHAPE)
    binary = board[list(_BINARY_CHANNELS)]
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("board binary channels are not binary")
    timer = np.rint(board[6] * 4).astype(np.uint8)
    if not np.allclose(board[6], timer.astype(np.float32) / 4):
        raise ValueError("bomb timer cannot be encoded exactly")
    return (state[:VECTOR_DIM].copy(), np.packbits(binary.astype(np.uint8), axis=0), timer)


def _decode(state):
    vector, bits, timer = state
    board = np.zeros(BOARD_SHAPE, dtype=np.float32)
    board[list(_BINARY_CHANNELS)] = np.unpackbits(
        bits, axis=0, count=len(_BINARY_CHANNELS))
    board[6] = timer.astype(np.float32) / 4
    return np.concatenate((vector, board.reshape(-1)))


class SpatialReplay(PrioritizedReplay):
    """Exact packed board storage with the original prioritized sampling."""

    def append(self, transition):
        super().append(transition._replace(
            state=_encode(transition.state),
            next_state=None if transition.next_state is None else _encode(transition.next_state)))

    def sample(self, size, beta):
        batch, indices, weights = super().sample(size, beta)
        return [item._replace(
            state=_decode(item.state),
            next_state=None if item.next_state is None else _decode(item.next_state))
            for item in batch], indices, weights

    def state_dict(self):
        result = {
            "capacity": self.capacity, "alpha": self.alpha,
            "rng_state": self.random.getstate(), "priorities": self.priorities,
            "position": self.position, "protected": self.protected,
            "protected_capacity": self.protected_capacity,
            "protected_reward_threshold": self.protected_reward_threshold,
            "format": "spatial-board12-packed-v1", "transitions": [],
        }
        for item in self.memory:
            result["transitions"].append({
                "state": tuple(torch.as_tensor(x) for x in item.state),
                "action": item.action, "reward": item.reward,
                "next_state": None if item.next_state is None else tuple(
                    torch.as_tensor(x) for x in item.next_state),
                "done": item.done,
                "next_legal": None if item.next_legal is None else torch.as_tensor(item.next_legal),
                "state_legal": torch.as_tensor(item.state_legal),
                "task_id": item.task_id, "steps": item.steps,
                "safety_class": item.safety_class,
            })
        return result

    def load_state_dict(self, state):
        if (state.get("format") != "spatial-board12-packed-v1"
                or int(state["capacity"]) != self.capacity
                or float(state["alpha"]) != self.alpha):
            raise ValueError("spatial replay contract changed")
        self.memory = [Transition(
            tuple(x.cpu().numpy() for x in item["state"]),
            item["action"], item["reward"],
            None if item["next_state"] is None else tuple(
                x.cpu().numpy() for x in item["next_state"]),
            item["done"],
            None if item["next_legal"] is None else item["next_legal"].cpu().numpy(),
            item["state_legal"].cpu().numpy(), item["task_id"], item["steps"],
            item.get("safety_class", "ordinary"),
        ) for item in state["transitions"]]
        self.priorities = list(state["priorities"])
        self.position = int(state["position"])
        self.protected = list(state["protected"])
        self.random.setstate(state["rng_state"])


class SpatialDuelingNetwork(nn.Module):
    def __init__(self, input_size=INPUT_DIM, action_count=6):
        super().__init__()
        if input_size != INPUT_DIM or action_count != 6:
            raise ValueError("spatial v6 requires 160 vector values and a 12x17x17 board")
        self.trunk = nn.Sequential(
            nn.Linear(VECTOR_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.board_branch = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(192, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_count)
        # An imported v6 policy can start with identical Q values. The board
        # branch is admitted gradually once training changes these zero columns.
        with torch.no_grad():
            self.fusion[0].weight.zero_()
            self.fusion[0].weight[:, :128].copy_(torch.eye(128))
            self.fusion[0].bias.zero_()

    def forward(self, state):
        if state.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected {INPUT_DIM} input values, got {state.shape[-1]}")
        vector = self.trunk(state[:, :VECTOR_DIM])
        board = state[:, VECTOR_DIM:].reshape(-1, *BOARD_SHAPE)
        hidden = self.fusion(torch.cat((vector, self.board_branch(board)), dim=1))
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)


class SpatialRainbowLite(RainbowLite):
    def __init__(self, input_size, action_count, **kwargs):
        super().__init__(input_size, action_count, **kwargs)
        self.policy = SpatialDuelingNetwork(input_size, action_count).to(self.device)
        self.target = SpatialDuelingNetwork(input_size, action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=kwargs["hyperparameters"]["learning_rate"])
        hp = kwargs["hyperparameters"]
        self.replay = SpatialReplay(
            hp["replay_capacity"], kwargs["seed"], hp["per_alpha"],
            protected_capacity=hp.get("protected_replay_capacity", 0),
            protected_reward_threshold=hp.get("protected_reward_threshold"))

    def load_checkpoint(self, checkpoint, training=False, training_task=None):
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["policy"]))
        self.updates = int(checkpoint.get("updates", 0))
        self.observations = int(checkpoint.get("observations", self.updates * self.train_interval))
        if training:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.replay.load_state_dict(checkpoint["replay"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("teacher") is not None:
                self.teacher = SpatialDuelingNetwork().to(self.device)
                self.teacher.load_state_dict(checkpoint["teacher"])
                self.teacher.eval()
                for parameter in self.teacher.parameters():
                    parameter.requires_grad_(False)

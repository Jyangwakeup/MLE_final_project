"""Explicit experiment contracts for project learning agents."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import settings as s
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract


@dataclass(frozen=True)
class AgentContract:
    agent: str
    algorithm: str
    feature_id: str
    checkpoint_name: str
    feature_schema: dict[str, Any]
    network_spec: dict[str, Any] | None
    hyperparameters: dict[str, Any]


_BASELINES = {
    "q_learning_agent": ("q_learning", "discrete-q-v2", "final.pkl", None),
    "dqn_agent": ("dqn", "discrete-q-v2", "final.pt", None),
}
_NEW_AGENTS = {
    "double_q_compact_agent", "double_dqn_continuous_agent",
    "cnn_double_dqn_agent", "hybrid_dueling_double_dqn_agent",
}


def resolve_agent_contract(agent: str) -> AgentContract:
    if agent in _BASELINES:
        algorithm, feature_id, checkpoint, network = _BASELINES[agent]
        board_shape = None
        return AgentContract(
            agent, algorithm, feature_id, checkpoint,
            feature_schema_contract(feature_id, board_shape), network, {},
        )
    if agent not in _NEW_AGENTS:
        raise ValueError(f"experiments do not define a learning contract for {agent!r}")
    module = import_module(f"agent_code.{agent}.callbacks")
    metadata = module.AGENT_METADATA
    feature_id = metadata["feature_id"]
    board_shape = (s.COLS, s.ROWS) if feature_id in {"board-v1", "hybrid-v1"} else None
    contract = AgentContract(
        agent=agent, algorithm=metadata["algorithm"], feature_id=feature_id,
        checkpoint_name=metadata["checkpoint_name"],
        feature_schema=feature_schema_contract(feature_id, board_shape),
        network_spec=metadata["network_spec"],
        hyperparameters=dict(metadata["hyperparameters"]),
    )
    if tuple(contract.feature_schema["action_order"]) != ACTIONS:
        raise ValueError("agent feature schema uses an incompatible action order")
    return contract

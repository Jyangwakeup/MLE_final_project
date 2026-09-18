"""Explicit experiment contracts for project learning agents."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import settings as s
from agent_code.team_agent.feature_system import (
    ACTIONS, feature_schema_contract, normalize_feature_id,
)


@dataclass(frozen=True)
class AgentContract:
    agent: str
    algorithm: str
    feature_id: str
    checkpoint_name: str | None
    feature_schema: dict[str, Any]
    network_spec: dict[str, Any] | None
    hyperparameters: dict[str, Any]
    trainable: bool = True


_BASELINES = {
    "q_learning_agent": ("q_learning", "discrete-q-v2", "final.pkl", None),
    "dqn_agent": ("dqn", "discrete-q-v2", "final.pt", None),
    "legal_random_agent": ("legal_random", "discrete-v1", "baseline.json", None),
}
_OFFICIAL_BASELINES = {
    "random_agent", "rule_based_agent", "peaceful_agent",
    "coin_collector_agent",
}
_NEW_AGENTS = {
    "optimized_double_q_lambda_crate_agent",
    "optimized_double_q_lambda_grouped_agent",
    "optimized_double_q_lambda_history_agent",
    "double_q_compact_agent", "double_dqn_continuous_agent",
    "cnn_double_dqn_agent", "hybrid_dueling_double_dqn_agent",
    "double_q_agent", "double_dqn_continuous_v2_agent",
    "double_dqn_continuous_v3_agent",
    "cnn_path_double_dqn_agent",
    "cnn_distilled_double_dqn_agent",
    "cnn_distillation_teacher_agent",
    "double_dqn_phase_agent",
    "double_dqn_continuous_v4_agent",
    "rainbow_lite_agent", "expected_sarsa_lambda_agent", "double_q_lambda_agent",
    "optimized_double_q_lambda_agent", "optimized_double_q_lambda_v4_agent",
    "rainbow_lite_no_safety_agent", "expected_sarsa_lambda_no_safety_agent",
    "rainbow_lite_v5_agent",
    "expected_sarsa_lambda_v5_agent",
}
_BASELINE_FEATURE_IDS = {
    "discrete-v1", "discrete-q-v2", "discrete-objective-v1",
}


def resolve_agent_contract(
    agent: str, feature_id: str | None = None,
) -> AgentContract:
    if agent in _OFFICIAL_BASELINES:
        resolved_feature_id = normalize_feature_id(feature_id or "discrete-v1")
        if resolved_feature_id != "discrete-v1":
            raise ValueError(
                f"{agent} only supports 'discrete-v1' evaluation metadata; "
                f"got {resolved_feature_id!r}")
        return AgentContract(
            agent=agent,
            algorithm="official_baseline",
            feature_id=resolved_feature_id,
            checkpoint_name=None,
            feature_schema=feature_schema_contract(resolved_feature_id),
            network_spec=None,
            hyperparameters={},
            trainable=False,
        )
    if agent in _BASELINES:
        algorithm, default_feature_id, checkpoint, network = _BASELINES[agent]
        resolved_feature_id = normalize_feature_id(feature_id or default_feature_id)
        allowed = (
            {"discrete-v1"}
            if agent == "legal_random_agent" else _BASELINE_FEATURE_IDS
        )
        if resolved_feature_id not in allowed:
            raise ValueError(
                f"{agent} only supports {sorted(allowed)}; "
                f"got {resolved_feature_id!r}")
        if agent == "q_learning_agent":
            from agent_code.q_learning_agent.callbacks import HYPERPARAMETERS
            hyperparameters = dict(HYPERPARAMETERS)
        elif agent == "dqn_agent":
            from agent_code.dqn_agent.callbacks import (
                HYPERPARAMETERS, network_spec as dqn_network_spec,
            )
            hyperparameters = dict(HYPERPARAMETERS)
            network = dqn_network_spec(
                feature_schema_contract(resolved_feature_id)["vector_shape"][0])
        else:
            hyperparameters = {}
        return AgentContract(
            agent, algorithm, resolved_feature_id, checkpoint,
            feature_schema_contract(resolved_feature_id), network, hyperparameters,
        )
    if agent not in _NEW_AGENTS:
        raise ValueError(f"experiments do not define a learning contract for {agent!r}")
    module = import_module(f"agent_code.{agent}.callbacks")
    metadata = module.AGENT_METADATA
    fixed_feature_id = metadata["feature_id"]
    # Agent-local experimental representations need not be globally registered
    # in team_agent.feature_system.  They publish their complete immutable
    # schema in callback metadata instead.
    requested_feature_id = (
        feature_id or fixed_feature_id
        if metadata.get("feature_schema") is not None
        else normalize_feature_id(feature_id or fixed_feature_id)
    )
    if requested_feature_id != fixed_feature_id:
        raise ValueError(
            f"{agent} requires feature ID {fixed_feature_id!r}; "
            f"got {requested_feature_id!r}")
    board_shape = (
        (s.COLS, s.ROWS)
        if fixed_feature_id in {"board-v1", "hybrid-v1"} else None
    )
    resolved_schema = metadata.get("feature_schema")
    if resolved_schema is None:
        resolved_schema = feature_schema_contract(fixed_feature_id, board_shape)
    contract = AgentContract(
        agent=agent,
        algorithm=metadata["algorithm"],
        feature_id=fixed_feature_id,
        checkpoint_name=metadata["checkpoint_name"],
        feature_schema=resolved_schema,
        network_spec=metadata["network_spec"],
        hyperparameters=dict(metadata["hyperparameters"]),
        trainable=bool(metadata.get("trainable", True)),
    )
    if tuple(contract.feature_schema["action_order"]) != ACTIONS:
        raise ValueError("agent feature schema uses an incompatible action order")
    return contract

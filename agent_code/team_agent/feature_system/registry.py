"""Registry and compatibility aliases for versioned feature extractors."""

from __future__ import annotations

from typing import Callable, Optional

from . import (
    board_v1, continuous_v1, continuous_v2, continuous_v2_legacy78,
    continuous_v3, continuous_v4, continuous_v5, continuous_phase_v1,
    discrete_compact_v1,
    discrete_objective_v1, discrete_q_v2, discrete_v1, hybrid_v1,
)
from .types import FeatureSchema


DEFAULT_FEATURE_ID = "discrete-v1"
LEGACY_FEATURE_IDS = {"v1": DEFAULT_FEATURE_ID}
_MODULES = {
    discrete_v1.FEATURE_ID: discrete_v1,
    discrete_q_v2.FEATURE_ID: discrete_q_v2,
    discrete_objective_v1.FEATURE_ID: discrete_objective_v1,
    discrete_compact_v1.FEATURE_ID: discrete_compact_v1,
    continuous_v1.FEATURE_ID: continuous_v1,
    continuous_v2.FEATURE_ID: continuous_v2,
    continuous_v2_legacy78.FEATURE_ID: continuous_v2_legacy78,
    continuous_v3.FEATURE_ID: continuous_v3,
    continuous_phase_v1.FEATURE_ID: continuous_phase_v1,
    continuous_v4.FEATURE_ID: continuous_v4,
    continuous_v5.FEATURE_ID: continuous_v5,
    board_v1.FEATURE_ID: board_v1,
    hybrid_v1.FEATURE_ID: hybrid_v1,
}


def available_feature_ids() -> tuple[str, ...]:
    return tuple(_MODULES)


def normalize_feature_id(
    feature_id: Optional[str] = None,
    legacy_version: Optional[str] = None,
) -> str:
    """Resolve a semantic ID while rejecting conflicting old/new settings."""
    normalized_id = LEGACY_FEATURE_IDS.get(feature_id, feature_id)
    normalized_legacy = LEGACY_FEATURE_IDS.get(legacy_version, legacy_version)
    if normalized_id is not None and normalized_legacy is not None:
        if normalized_id != normalized_legacy:
            raise ValueError(
                f"feature_id {feature_id!r} conflicts with "
                f"feature_version {legacy_version!r}")
    resolved = normalized_id or normalized_legacy or DEFAULT_FEATURE_ID
    if resolved not in _MODULES:
        raise ValueError(
            f"unknown feature ID {resolved!r}; available: {available_feature_ids()}")
    return resolved


def get_feature_schema(
    feature_id: str,
    board_shape: Optional[tuple[int, int]] = None,
) -> FeatureSchema:
    resolved = normalize_feature_id(feature_id)
    return _MODULES[resolved].schema(board_shape=board_shape)


def get_feature_extractor(feature_id: str) -> Callable:
    resolved = normalize_feature_id(feature_id)
    return _MODULES[resolved].extract


def extract_features(game_state: dict, feature_id: str = DEFAULT_FEATURE_ID):
    return get_feature_extractor(feature_id)(game_state)


def feature_schema_contract(
    feature_id: str,
    board_shape: Optional[tuple[int, int]] = None,
) -> dict:
    return get_feature_schema(feature_id, board_shape=board_shape).to_dict()


def validate_checkpoint_feature_contract(
    payload: dict,
    expected_feature_id: str,
    expected_actions: tuple[str, ...],
    *,
    board_shape: Optional[tuple[int, int]] = None,
) -> None:
    """Validate new contracts strictly while admitting explicit legacy ``v1``."""
    expected = normalize_feature_id(expected_feature_id)
    if "feature_id" not in payload and "feature_version" not in payload:
        raise ValueError("checkpoint has no feature version/ID")
    has_new_contract = "feature_id" in payload
    try:
        actual = normalize_feature_id(
            payload.get("feature_id"), payload.get("feature_version"))
    except ValueError as exception:
        raise ValueError(f"incompatible feature version/ID: {exception}") from exception
    if actual != expected:
        raise ValueError(
            f"incompatible feature version/ID {actual!r}; expected {expected!r}")
    if payload.get("actions") is not None:
        if tuple(payload["actions"]) != tuple(expected_actions):
            raise ValueError("incompatible checkpoint action order")
    elif has_new_contract:
        raise ValueError("new checkpoint feature contract is missing actions")
    if has_new_contract:
        if "feature_schema" not in payload:
            raise ValueError("new checkpoint feature contract is missing feature_schema")
        expected_schema = feature_schema_contract(expected, board_shape=board_shape)
        if payload["feature_schema"] != expected_schema:
            raise ValueError("incompatible checkpoint feature schema or shape")

"""Stable public API for reusable, versioned Bomberman features."""

from .common import (
    ACTIONS,
    HORIZON,
    FeatureContext,
    build_context,
    build_safety_context,
    danger_after_action,
    danger_at_steps,
    earliest_danger_time,
)
from ..temporal_safety_features import ReachabilityResult
from .registry import (
    DEFAULT_FEATURE_ID,
    available_feature_ids,
    extract_features,
    feature_schema_contract,
    get_feature_extractor,
    get_feature_schema,
    normalize_feature_id,
    validate_checkpoint_feature_contract,
)
from .types import (
    ActionTransform,
    BoardFeatures,
    DiscreteFeatures,
    FeatureSchema,
    HybridFeatures,
    VectorFeatures,
)

__all__ = (
    "ACTIONS", "HORIZON", "DEFAULT_FEATURE_ID", "FeatureSchema",
    "FeatureContext", "ReachabilityResult", "build_context", "build_safety_context",
    "danger_at_steps", "earliest_danger_time", "danger_after_action",
    "ActionTransform", "DiscreteFeatures", "VectorFeatures", "BoardFeatures",
    "HybridFeatures", "available_feature_ids", "get_feature_schema",
    "get_feature_extractor", "extract_features", "normalize_feature_id",
    "feature_schema_contract", "validate_checkpoint_feature_contract",
)

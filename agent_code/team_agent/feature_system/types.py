"""Public types used by every versioned feature representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureSchema:
    feature_id: str
    output_kind: str
    action_order: Tuple[str, ...]
    state_fields: Optional[Tuple[str, ...]] = None
    category_counts: Optional[Tuple[int, ...]] = None
    vector_fields: Optional[Tuple[str, ...]] = None
    board_channels: Optional[Tuple[str, ...]] = None
    state_shape: Optional[Tuple[int, ...]] = None
    vector_shape: Optional[Tuple[int, ...]] = None
    board_shape: Optional[Tuple[Optional[int], ...]] = None
    theoretical_state_count: Optional[int] = None
    normalization: Optional[Mapping[str, str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable contract for metadata/checkpoints."""
        return {
            "feature_id": self.feature_id,
            "output_kind": self.output_kind,
            "action_order": list(self.action_order),
            "state_fields": None if self.state_fields is None else list(self.state_fields),
            "category_counts": (
                None if self.category_counts is None else list(self.category_counts)
            ),
            "vector_fields": None if self.vector_fields is None else list(self.vector_fields),
            "board_channels": (
                None if self.board_channels is None else list(self.board_channels)
            ),
            "state_shape": None if self.state_shape is None else list(self.state_shape),
            "vector_shape": None if self.vector_shape is None else list(self.vector_shape),
            "board_shape": None if self.board_shape is None else list(self.board_shape),
            "theoretical_state_count": self.theoretical_state_count,
            "normalization": dict(self.normalization or {}),
        }


@dataclass(frozen=True)
class ActionTransform:
    world_to_canonical: Tuple[int, ...]
    canonical_to_world: Tuple[int, ...]


IDENTITY_ACTION_TRANSFORM = ActionTransform(tuple(range(6)), tuple(range(6)))


@dataclass(frozen=True)
class DiscreteFeatures:
    feature_id: str
    state_key: Tuple[int, ...]
    vector: np.ndarray
    legal_mask: np.ndarray
    action_transform: ActionTransform = IDENTITY_ACTION_TRANSFORM


@dataclass(frozen=True)
class VectorFeatures:
    feature_id: str
    vector: np.ndarray
    legal_mask: np.ndarray


@dataclass(frozen=True)
class BoardFeatures:
    feature_id: str
    board: np.ndarray
    legal_mask: np.ndarray


@dataclass(frozen=True)
class HybridFeatures:
    feature_id: str
    board: np.ndarray
    vector: np.ndarray
    legal_mask: np.ndarray

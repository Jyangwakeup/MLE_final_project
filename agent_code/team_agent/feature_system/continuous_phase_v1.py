"""Phase-aware extension of continuous-v3 for Task 3 and Task 4."""

from __future__ import annotations

import numpy as np

from ..phase import PHASE_STATE_FIELDS, phase_facts
from . import continuous_v3
from .common import ACTIONS
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-phase-v1"
FEATURE_DIM = 117
VECTOR_FIELDS = continuous_v3.VECTOR_FIELDS + PHASE_STATE_FIELDS


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v3.schema().normalization,
            "phase_weights": "continuous triangular mixture; sums to one",
            "depletion": "fraction of initial crates/opponents removed, or zero without any",
            "score_margin": "clip((self-best alive opponent)/10,-1,1)",
            "mobility": "log1p(best H=7 endpoint area)/log1p(walkable tiles)",
            "opponent_closeness": "exp(-Manhattan distance/4)",
            "late_mobility_crisis": "late and safe actions<=1 and endpoint ratio<=0.10",
        },
    )


def extract(
    game_state: dict,
    previous_action: str | None = None,
    wait_streak: int = 0,
    own_bomb: dict[str, object] | None = None,
    previous_position: tuple[int, int] | None = None,
    previous_coin_target: tuple[int, int] | None = None,
    *,
    initial_crates: int | None = None,
    initial_opponents: int | None = None,
    phase_values: dict[str, float] | None = None,
) -> VectorFeatures | None:
    if game_state is None:
        return None
    base = continuous_v3.extract(
        game_state, previous_action, wait_streak, own_bomb,
        previous_position, previous_coin_target,
    )
    facts = phase_values or phase_facts(
        game_state,
        initial_crates=(
            int(np.count_nonzero(game_state["field"] == 1))
            if initial_crates is None else int(initial_crates)
        ),
        initial_opponents=(
            len(game_state.get("others", ()))
            if initial_opponents is None else int(initial_opponents)
        ),
    )
    suffix = np.asarray([facts[name] for name in PHASE_STATE_FIELDS], dtype=np.float32)
    vector = np.concatenate((base.vector, suffix)).astype(np.float32, copy=False)
    return VectorFeatures(FEATURE_ID, vector, base.legal_mask.copy())

"""Continuous objective features with explicit action-conditional safety margins."""

from __future__ import annotations

from math import log1p

import numpy as np

import settings as s
from . import continuous_v1
from ..danger import HORIZON, blast_coords
from ..temporal_safety_features import safety_margin_after_first_step, temporal_maps
from . import continuous_v2
from .common import ACTIONS, build_context
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v3"
SAFETY_FIELDS = tuple(
    f"{action.lower()}_{field}"
    for action in ACTIONS
    for field in ("survives_horizon", "escape_slack", "safe_second_action_fraction")
)
OWN_BOMB_FIELDS = (
    "safe_action_fraction", "own_bomb_pending", "own_bomb_visible",
    "own_bomb_timer", "in_own_blast_line",
)
VECTOR_FIELDS = continuous_v2.VECTOR_FIELDS + SAFETY_FIELDS + OWN_BOMB_FIELDS


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(107,),
        normalization={
            **continuous_v2.schema().normalization,
            "safe_area": "log1p(last safe frontier size)/log1p(width*height)",
            "survives_horizon": "binary 0 or 1",
            "escape_slack": "clip((danger deadline-shortest escape step)/H,-1,1)",
            "safe_second_action_fraction": "survivable second actions/5",
            "safe_action_fraction": "horizon-survivable physical actions/6",
            "own_bomb_facts": "binary flags and timer/BOMB_TIMER",
        },
    )


def extract(
    game_state: dict,
    previous_action: str | None = None,
    wait_streak: int = 0,
    own_bomb: dict[str, object] | None = None,
    previous_position: tuple[int, int] | None = None,
    previous_coin_target: tuple[int, int] | None = None,
) -> VectorFeatures | None:
    if game_state is None:
        return None
    context = build_context(game_state)
    base_features = continuous_v2.extract(
        game_state, previous_action=previous_action,
        previous_position=previous_position,
        previous_coin_target=previous_coin_target,
        _context=context,
    )
    base = base_features.vector.copy()
    area_scale = log1p(float(context.width * context.height))
    for index, action in enumerate(ACTIONS):
        offset = index * len(continuous_v1.ACTION_FIELDS)
        reachability = context.movement_reachability[action]
        base[offset + 6] = log1p(float(reachability.reachable_area)) / area_scale
    prefix = base

    normal_danger, normal_blocked = context.normal_danger, context.normal_blocked
    bomb_maps = (
        (context.bomb_danger, context.bomb_blocked)
        if bool(context.legal_mask[ACTIONS.index("BOMB")]) else None
    )
    safety_values = []
    survivable_count = 0
    for index, action in enumerate(ACTIONS):
        reachability = context.movement_reachability[action]
        survives = bool(context.legal_mask[index] and reachability.survives_horizon)
        survivable_count += int(survives)
        if not context.legal_mask[index]:
            margin, branches = -1.0, 0
        else:
            danger, blocked = bomb_maps if action == "BOMB" else (
                normal_danger, normal_blocked)
            first_action = "WAIT" if action == "BOMB" else action
            detail = safety_margin_after_first_step(
                context.position, first_action, danger, blocked)
            margin, branches = detail.escape_slack, detail.survivable_second_actions
        safety_values.extend((float(survives), float(margin), branches / 5.0))

    own_bomb = dict(own_bomb or {})
    own_position = own_bomb.get("position")
    pending = bool(own_bomb.get("pending", False))
    visible = bool(own_bomb.get("visible", False))
    timer = own_bomb.get("timer")
    in_blast = False
    if own_position is not None:
        in_blast = tuple(context.position) in set(
            blast_coords(game_state["field"], tuple(own_position), s.BOMB_POWER))
    globals_ = (
        survivable_count / float(len(ACTIONS)), float(pending), float(visible),
        0.0 if timer is None else max(0.0, min(float(timer), s.BOMB_TIMER)) / s.BOMB_TIMER,
        float(in_blast),
    )
    vector = np.concatenate((
        prefix, np.asarray(safety_values, dtype=np.float32),
        np.asarray(globals_, dtype=np.float32),
    )).astype(np.float32, copy=False)
    return VectorFeatures(FEATURE_ID, vector, context.legal_mask.copy(), context)

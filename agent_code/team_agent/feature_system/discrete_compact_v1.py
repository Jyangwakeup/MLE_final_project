"""Symmetry-normalized compact categorical features for tabular methods."""

from __future__ import annotations

from math import prod

import numpy as np

from .common import (
    ACTIONS,
    MOVE_ACTIONS,
    FeatureContext,
    action_destination,
    build_safety_context,
    crate_frontiers,
    distance_to_targets,
    navigation_blocked,
    physical_legal_mask,
)
from .symmetry import Symmetry, canonical_symmetry
from .types import DiscreteFeatures, FeatureSchema


FEATURE_ID = "discrete-compact-v1"
STATE_FIELDS = (
    "canonical_move_up_safety", "canonical_move_right_safety",
    "canonical_move_down_safety", "canonical_move_left_safety",
    "current_tile_danger_t1", "current_tile_danger_t2",
    "current_tile_danger_t3", "bomb_safety", "objective_kind",
    "objective_direction", "objective_distance", "bomb_utility",
)
CATEGORY_COUNTS = (3, 3, 3, 3, 2, 2, 2, 3, 4, 5, 4, 4)


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="discrete",
        action_order=ACTIONS,
        state_fields=STATE_FIELDS,
        category_counts=CATEGORY_COUNTS,
        state_shape=(12,),
        vector_shape=(38,),
        theoretical_state_count=prod(CATEGORY_COUNTS),
        normalization={
            "vector": "one-hot encoding of the canonical categorical state",
            "symmetry": "lexicographically minimal D4 state (shape-preserving subgroup on rectangles)",
        },
    )


def _transform_game_state(game_state: dict, symmetry: Symmetry) -> dict:
    transformed = dict(game_state)
    transformed["field"] = symmetry.transform_array(game_state["field"])
    transformed["explosion_map"] = symmetry.transform_array(game_state["explosion_map"])
    transformed["coins"] = [symmetry.coordinate(position) for position in game_state["coins"]]
    transformed["bombs"] = [
        (symmetry.coordinate(position), timer) for position, timer in game_state["bombs"]
    ]
    own = game_state["self"]
    transformed["self"] = (*own[:3], symmetry.coordinate(own[3]))
    transformed["others"] = [
        (*other[:3], symmetry.coordinate(other[3])) for other in game_state["others"]
    ]
    return transformed


def _movement_category(legal: bool, survives_horizon: bool) -> int:
    if not legal:
        return 0
    return 2 if survives_horizon else 1


def _target_distance(
    blocked: np.ndarray,
    target: tuple[int, int],
    origin: tuple[int, int],
    *,
    allow_blocked_target: bool,
) -> tuple[float, np.ndarray]:
    distances = distance_to_targets(
        blocked, (target,), allow_blocked_targets=allow_blocked_target)
    return float(distances[origin]), distances


def _select_objective(game_state: dict, context: FeatureContext):
    """Choose one deterministic reachable objective and its distance map."""
    blocked = navigation_blocked(game_state)
    # The grid is undirected, so one BFS from the agent gives the same path
    # length that the previous per-target BFS returned for every traversable
    # coin/frontier. Opponents occupy blocked endpoint tiles; their distance is
    # therefore one plus the best reachable orthogonal neighbour.
    from_origin = distance_to_targets(blocked, (context.position,))
    target_groups = (
        (1, tuple(game_state["coins"])),
        (2, crate_frontiers(game_state, blocked)),
        (3, tuple(other[3] for other in game_state["others"])),
    )
    candidates = []
    for kind, targets in target_groups:
        for target in sorted(set(targets)):
            if kind == 3:
                neighbouring = []
                for action in MOVE_ACTIONS:
                    candidate = action_destination(target, action)
                    if (0 <= candidate[0] < context.width
                            and 0 <= candidate[1] < context.height
                            and np.isfinite(from_origin[candidate])):
                        neighbouring.append(float(from_origin[candidate]) + 1.0)
                distance = min(neighbouring, default=np.inf)
            else:
                distance = float(from_origin[target])
            if np.isfinite(distance):
                candidates.append((distance, kind, target[0], target[1], target))
    if not candidates:
        return 0, None, None, None
    distance, kind, _, _, target = min(
        candidates, key=lambda item: item[:4])
    _, distances = _target_distance(
        blocked, target, context.position, allow_blocked_target=kind == 3)
    return kind, target, distance, distances


def _objective_direction(
    context: FeatureContext,
    kind: int,
    target: tuple[int, int],
    distance: float,
    distances: np.ndarray,
) -> int:
    if distance <= 0:
        return 0
    for index, action in enumerate(MOVE_ACTIONS):
        candidate = action_destination(context.position, action)
        endpoint_step = kind == 3 and candidate == target
        if (context.legal_mask[index] or endpoint_step) and distances[candidate] < distance:
            return index + 1
    return 0


def _distance_bucket(kind: int, distance: float) -> int:
    if kind == 0:
        return 0
    if distance <= 1:
        return 1
    if distance <= 4:
        return 2
    return 3


def _one_hot(state_key: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(sum(CATEGORY_COUNTS), dtype=np.float32)
    offset = 0
    for value, count in zip(state_key, CATEGORY_COUNTS):
        vector[offset + value] = 1.0
        offset += count
    return vector


def extract(game_state: dict) -> DiscreteFeatures:
    if game_state is None:
        return None
    symmetry = canonical_symmetry(game_state)
    canonical_state = _transform_game_state(game_state, symmetry)
    context = build_safety_context(canonical_state)
    movement = tuple(
        _movement_category(
            bool(context.legal_mask[index]),
            context.movement_reachability[action].survives_horizon,
        )
        for index, action in enumerate(MOVE_ACTIONS)
    )
    x, y = context.position
    exact_danger = tuple(int(context.normal_danger[step, x, y]) for step in (1, 2, 3))
    if not context.legal_mask[5]:
        bomb_safety = 0
    else:
        bomb_safety = 2 if context.bomb_reachability.survives_horizon else 1

    kind, target, distance, distances = _select_objective(canonical_state, context)
    direction = 0 if kind == 0 else _objective_direction(
        context, kind, target, distance, distances)
    distance_bucket = _distance_bucket(kind, distance if distance is not None else np.inf)
    hits_crate = context.crates_in_blast > 0
    hits_opponent = context.opponents_in_blast > 0
    if not context.legal_mask[5]:
        bomb_utility = 0
    else:
        bomb_utility = int(hits_crate) + 2 * int(hits_opponent)

    state_key = movement + exact_danger + (
        bomb_safety, kind, direction, distance_bucket, bomb_utility)
    return DiscreteFeatures(
        feature_id=FEATURE_ID,
        state_key=state_key,
        vector=_one_hot(state_key),
        legal_mask=physical_legal_mask(game_state),
        action_transform=symmetry.action_transform,
    )

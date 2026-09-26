"""Shared, objective calculations for all public feature representations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

import settings as s
from ..danger import HORIZON, blast_coords
from ..temporal_safety_features import (
    ReachabilityResult,
    detailed_reachability_after_first_step,
    detailed_reachability_all_first_steps,
    temporal_maps,
)


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_ACTIONS = ACTIONS[:4]
DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}
FREE = 0
CRATE = 1


@dataclass(frozen=True)
class FeatureContext:
    field: np.ndarray
    width: int
    height: int
    position: tuple[int, int]
    legal_mask: np.ndarray
    normal_danger: np.ndarray
    normal_blocked: np.ndarray
    bomb_danger: Optional[np.ndarray]
    bomb_blocked: Optional[np.ndarray]
    earliest_danger: np.ndarray
    movement_reachability: Mapping[str, ReachabilityResult]
    bomb_reachability: Optional[ReachabilityResult]
    coin_distance: np.ndarray
    crate_frontier_distance: np.ndarray
    opponent_distance: np.ndarray
    reachable_coin_exists: bool
    reachable_crate_frontier_exists: bool
    reachable_opponent_exists: bool
    crates_in_blast: int
    opponents_in_blast: int


def in_bounds(position: tuple[int, int], shape: tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < shape[0] and 0 <= y < shape[1]


def action_destination(position: tuple[int, int], action: str) -> tuple[int, int]:
    dx, dy = DIRECTIONS[action]
    return position[0] + dx, position[1] + dy


def physical_legal_mask(game_state: dict) -> np.ndarray:
    """Return physical legality only; danger never makes an action illegal."""
    field = game_state["field"]
    position = game_state["self"][3]
    occupied = {bomb_position for bomb_position, _ in game_state["bombs"]}
    occupied.update(other[3] for other in game_state["others"])
    values = []
    for action in MOVE_ACTIONS:
        destination = action_destination(position, action)
        values.append(
            in_bounds(destination, field.shape)
            and field[destination] == FREE
            and destination not in occupied
        )
    values.extend((True, bool(game_state["self"][2])))
    return np.asarray(values, dtype=bool)


def danger_at_steps(
    danger: np.ndarray, steps: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """Select exact future danger slices as an independent bool array."""
    if any(step <= 0 or step >= danger.shape[0] for step in steps):
        raise ValueError("danger step outside prediction horizon")
    return np.asarray(danger[list(steps)], dtype=bool).copy()


def earliest_danger_time(danger: np.ndarray) -> np.ndarray:
    """Return earliest t>=1 per tile; HORIZON+1 means no predicted danger."""
    result = np.full(danger.shape[1:], danger.shape[0], dtype=np.int16)
    for time_step in range(1, danger.shape[0]):
        unseen = result == danger.shape[0]
        result[unseen & danger[time_step]] = time_step
    return result


def danger_after_action(
    context: FeatureContext,
    action: str,
    hypothetical_bomb: bool = False,
) -> tuple[bool, bool, bool]:
    """Return exact t=1..3 danger at the fixed post-action position."""
    action_index = ACTIONS.index(action)
    if not context.legal_mask[action_index]:
        return False, False, False
    destination = action_destination(context.position, action)
    danger = (
        context.bomb_danger
        if action == "BOMB" or hypothetical_bomb
        else context.normal_danger
    )
    if danger is None:
        return False, False, False
    return tuple(bool(danger[step, destination[0], destination[1]]) for step in (1, 2, 3))


def navigation_blocked(game_state: dict) -> np.ndarray:
    blocked = np.asarray(game_state["field"] != FREE, dtype=bool).copy()
    for position, _ in game_state["bombs"]:
        blocked[position] = True
    for _, _, _, position in game_state["others"]:
        blocked[position] = True
    blocked[game_state["self"][3]] = False
    return blocked


def distance_to_targets(
    blocked: np.ndarray,
    targets: tuple[tuple[int, int], ...],
    *,
    allow_blocked_targets: bool = False,
) -> np.ndarray:
    """Return static shortest-path distance to the nearest permitted endpoint."""
    integer_distances = np.full(blocked.shape, -1, dtype=np.int16)
    frontier = deque()
    for target in sorted(set(targets)):
        if (
            in_bounds(target, blocked.shape)
            and (allow_blocked_targets or not blocked[target])
        ):
            integer_distances[target] = 0
            frontier.append(target)
    while frontier:
        x, y = frontier.popleft()
        next_distance = integer_distances[x, y] + 1
        # Inline the four neighbours: this BFS sits on every feature/reward
        # hot path, and avoiding action-string and helper dispatch is material
        # on the small 17x17 board.
        for xx, yy in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if (
                0 <= xx < blocked.shape[0]
                and 0 <= yy < blocked.shape[1]
                and not blocked[xx, yy]
                and integer_distances[xx, yy] < 0
            ):
                integer_distances[xx, yy] = next_distance
                frontier.append((xx, yy))
    distances = integer_distances.astype(np.float32)
    distances[integer_distances < 0] = np.inf
    return distances


def crate_frontiers(game_state: dict, blocked: np.ndarray) -> tuple[tuple[int, int], ...]:
    field = game_state["field"]
    frontiers = set()
    for x, y in np.argwhere(field == CRATE):
        for action in MOVE_ACTIONS:
            dx, dy = DIRECTIONS[action]
            candidate = int(x + dx), int(y + dy)
            if in_bounds(candidate, field.shape) and not blocked[candidate]:
                frontiers.add(candidate)
    return tuple(sorted(frontiers))


def _unavailable_reachability() -> ReachabilityResult:
    return ReachabilityResult(False, False, False, 0, frozenset(), 0)


def _build_context(game_state: dict, *, include_navigation: bool) -> FeatureContext:
    """Compute shared feature facts without mutating ``game_state``.

    Compact categorical features need temporal safety but calculate their one
    selected objective in canonical coordinates.  Letting that representation
    skip the three unused all-objective distance maps avoids redundant BFS
    passes while preserving the public full-context behaviour.
    """
    if game_state is None:
        raise ValueError("game_state must not be None")
    field = game_state["field"]
    width, height = field.shape
    position = game_state["self"][3]
    legal_mask = physical_legal_mask(game_state)

    normal_danger, normal_blocked = temporal_maps(
        game_state, HORIZON, hypothetical_bomb=False)
    movement_reachability = detailed_reachability_all_first_steps(
        position, normal_danger, normal_blocked)
    for index, action in enumerate(ACTIONS[:5]):
        result = movement_reachability[action]
        # The public legality contract comes from the current physical state.
        if result.legal != bool(legal_mask[index]):
            result = ReachabilityResult(
                bool(legal_mask[index]), result.safe_next, result.survives_horizon,
                result.safe_horizon, result.reachable_positions, result.reachable_area,
            )
        movement_reachability[action] = result

    bomb_danger = None
    bomb_blocked = None
    bomb_reachability = None
    if legal_mask[5]:
        bomb_danger, bomb_blocked = temporal_maps(
            game_state, HORIZON, hypothetical_bomb=True)
        raw_bomb = detailed_reachability_after_first_step(
            position, "WAIT", bomb_danger, bomb_blocked)
        bomb_reachability = ReachabilityResult(
            True, raw_bomb.safe_next, raw_bomb.survives_horizon,
            raw_bomb.safe_horizon, raw_bomb.reachable_positions,
            raw_bomb.reachable_area,
        )
    movement_reachability["BOMB"] = bomb_reachability or _unavailable_reachability()

    if include_navigation:
        blocked = navigation_blocked(game_state)
        coin_targets = tuple(game_state["coins"])
        frontier_targets = crate_frontiers(game_state, blocked)
        opponent_targets = tuple(other[3] for other in game_state["others"])
        coin_distance = distance_to_targets(blocked, coin_targets)
        frontier_distance = distance_to_targets(blocked, frontier_targets)
        opponent_distance = distance_to_targets(
            blocked, opponent_targets, allow_blocked_targets=True)
    else:
        # Keep the context type stable for internal consumers.  These small
        # arrays are never read by compact-v1 and are much cheaper than BFS.
        coin_distance = np.full(field.shape, np.inf, dtype=np.float32)
        frontier_distance = np.full(field.shape, np.inf, dtype=np.float32)
        opponent_distance = np.full(field.shape, np.inf, dtype=np.float32)

    blast = set(blast_coords(field, position, s.BOMB_POWER))
    crates_in_blast = sum(field[coordinate] == CRATE for coordinate in blast)
    opponents_in_blast = sum(other[3] in blast for other in game_state["others"])
    return FeatureContext(
        field=field,
        width=width,
        height=height,
        position=position,
        legal_mask=legal_mask,
        normal_danger=normal_danger,
        normal_blocked=normal_blocked,
        bomb_danger=bomb_danger,
        bomb_blocked=bomb_blocked,
        earliest_danger=earliest_danger_time(normal_danger),
        movement_reachability=movement_reachability,
        bomb_reachability=bomb_reachability,
        coin_distance=coin_distance,
        crate_frontier_distance=frontier_distance,
        opponent_distance=opponent_distance,
        reachable_coin_exists=bool(np.isfinite(coin_distance[position])),
        reachable_crate_frontier_exists=bool(np.isfinite(frontier_distance[position])),
        reachable_opponent_exists=bool(np.isfinite(opponent_distance[position])),
        crates_in_blast=int(crates_in_blast),
        opponents_in_blast=int(opponents_in_blast),
    )


def build_context(game_state: dict) -> FeatureContext:
    """Compute the complete shared feature context."""
    return _build_context(game_state, include_navigation=True)


def build_safety_context(game_state: dict) -> FeatureContext:
    """Compute temporal safety facts without unused objective-distance maps."""
    return _build_context(game_state, include_navigation=False)


def normalized_distance_delta(
    distances: np.ndarray,
    before: tuple[int, int],
    after: tuple[int, int],
) -> float:
    """Normalize a shortest-path change, including finite/unreachable transitions."""
    before_distance = float(distances[before])
    after_distance = float(distances[after])
    if not np.isfinite(before_distance) and not np.isfinite(after_distance):
        return 0.0
    if np.isfinite(before_distance) and not np.isfinite(after_distance):
        return -1.0
    if not np.isfinite(before_distance) and np.isfinite(after_distance):
        return 1.0
    denominator = max(1, int(distances.size) - 1)
    return float(np.clip((before_distance - after_distance) / denominator, -1.0, 1.0))

"""One-step opponent uncertainty for opponent-robust survival checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

import settings as s


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}


@dataclass(frozen=True)
class OpponentTransitionScenario:
    """One distinct result of simultaneous choices and sequential execution."""

    game_state: dict
    action_profile: tuple[str, ...]
    execution_order: tuple[int, ...]


def _initially_physical_actions(
    field: np.ndarray,
    position: tuple[int, int],
    bombs_left: bool,
    occupied: frozenset[tuple[int, int]],
    bomb_positions: frozenset[tuple[int, int]],
) -> tuple[str, ...]:
    actions = ["WAIT"]
    for action, (dx, dy) in DIRECTIONS.items():
        target = (position[0] + dx, position[1] + dy)
        if (
            0 <= target[0] < field.shape[0]
            and 0 <= target[1] < field.shape[1]
            and field[target] == 0
            and target not in occupied
            and target not in bomb_positions
        ):
            actions.append(action)
    if bombs_left:
        actions.append("BOMB")
    return tuple(actions)


def _outcome_key(state: dict) -> tuple:
    return (
        tuple(state["self"][3]),
        tuple((other[0], tuple(other[3]), bool(other[2])) for other in state["others"]),
        tuple(sorted((tuple(position), int(timer)) for position, timer in state["bombs"])),
    )


def enumerate_opponent_transition_scenarios(
    game_state: dict,
    own_action: str,
) -> tuple[OpponentTransitionScenario, ...]:
    """Enumerate distinct one-step action-phase outcomes.

    Every actor chooses from actions that are physically legal in the shared
    pre-step state.  The choices are then applied in every execution order with
    the same dynamic occupancy rule as ``GenericWorld.perform_agent_action``.
    The returned state is still immediately before bombs/explosions advance;
    the temporal danger model performs that deterministic part of the step.
    """
    if own_action not in ACTIONS:
        raise ValueError(f"unsupported own action: {own_action!r}")
    actors = [tuple(game_state["self"])] + [tuple(other) for other in game_state["others"]]
    field = np.asarray(game_state["field"])
    existing_bombs = tuple((tuple(position), int(timer)) for position, timer in game_state["bombs"])
    bomb_positions = frozenset(position for position, _ in existing_bombs)
    occupied = frozenset(tuple(actor[3]) for actor in actors)
    opponent_choices = [
        _initially_physical_actions(
            field, tuple(actor[3]), bool(actor[2]), occupied, bomb_positions)
        for actor in actors[1:]
    ]
    profiles = product(*opponent_choices) if opponent_choices else [()]
    unique: dict[tuple, OpponentTransitionScenario] = {}
    for opponent_actions in profiles:
        action_profile = (own_action, *opponent_actions)
        for order in permutations(range(len(actors))):
            positions = [tuple(actor[3]) for actor in actors]
            bombs_left = [bool(actor[2]) for actor in actors]
            bombs = list(existing_bombs)
            current_bomb_positions = set(bomb_positions)
            for actor_index in order:
                action = action_profile[actor_index]
                if action in DIRECTIONS:
                    dx, dy = DIRECTIONS[action]
                    target = (
                        positions[actor_index][0] + dx,
                        positions[actor_index][1] + dy,
                    )
                    other_positions = {
                        position for index, position in enumerate(positions)
                        if index != actor_index
                    }
                    if (
                        0 <= target[0] < field.shape[0]
                        and 0 <= target[1] < field.shape[1]
                        and field[target] == 0
                        and target not in current_bomb_positions
                        and target not in other_positions
                    ):
                        positions[actor_index] = target
                elif action == "BOMB" and bombs_left[actor_index]:
                    bomb_position = positions[actor_index]
                    bombs.append((bomb_position, int(s.BOMB_TIMER)))
                    current_bomb_positions.add(bomb_position)
                    bombs_left[actor_index] = False

            own = (*actors[0][:2], bombs_left[0], positions[0])
            others = [
                (*actor[:2], bombs_left[index], positions[index])
                for index, actor in enumerate(actors[1:], start=1)
            ]
            scenario_state = {
                **game_state,
                "self": own,
                "others": others,
                "bombs": bombs,
            }
            key = _outcome_key(scenario_state)
            unique.setdefault(key, OpponentTransitionScenario(
                scenario_state, tuple(action_profile), tuple(order)))
    return tuple(unique.values())

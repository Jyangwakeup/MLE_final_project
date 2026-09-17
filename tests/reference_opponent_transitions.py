# Frozen reference from ae86fac; independent of production order reduction.
"""Pure opponent/action transitions used by robust bomb-safety search.

The functions in this module intentionally live inside ``agent_code``: frozen
submission inference must not import the repository's environment module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

import settings as s
from agent_code.team_agent.danger import blast_coords


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
DIRECTIONS = {
    "UP": (0, -1), "RIGHT": (1, 0),
    "DOWN": (0, 1), "LEFT": (-1, 0),
}


@dataclass(frozen=True)
class OpponentTransitionScenario:
    """One distinct full-step result under simultaneous action choices."""

    game_state: dict
    action_profile: tuple[str, ...]
    execution_order: tuple[int, ...]
    self_alive: bool = True


def physical_actions(field, position, bombs_left, occupied, bomb_positions):
    """Return actions legal in the shared state before execution begins."""
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
    if bombs_left and position not in bomb_positions:
        actions.append("BOMB")
    return tuple(actions)


def physical_actions_for_self(game_state: dict) -> tuple[str, ...]:
    actors = [tuple(game_state["self"])] + [
        tuple(other) for other in game_state["others"]]
    return physical_actions(
        np.asarray(game_state["field"]), tuple(actors[0][3]),
        bool(actors[0][2]), frozenset(tuple(actor[3]) for actor in actors),
        frozenset(tuple(position) for position, _ in game_state["bombs"]),
    )


def _state_key(state: dict, self_alive: bool = True) -> tuple:
    return (
        self_alive, tuple(state["self"][3]), bool(state["self"][2]),
        tuple(sorted((str(other[0]), tuple(other[3]), bool(other[2]))
                     for other in state["others"])),
        tuple(sorted((tuple(position), int(timer))
                     for position, timer in state["bombs"])),
        np.asarray(state["field"], dtype=np.int8).tobytes(),
        np.asarray(state["explosion_map"], dtype=np.int8).tobytes(),
    )


def canonical_state_key(state: dict) -> tuple:
    """Hashable identity for memoizing the finite safety game."""
    return _state_key(state)


def _apply_action_phase(game_state, action_profile, order):
    actors = [tuple(game_state["self"])] + [
        tuple(other) for other in game_state["others"]]
    field = np.asarray(game_state["field"])
    positions = [tuple(actor[3]) for actor in actors]
    bombs_left = [bool(actor[2]) for actor in actors]
    bombs = [(tuple(position), int(timer))
             for position, timer in game_state["bombs"]]
    bomb_positions = {position for position, _ in bombs}
    for actor_index in order:
        action = action_profile[actor_index]
        if action in DIRECTIONS:
            dx, dy = DIRECTIONS[action]
            target = (positions[actor_index][0] + dx,
                      positions[actor_index][1] + dy)
            other_positions = {
                position for index, position in enumerate(positions)
                if index != actor_index
            }
            if (
                0 <= target[0] < field.shape[0]
                and 0 <= target[1] < field.shape[1]
                and field[target] == 0
                and target not in bomb_positions
                and target not in other_positions
            ):
                positions[actor_index] = target
        elif action == "BOMB" and bombs_left[actor_index]:
            position = positions[actor_index]
            if position not in bomb_positions:
                bombs.append((position, int(s.BOMB_TIMER)))
                bomb_positions.add(position)
                bombs_left[actor_index] = False
    updated = [(*actor[:2], bombs_left[index], positions[index])
               for index, actor in enumerate(actors)]
    return updated, bombs


def _progress_world(game_state, actors, bombs):
    """Apply official post-action bomb/explosion ordering for one step."""
    field = np.asarray(game_state["field"], dtype=int).copy()
    current_explosions = np.asarray(game_state["explosion_map"], dtype=int)
    dangerous = current_explosions > 0
    next_explosions = np.maximum(current_explosions - 1, 0)
    remaining_bombs = []
    for position, timer in bombs:
        if timer <= 0:
            for coordinate in blast_coords(field, position, s.BOMB_POWER):
                if field[coordinate] == 1:
                    field[coordinate] = 0
                dangerous[coordinate] = True
                next_explosions[coordinate] = max(
                    int(next_explosions[coordinate]), s.EXPLOSION_TIMER - 1)
        else:
            remaining_bombs.append((position, timer - 1))
    self_alive = not bool(dangerous[tuple(actors[0][3])])
    surviving_others = [actor for actor in actors[1:]
                        if not bool(dangerous[tuple(actor[3])])]
    return ({
        **game_state,
        "step": int(game_state.get("step", 0)) + 1,
        "field": field,
        "self": actors[0],
        "others": surviving_others,
        "bombs": remaining_bombs,
        "explosion_map": next_explosions,
    }, self_alive)


def enumerate_opponent_transition_scenarios(
    game_state: dict, own_action: str, *, progress_world: bool = False,
) -> tuple[OpponentTransitionScenario, ...]:
    """Enumerate all distinct opponent choices and execution orders."""
    if own_action not in ACTIONS:
        raise ValueError(f"unsupported own action: {own_action!r}")
    actors = [tuple(game_state["self"])] + [
        tuple(other) for other in game_state["others"]]
    field = np.asarray(game_state["field"])
    bombs = tuple((tuple(position), int(timer))
                  for position, timer in game_state["bombs"])
    bomb_positions = frozenset(position for position, _ in bombs)
    occupied = frozenset(tuple(actor[3]) for actor in actors)
    opponent_choices = [
        physical_actions(field, tuple(actor[3]), bool(actor[2]),
                         occupied, bomb_positions)
        for actor in actors[1:]
    ]
    profiles = product(*opponent_choices) if opponent_choices else [()]
    unique = {}
    for opponent_actions in profiles:
        action_profile = (own_action, *opponent_actions)
        for order in permutations(range(len(actors))):
            updated, updated_bombs = _apply_action_phase(
                game_state, action_profile, tuple(order))
            if progress_world:
                scenario_state, self_alive = _progress_world(
                    game_state, updated, updated_bombs)
            else:
                scenario_state = {
                    **game_state, "self": updated[0], "others": updated[1:],
                    "bombs": updated_bombs,
                }
                self_alive = True
            key = _state_key(scenario_state, self_alive)
            unique.setdefault(key, OpponentTransitionScenario(
                scenario_state, tuple(action_profile), tuple(order), self_alive))
    return tuple(unique.values())

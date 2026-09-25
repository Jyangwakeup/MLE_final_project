from agent_code.learning_common.action_history import (
    action_history_state as base_action_history_state,
    init_action_history as base_init_action_history,
    load_action_history_state as base_load_action_history_state,
    own_bomb_history_for_state,
    record_selected_action as base_record_selected_action,
)
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.common import distance_to_targets, navigation_blocked
from agent_code.team_agent.feature_system.continuous_v5 import selected_crate_target
from agent_code.team_agent.feature_system.continuous_v6 import FEATURE_ID, extract


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def init_action_history(owner):
    base_init_action_history(owner)
    owner.feature_bomb_objective = None
    owner.feature_bomb_objective_ttl = 0
    owner.feature_bomb_objective_key = None
    owner.feature_tracked_opponent_name = None
    owner.feature_opponent_positions = {}
    owner.feature_opponent_key = None


def _ensure_state(owner):
    if not hasattr(owner, "feature_bomb_objective_ttl"):
        init_action_history(owner)


def _update_bomb_objective(owner, game_state):
    _ensure_state(owner)
    key = (game_state.get("round"), game_state.get("step"))
    if owner.feature_bomb_objective_key == key:
        return
    owner.feature_bomb_objective_key = key
    if bool(game_state["self"][2]) and owner.feature_bomb_objective is not None:
        owner.feature_bomb_objective_ttl = max(0, owner.feature_bomb_objective_ttl - 1)
        if owner.feature_bomb_objective_ttl == 0:
            owner.feature_bomb_objective = None


def _select_opponent(owner, game_state):
    _ensure_state(owner)
    opponents = {str(item[0]): tuple(item[3]) for item in game_state["others"]}
    previous_name = owner.feature_tracked_opponent_name
    if previous_name in opponents:
        name = previous_name
    elif opponents:
        position = tuple(game_state["self"][3])
        distances = distance_to_targets(
            navigation_blocked(game_state), tuple(opponents.values()),
            allow_blocked_targets=True)
        name = min(
            opponents,
            key=lambda candidate: (
                float(distances[opponents[candidate]]),
                abs(opponents[candidate][0] - position[0])
                + abs(opponents[candidate][1] - position[1]),
                candidate,
            ),
        )
    else:
        name = None
    return (
        name,
        None if name is None else opponents[name],
        bool(name is not None and name == previous_name),
        owner.feature_opponent_positions.get(name),
        opponents,
    )


def record_selected_action(owner, game_state, action):
    base_record_selected_action(owner, game_state, action)
    _update_bomb_objective(owner, game_state)
    if action == "BOMB" and bool(game_state["self"][2]):
        owner.feature_bomb_objective = selected_crate_target(game_state)
        owner.feature_bomb_objective_ttl = 24 if owner.feature_bomb_objective is not None else 0


def action_history_state(owner):
    state = base_action_history_state(owner)
    state.update({
        "bomb_objective": getattr(owner, "feature_bomb_objective", None),
        "bomb_objective_ttl": int(getattr(owner, "feature_bomb_objective_ttl", 0)),
        "bomb_objective_key": getattr(owner, "feature_bomb_objective_key", None),
        "tracked_opponent_name": getattr(owner, "feature_tracked_opponent_name", None),
        "opponent_positions": getattr(owner, "feature_opponent_positions", {}),
        "opponent_key": getattr(owner, "feature_opponent_key", None),
    })
    return state


def load_action_history_state(owner, state):
    base_load_action_history_state(owner, state)
    state = state or {}
    objective = state.get("bomb_objective")
    owner.feature_bomb_objective = None if objective is None else tuple(objective)
    owner.feature_bomb_objective_ttl = int(state.get("bomb_objective_ttl", 0))
    key = state.get("bomb_objective_key")
    owner.feature_bomb_objective_key = None if key is None else tuple(key)
    owner.feature_tracked_opponent_name = state.get("tracked_opponent_name")
    owner.feature_opponent_positions = {
        str(name): tuple(position)
        for name, position in state.get("opponent_positions", {}).items()
    }
    key = state.get("opponent_key")
    owner.feature_opponent_key = None if key is None else tuple(key)


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    _update_bomb_objective(owner, game_state)
    name, target, persisted, previous_target, opponents = _select_opponent(owner, game_state)
    result = extract(
        game_state, previous_action=previous_action, wait_streak=wait_streak,
        own_bomb=own_bomb_history_for_state(owner, game_state),
        previous_position=getattr(owner, "feature_previous_position", None),
        previous_coin_target=getattr(owner, "feature_previous_coin_target", None),
        position_history=getattr(owner, "feature_position_history", ()),
        bomb_objective=getattr(owner, "feature_bomb_objective", None),
        tracked_opponent=target, tracked_opponent_persisted=persisted,
        previous_opponent_position=previous_target,
    )
    key = (game_state.get("round"), game_state.get("step"))
    if owner.feature_opponent_key != key:
        owner.feature_opponent_key = key
        owner.feature_tracked_opponent_name = name
        owner.feature_opponent_positions = opponents
    return result

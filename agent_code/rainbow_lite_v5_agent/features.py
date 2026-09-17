from agent_code.learning_common.action_history import (
    action_history_state as base_action_history_state,
    init_action_history as base_init_action_history,
    load_action_history_state as base_load_action_history_state,
    own_bomb_history_for_state,
    record_selected_action as base_record_selected_action,
)
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v5 import (
    FEATURE_ID, extract, selected_crate_target,
)


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def init_action_history(owner):
    base_init_action_history(owner)
    owner.feature_bomb_objective = None
    owner.feature_bomb_objective_ttl = 0
    owner.feature_bomb_objective_key = None


def _update_bomb_objective(owner, game_state):
    if not hasattr(owner, "feature_bomb_objective_ttl"):
        owner.feature_bomb_objective = None
        owner.feature_bomb_objective_ttl = 0
        owner.feature_bomb_objective_key = None
    key = (game_state.get("round"), game_state.get("step"))
    if owner.feature_bomb_objective_key == key:
        return
    owner.feature_bomb_objective_key = key
    if bool(game_state["self"][2]) and owner.feature_bomb_objective is not None:
        owner.feature_bomb_objective_ttl = max(0, owner.feature_bomb_objective_ttl - 1)
        if owner.feature_bomb_objective_ttl == 0:
            owner.feature_bomb_objective = None


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


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    _update_bomb_objective(owner, game_state)
    return extract(
        game_state, previous_action=previous_action, wait_streak=wait_streak,
        own_bomb=own_bomb_history_for_state(owner, game_state),
        previous_position=getattr(owner, "feature_previous_position", None),
        previous_coin_target=getattr(owner, "feature_previous_coin_target", None),
        position_history=getattr(owner, "feature_position_history", ()),
        bomb_objective=getattr(owner, "feature_bomb_objective", None),
    )

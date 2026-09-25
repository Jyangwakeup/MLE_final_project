import numpy as np

from agent_code.rainbow_lite_v6_agent import features as base
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v7 import FEATURE_ID
from agent_code.team_agent.feature_system.types import VectorFeatures

FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def init_action_history(owner):
    base.init_action_history(owner)
    owner.feature_kill_round = None
    owner.feature_previous_score = 0
    owner.feature_own_kills = 0


def _update_kills(owner, game_state):
    round_index = game_state.get("round")
    score = int(game_state["self"][1])
    if getattr(owner, "feature_kill_round", None) != round_index:
        owner.feature_kill_round = round_index
        owner.feature_previous_score = score
        owner.feature_own_kills = 0
        return
    delta = max(0, score - int(owner.feature_previous_score))
    owner.feature_own_kills += delta // 5
    owner.feature_previous_score = score


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    _update_kills(owner, game_state)
    result = base.features_for_state(owner, game_state, previous_action, wait_streak)
    kills = max(0, min(int(owner.feature_own_kills), 3))
    vector = np.concatenate((result.vector, np.asarray(
        (kills / 3.0, float(kills >= 1)), dtype=np.float32)))
    return VectorFeatures(FEATURE_ID, vector, result.legal_mask.copy(), result.context)


def record_selected_action(owner, game_state, action):
    base.record_selected_action(owner, game_state, action)


def action_history_state(owner):
    state = base.action_history_state(owner)
    state.update({
        "kill_round": getattr(owner, "feature_kill_round", None),
        "previous_score": int(getattr(owner, "feature_previous_score", 0)),
        "own_kills": int(getattr(owner, "feature_own_kills", 0)),
    })
    return state


def load_action_history_state(owner, state):
    base.load_action_history_state(owner, state)
    state = state or {}
    owner.feature_kill_round = state.get("kill_round")
    owner.feature_previous_score = int(state.get("previous_score", 0))
    owner.feature_own_kills = int(state.get("own_kills", 0))

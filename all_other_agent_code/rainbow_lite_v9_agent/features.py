from all_other_agent_code.rainbow_lite_v8_agent import features as base
from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v9 import (
    FEATURE_ID, compact,
)
from agent_code.team_agent.feature_system.types import VectorFeatures


ACTIONS = base.ACTIONS
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)

init_action_history = base.init_action_history
record_selected_action = base.record_selected_action
action_history_state = base.action_history_state
load_action_history_state = base.load_action_history_state


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    result = base.features_for_state(
        owner, game_state, previous_action, wait_streak)
    return VectorFeatures(
        FEATURE_ID, compact(result.vector), result.legal_mask.copy(),
        result.context,
    )

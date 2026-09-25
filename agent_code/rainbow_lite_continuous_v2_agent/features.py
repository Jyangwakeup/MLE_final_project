from agent_code.learning_common.action_history import (
    action_history_state,
    init_action_history,
    load_action_history_state,
    record_selected_action,
)
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import FEATURE_ID, extract


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    return extract(
        game_state,
        previous_action=previous_action,
        previous_position=getattr(owner, "feature_previous_position", None),
        previous_coin_target=getattr(owner, "feature_previous_coin_target", None),
    )

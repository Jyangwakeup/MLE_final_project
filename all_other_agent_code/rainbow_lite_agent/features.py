from agent_code.learning_common.action_history import project_history, snapshot_history
from agent_code.learning_common.action_history import own_bomb_history_for_state
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v4 import FEATURE_ID, extract


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(
    owner, game_state, previous_action=None, wait_streak=0,
):
    return extract(
        game_state, previous_action=previous_action,
        wait_streak=wait_streak,
        own_bomb=own_bomb_history_for_state(owner, game_state),
        previous_position=getattr(owner, "feature_previous_position", None),
        previous_coin_target=getattr(owner, "feature_previous_coin_target", None),
        position_history=project_history(snapshot_history(owner), game_state).position_history,
    )

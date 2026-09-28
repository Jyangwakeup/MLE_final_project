from agent_code.die_hardest._vendor.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.die_hardest._vendor.team_agent.feature_system.continuous_v2 import (
    FEATURE_ID, extract,
)
from agent_code.die_hardest._vendor.team_agent.feature_system.continuous_v2_legacy78 import (
    extract as extract_legacy78,
)


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(
    game_state, previous_action=None, wait_streak=0, *,
    previous_position=None, previous_coin_target=None, legacy78=False,
):
    if legacy78:
        return extract_legacy78(game_state, previous_action, wait_streak)
    return extract(
        game_state, previous_action=previous_action,
        previous_position=previous_position,
        previous_coin_target=previous_coin_target,
    )

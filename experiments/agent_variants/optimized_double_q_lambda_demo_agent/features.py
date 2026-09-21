"""Keep the established continuous-v2 feature adapter unchanged."""

from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import FEATURE_ID, extract


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(
    owner, game_state, previous_action=None, wait_streak=0, *,
    previous_position=None, previous_coin_target=None,
):
    return extract(
        game_state, previous_action=previous_action,
        previous_position=previous_position,
        previous_coin_target=previous_coin_target,
    )

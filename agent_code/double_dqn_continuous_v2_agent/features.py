from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import extract


FEATURE_ID = "continuous-v2"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(game_state, previous_action=None, wait_streak=0):
    return extract(game_state, previous_action, wait_streak)

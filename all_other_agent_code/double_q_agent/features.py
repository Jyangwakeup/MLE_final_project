"""Objective-feature adapter for Double Q-learning."""

from agent_code.q_learning_agent.features import features_for_state as _extract
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract


FEATURE_ID = "discrete-objective-v1"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(game_state, previous_action=None, wait_streak=0):
    return _extract(game_state, previous_action, FEATURE_ID, wait_streak)

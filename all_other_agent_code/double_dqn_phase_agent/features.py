from agent_code.learning_common.action_history import own_bomb_history_for_state
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_phase_v1 import extract
from agent_code.team_agent.phase import ensure_phase_history, phase_facts_for_owner


FEATURE_ID = "continuous-phase-v1"
FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    ensure_phase_history(owner, game_state)
    return extract(
        game_state, previous_action, wait_streak,
        own_bomb_history_for_state(owner, game_state),
        previous_position=getattr(owner, "feature_previous_position", None),
        previous_coin_target=getattr(owner, "feature_previous_coin_target", None),
        initial_crates=owner.phase_initial_crates,
        initial_opponents=owner.phase_initial_opponents,
        phase_values=phase_facts_for_owner(owner, game_state),
    )

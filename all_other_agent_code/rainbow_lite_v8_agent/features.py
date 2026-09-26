import numpy as np
import settings as s

from all_other_agent_code.rainbow_lite_v7_agent import features as base
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v8 import FEATURE_ID
from agent_code.team_agent.feature_system.types import VectorFeatures


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


init_action_history = base.init_action_history
record_selected_action = base.record_selected_action
action_history_state = base.action_history_state
load_action_history_state = base.load_action_history_state


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    result = base.features_for_state(
        owner, game_state, previous_action, wait_streak)
    context = result.context
    bomb = context.bomb_reachability
    useful = (
        context.crates_in_blast / float(4 * s.BOMB_POWER)
        if bool(context.legal_mask[ACTIONS.index("BOMB")])
        and bomb is not None
        and bomb.survives_horizon
        and context.crates_in_blast > 0
        else 0.0
    )
    vector = np.concatenate((
        result.vector, np.asarray((useful,), dtype=np.float32)))
    return VectorFeatures(
        FEATURE_ID, vector.astype(np.float32, copy=False),
        result.legal_mask.copy(), context)

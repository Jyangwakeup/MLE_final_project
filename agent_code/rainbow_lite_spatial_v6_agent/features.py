"""Join existing v6 decision features with the 12-channel board."""

import numpy as np
import settings as s

from all_other_agent_code.rainbow_lite_v6_agent import features as base
from agent_code.team_agent.feature_system import ACTIONS, feature_schema_contract
from agent_code.team_agent.feature_system import board_v1
from agent_code.team_agent.feature_system.spatial_v6 import FEATURE_ID
from agent_code.team_agent.feature_system.types import VectorFeatures

FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID, (s.COLS, s.ROWS))
VECTOR_DIM = 160
BOARD_SHAPE = (12, s.COLS, s.ROWS)

init_action_history = base.init_action_history
record_selected_action = base.record_selected_action
action_history_state = base.action_history_state
load_action_history_state = base.load_action_history_state


def features_for_state(owner, game_state, previous_action=None, wait_streak=0):
    if game_state is None:
        return None
    if owner is None:
        from types import SimpleNamespace
        owner = SimpleNamespace()
        init_action_history(owner)
    vector = base.features_for_state(owner, game_state, previous_action, wait_streak)
    board = board_v1.build_from_context(game_state, vector.context)
    # board-v1 currently reuses `position` while iterating coins; restore the
    # self plane here without changing other agents' feature contracts.
    board.board[3] = 0.0
    x, y = game_state["self"][3]
    board.board[3, x, y] = 1.0
    if vector.vector.shape != (VECTOR_DIM,) or board.board.shape != BOARD_SHAPE:
        raise ValueError("spatial v6 input shape changed")
    joined = np.concatenate((vector.vector, board.board.reshape(-1))).astype(np.float32)
    return VectorFeatures(FEATURE_ID, joined, vector.legal_mask.copy(), vector.context)

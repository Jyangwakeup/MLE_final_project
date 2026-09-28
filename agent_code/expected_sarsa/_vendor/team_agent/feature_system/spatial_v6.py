"""Contract for the v6 vector followed by a 12-channel board."""

from . import board_v1, continuous_v6
from .common import ACTIONS
from .types import FeatureSchema

FEATURE_ID = "spatial-v6-board12"


def schema(board_shape=None):
    shape = (12, None, None) if board_shape is None else (12, *board_shape)
    width, height = shape[1:]
    size = None if width is None else continuous_v6.FEATURE_DIM + 12 * width * height
    return FeatureSchema(
        feature_id=FEATURE_ID, output_kind="vector", action_order=ACTIONS,
        board_channels=board_v1.BOARD_CHANNELS, board_shape=shape,
        vector_shape=None if size is None else (size,),
        normalization={"vector_prefix": continuous_v6.FEATURE_ID,
                       "board_suffix": board_v1.FEATURE_ID,
                       "self_plane": "exactly one 1 at game_state['self'][3]",
                       "layout": "vector then channel-major board[x,y]"},
    )


def extract(game_state):
    from agent_code.rainbow_lite_spatial_v6_agent.features import features_for_state
    return features_for_state(None, game_state)

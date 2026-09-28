"""Agent-centred quadrant-density Rainbow-lite V10 runtime."""

from pathlib import Path

from all_other_agent_code.rainbow_lite_agent import callbacks as _impl
from .features import (
    ACTIONS, FEATURE_ID, FEATURE_SCHEMA, features_for_state,
    load_action_history_state, record_selected_action,
)


ALGORITHM = "rainbow_lite"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = dict(_impl.HYPERPARAMETERS)
NETWORK_SPEC = {
    "id": "continuous-v10-agent-quadrant-density-dueling-mlp-128x2-per-n4",
    "input_shape": [170],
    "layers": ["Linear(170,128)", "ReLU", "Linear(128,128)", "ReLU", "Dueling(128,6)"],
    "dueling": True, "double_dqn": True, "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": NETWORK_SPEC,
    "hyperparameters": HYPERPARAMETERS,
}

_impl.ACTIONS = ACTIONS
_impl.FEATURE_ID = FEATURE_ID
_impl.FEATURE_SCHEMA = FEATURE_SCHEMA
_impl.MODEL_FILE = MODEL_FILE
_impl.HYPERPARAMETERS = HYPERPARAMETERS
_impl.NETWORK_SPEC = NETWORK_SPEC
_impl.features_for_state = features_for_state
_impl.load_action_history_state = load_action_history_state
_impl.record_selected_action = record_selected_action

setup = _impl.setup
act = _impl.act
state_to_features = _impl.state_to_features
legal_actions = _impl.legal_actions
_features_for = _impl._features_for

"""Low-drift Rainbow-lite V6 runtime for conservative continuation."""

from pathlib import Path

from all_other_agent_code.rainbow_lite_agent import callbacks as _base_impl
from all_other_agent_code.rainbow_lite_v6_agent import callbacks as _impl
from .features import (
    ACTIONS, FEATURE_ID, FEATURE_SCHEMA, features_for_state,
    load_action_history_state, record_selected_action,
)

ALGORITHM = "rainbow_lite"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = {
    **_impl.HYPERPARAMETERS,
    "learning_rate": 1e-4,
    "replay_capacity": 100_000,
    "train_interval": 4,
    "protected_replay_capacity": 10_000,
    "protected_reward_threshold": 4.0,
}
NETWORK_SPEC = dict(_impl.NETWORK_SPEC)
AGENT_METADATA = {
    "algorithm": ALGORITHM,
    "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name,
    "network_spec": NETWORK_SPEC,
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

# ``setup`` and ``act`` are defined in the base module, so their globals must
# be rebound there as well as on the V6 wrapper.  Keeping both bindings
# explicit prevents metadata/runtime hyperparameter drift.
_base_impl.ACTIONS = ACTIONS
_base_impl.FEATURE_ID = FEATURE_ID
_base_impl.FEATURE_SCHEMA = FEATURE_SCHEMA
_base_impl.MODEL_FILE = MODEL_FILE
_base_impl.HYPERPARAMETERS = HYPERPARAMETERS
_base_impl.NETWORK_SPEC = NETWORK_SPEC
_base_impl.features_for_state = features_for_state
_base_impl.load_action_history_state = load_action_history_state
_base_impl.record_selected_action = record_selected_action

setup = _impl.setup
act = _impl.act
state_to_features = _impl.state_to_features
legal_actions = _impl.legal_actions
_features_for = _impl._features_for

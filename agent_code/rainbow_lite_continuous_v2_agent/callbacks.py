"""Continuous-v2 specialization of the shared Rainbow-lite runtime."""

from pathlib import Path

import torch

from agent_code.rainbow_lite_agent import callbacks as _impl
from .features import (
    ACTIONS,
    FEATURE_ID,
    FEATURE_SCHEMA,
    features_for_state,
    load_action_history_state,
    record_selected_action,
)


ALGORITHM = "rainbow_lite"
MODEL_FILE = Path(__file__).with_name("final.pt")
HYPERPARAMETERS = dict(_impl.HYPERPARAMETERS)
NETWORK_SPEC = {
    "id": "continuous-v2-dueling-mlp-128x2-per-n4",
    "input_shape": [84],
    "layers": [
        "Linear(84,128)", "ReLU", "Linear(128,128)", "ReLU",
        "Dueling(128,6)",
    ],
    "dueling": True,
    "double_dqn": True,
    "output_actions": 6,
}
AGENT_METADATA = {
    "algorithm": ALGORITHM,
    "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name,
    "network_spec": NETWORK_SPEC,
    "hyperparameters": HYPERPARAMETERS,
}

# The shared implementation reads these contracts from module globals. Each
# experiment runs its learning agent in an isolated process.
_impl.ACTIONS = ACTIONS
_impl.FEATURE_ID = FEATURE_ID
_impl.FEATURE_SCHEMA = FEATURE_SCHEMA
_impl.MODEL_FILE = MODEL_FILE
_impl.HYPERPARAMETERS = HYPERPARAMETERS
_impl.NETWORK_SPEC = NETWORK_SPEC
_impl.features_for_state = features_for_state
_impl.load_action_history_state = load_action_history_state
_impl.record_selected_action = record_selected_action

_V5_COUNTERS = (
    "robust_safety_interventions",
    "robust_to_v1_fallbacks",
    "opponent_robust_interventions",
    "opponent_to_v3_fallbacks",
    "opponent_scenarios_evaluated",
    "robust_guarantee_losses",
    "robust_search_timeouts",
    "robust_states_evaluated",
    "v1_to_physical_fallbacks",
    "avoidable_escape_collapses",
    "own_bomb_cycles",
)


def setup(self):
    _impl.setup(self)
    if not self.model_file.exists():
        return
    checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
    for name in _V5_COUNTERS:
        setattr(self, name, int(checkpoint.get(name, 0)))
    escape = checkpoint.get("own_bomb_escape_state", {})
    self._own_bomb_cycle_had_safe_alternative = bool(
        escape.get("had_safe_alternative", False))
    self._own_bomb_cycle_collapse_recorded = bool(
        escape.get("collapse_recorded", False))
    self._own_bomb_placement_certificate = escape.get(
        "placement_certificate")


act = _impl.act
state_to_features = _impl.state_to_features
legal_actions = _impl.legal_actions
_features_for = _impl._features_for

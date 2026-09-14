import os
from pathlib import Path
import shutil

from agent_code.learning_common.neural_agent import (
    neural_end_round, neural_game_events, setup_neural_training,
)
from agent_code.team_agent.feature_system import ACTIONS

from .callbacks import (
    ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)
from .features import features_for_state, reset_history


def _extract(owner):
    return lambda state: features_for_state(owner, state)


def _state(features):
    return features.board.copy()


def setup_training(self):
    setup_neural_training(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    neural_game_events(
        self, old_game_state, self_action, new_game_state, events,
        actions=ACTIONS, extractor=_extract(self), state_value=_state,
    )


def end_of_round(self, last_game_state, last_action, events):
    neural_end_round(
        self, last_game_state, last_action, events, actions=ACTIONS,
        extractor=_extract(self), state_value=_state, algorithm=ALGORITHM,
        feature_id=FEATURE_ID, feature_schema=FEATURE_SCHEMA,
        hyperparameters=HYPERPARAMETERS, network_spec=NETWORK_SPEC,
    )
    run_dir = os.getenv("BOMBERMAN_RUN_DIR")
    action_steps = int(getattr(self, "stage_action_steps", 0))
    milestone = action_steps // 25_000
    previous = int(getattr(self, "cnn_path_snapshot_milestone", 0))
    if run_dir and milestone > previous:
        snapshots = Path(run_dir) / "checkpoints" / "snapshots"
        snapshots.mkdir(exist_ok=True)
        shutil.copy2(self.model_file, snapshots / f"policy_{milestone * 25_000:06d}.pt")
        self.cnn_path_snapshot_milestone = milestone
    reset_history(self)

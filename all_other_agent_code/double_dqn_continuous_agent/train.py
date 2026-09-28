from agent_code.learning_common.neural_agent import (
    neural_end_round, neural_game_events, setup_neural_training,
)
from .callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
)
from .features import features_for_state


def _state(features):
    return features.vector.copy()


def setup_training(self):
    setup_neural_training(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    extractor = lambda state: features_for_state(
        state, previous_action=getattr(self, "previous_action", None),
        previous_position=getattr(self, "reward_previous_position", None),
        previous_coin_target=getattr(self, "reward_previous_coin_target", None))
    neural_game_events(
        self, old_game_state, self_action, new_game_state, events,
        actions=ACTIONS, extractor=extractor, state_value=_state,
    )


def end_of_round(self, last_game_state, last_action, events):
    extractor = lambda state: features_for_state(
        state, previous_action=getattr(self, "previous_action", None),
        previous_position=getattr(self, "reward_previous_position", None),
        previous_coin_target=getattr(self, "reward_previous_coin_target", None))
    neural_end_round(
        self, last_game_state, last_action, events, actions=ACTIONS,
        extractor=extractor, state_value=_state, algorithm=ALGORITHM,
        feature_id=FEATURE_ID, feature_schema=FEATURE_SCHEMA,
        hyperparameters=HYPERPARAMETERS, network_spec=NETWORK_SPEC,
    )

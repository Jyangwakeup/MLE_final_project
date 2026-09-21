"""Online Watkins Double Q(lambda) callbacks after demonstration pretraining."""

from agent_code.learning_common import linear_agent


def setup_training(self):
    linear_agent.setup_training(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    linear_agent.game_events(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state, last_action, events):
    linear_agent.end_round(self, last_game_state, last_action, events)

"""Training callbacks for phase-aware Rainbow-lite V7."""

from . import callbacks as _callbacks  # noqa: F401
from agent_code.rainbow_lite_agent import train as _impl
from .features import action_history_state, init_action_history

_impl.action_history_state = action_history_state
_impl.init_action_history = init_action_history

from agent_code.rainbow_lite_agent.train import (  # noqa: E402
    end_of_round, game_events_occurred, setup_training,
)

__all__ = ["setup_training", "game_events_occurred", "end_of_round"]

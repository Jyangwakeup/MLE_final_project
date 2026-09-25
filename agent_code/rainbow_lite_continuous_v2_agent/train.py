"""Training callbacks for the continuous-v2 Rainbow-lite specialization."""

from . import callbacks as _callbacks  # noqa: F401
from agent_code.rainbow_lite_agent import train as _impl
from .features import action_history_state, init_action_history
import torch


_impl.action_history_state = action_history_state
_impl.init_action_history = init_action_history

from agent_code.rainbow_lite_agent.train import (  # noqa: E402
    end_of_round as _base_end_of_round,
    game_events_occurred,
    setup_training,
)


def end_of_round(self, last_game_state, last_action, events):
    _base_end_of_round(self, last_game_state, last_action, events)
    checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
    for name in _callbacks._V5_COUNTERS:
        checkpoint[name] = int(getattr(self, name, 0))
    checkpoint["own_bomb_escape_state"] = {
        "had_safe_alternative": bool(getattr(
            self, "_own_bomb_cycle_had_safe_alternative", False)),
        "collapse_recorded": bool(getattr(
            self, "_own_bomb_cycle_collapse_recorded", False)),
        "placement_certificate": getattr(
            self, "_own_bomb_placement_certificate", None),
    }
    checkpoint["safety_replay_spec"] = self.safety_replay_spec
    temporary = self.model_file.with_name(self.model_file.name + ".v11.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(self.model_file)

__all__ = ["setup_training", "game_events_occurred", "end_of_round"]

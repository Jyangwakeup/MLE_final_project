"""Action-history state that has identical training and evaluation semantics."""

from __future__ import annotations


def init_action_history(owner) -> None:
    owner.feature_previous_action = None
    owner.feature_wait_streak = 0
    owner.feature_history_round = None


def action_history_for_state(owner, game_state: dict) -> tuple[str | None, int]:
    """Return history before ``game_state`` and reset it at a round boundary."""
    round_index = game_state.get("round")
    if getattr(owner, "feature_history_round", None) != round_index:
        owner.feature_previous_action = None
        owner.feature_wait_streak = 0
        owner.feature_history_round = round_index
    return (
        getattr(owner, "feature_previous_action", None),
        int(getattr(owner, "feature_wait_streak", 0)),
    )


def record_selected_action(owner, game_state: dict, action: str) -> None:
    """Advance feature history immediately after an action is selected."""
    action_history_for_state(owner, game_state)
    owner.feature_previous_action = action
    owner.feature_wait_streak = (
        int(owner.feature_wait_streak) + 1 if action == "WAIT" else 0
    )


def action_history_state(owner) -> dict[str, object]:
    return {
        "previous_action": getattr(owner, "feature_previous_action", None),
        "wait_streak": int(getattr(owner, "feature_wait_streak", 0)),
        "round": getattr(owner, "feature_history_round", None),
    }


def load_action_history_state(owner, state: dict | None) -> None:
    init_action_history(owner)
    if not state:
        return
    owner.feature_previous_action = state.get("previous_action")
    owner.feature_wait_streak = int(state.get("wait_streak", 0))
    owner.feature_history_round = state.get("round")

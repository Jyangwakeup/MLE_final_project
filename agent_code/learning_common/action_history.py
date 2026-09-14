"""Action-history state that has identical training and evaluation semantics."""

from __future__ import annotations


def init_action_history(owner) -> None:
    owner.feature_previous_action = None
    owner.feature_wait_streak = 0
    owner.feature_history_round = None
    owner.feature_own_bomb_position = None
    owner.feature_own_bomb_pending = False
    owner.feature_previous_position = None
    owner.feature_previous_coin_target = None


def action_history_for_state(owner, game_state: dict) -> tuple[str | None, int]:
    """Return history before ``game_state`` and reset it at a round boundary."""
    round_index = game_state.get("round")
    if getattr(owner, "feature_history_round", None) != round_index:
        owner.feature_previous_action = None
        owner.feature_wait_streak = 0
        owner.feature_history_round = round_index
        owner.feature_own_bomb_position = None
        owner.feature_own_bomb_pending = False
        owner.feature_previous_position = None
        owner.feature_previous_coin_target = None
    if bool(game_state["self"][2]):
        owner.feature_own_bomb_position = None
        owner.feature_own_bomb_pending = False
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
    from agent_code.team_agent.feature_system.continuous_v2 import selected_coin_target
    owner.feature_previous_position = tuple(game_state["self"][3])
    owner.feature_previous_coin_target = selected_coin_target(game_state)
    if action == "BOMB" and bool(game_state["self"][2]):
        owner.feature_own_bomb_position = tuple(game_state["self"][3])
        owner.feature_own_bomb_pending = True


def action_history_state(owner) -> dict[str, object]:
    return {
        "previous_action": getattr(owner, "feature_previous_action", None),
        "wait_streak": int(getattr(owner, "feature_wait_streak", 0)),
        "round": getattr(owner, "feature_history_round", None),
        "own_bomb_position": getattr(owner, "feature_own_bomb_position", None),
        "own_bomb_pending": bool(getattr(owner, "feature_own_bomb_pending", False)),
        "previous_position": getattr(owner, "feature_previous_position", None),
        "previous_coin_target": getattr(owner, "feature_previous_coin_target", None),
    }


def load_action_history_state(owner, state: dict | None) -> None:
    init_action_history(owner)
    if not state:
        return
    owner.feature_previous_action = state.get("previous_action")
    owner.feature_wait_streak = int(state.get("wait_streak", 0))
    owner.feature_history_round = state.get("round")
    position = state.get("own_bomb_position")
    owner.feature_own_bomb_position = None if position is None else tuple(position)
    owner.feature_own_bomb_pending = bool(state.get("own_bomb_pending", False))
    previous_position = state.get("previous_position")
    owner.feature_previous_position = (
        None if previous_position is None else tuple(previous_position))
    previous_target = state.get("previous_coin_target")
    owner.feature_previous_coin_target = (
        None if previous_target is None else tuple(previous_target))


def own_bomb_history_for_state(owner, game_state: dict) -> dict[str, object]:
    """Return factual state for the most recently placed own bomb."""
    action_history_for_state(owner, game_state)
    position = getattr(owner, "feature_own_bomb_position", None)
    timer = None
    if position is not None:
        for bomb_position, bomb_timer in game_state["bombs"]:
            if tuple(bomb_position) == tuple(position):
                timer = int(bomb_timer)
                break
    return {
        "position": position,
        "pending": bool(getattr(owner, "feature_own_bomb_pending", False)),
        "visible": timer is not None,
        "timer": timer,
    }

"""Action-history state that has identical training and evaluation semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class HistorySnapshot:
    previous_action: str | None = None
    wait_streak: int = 0
    round: int | None = None
    own_bomb_position: tuple | None = None
    own_bomb_pending: bool = False
    own_bomb_placed_step: int | None = None
    previous_position: tuple | None = None
    previous_coin_target: tuple | None = None
    position_history: tuple = ()
    position_history_key: tuple | None = None


def snapshot_history(owner) -> HistorySnapshot:
    state = action_history_state(owner)
    state["position_history"] = tuple(state["position_history"])
    return HistorySnapshot(**state)


def project_history(history: HistorySnapshot, game_state: dict) -> HistorySnapshot:
    """Project a snapshot onto an observation without advancing live history."""
    if history.round != game_state.get('round'):
        history = HistorySnapshot(round=game_state.get('round'))
    if bool(game_state['self'][2]):
        history = replace(history, own_bomb_position=None, own_bomb_pending=False,
                          own_bomb_placed_step=None)
    key = (game_state.get("round"), game_state.get("step"))
    if history.position_history_key != key:
        positions = (*history.position_history, tuple(game_state["self"][3]))
        history = replace(history, position_history=positions[-POSITION_HISTORY_LIMIT:],
                          position_history_key=key)
    return history


def advance_observation(owner, game_state: dict) -> HistorySnapshot:
    history = project_history(snapshot_history(owner), game_state)
    load_action_history_state(owner, asdict(history))
    return history


def bomb_history(history: HistorySnapshot, game_state: dict) -> dict[str, object]:
    """Read bomb visibility from a state-specific immutable history."""
    position = history.own_bomb_position
    timer = next((int(timer) for pos, timer in game_state['bombs']
                  if position is not None and tuple(pos) == position), None)
    return dict(position=position, pending=history.own_bomb_pending,
                visible=timer is not None, timer=timer, placed_step=history.own_bomb_placed_step)


POSITION_HISTORY_LIMIT = 33


def init_action_history(owner) -> None:
    owner.feature_previous_action = None
    owner.feature_wait_streak = 0
    owner.feature_history_round = None
    owner.feature_own_bomb_position = None
    owner.feature_own_bomb_pending = False
    owner.feature_own_bomb_placed_step = None
    owner.feature_previous_position = None
    owner.feature_previous_coin_target = None
    owner.feature_position_history = []
    owner.feature_position_history_key = None


def action_history_for_state(owner, game_state: dict) -> tuple[str | None, int]:
    """Read projected history without modifying live decision state."""
    history = project_history(snapshot_history(owner), game_state)
    return history.previous_action, history.wait_streak


def record_selected_action(owner, game_state: dict, action: str) -> None:
    """Advance feature history immediately after an action is selected."""
    advance_observation(owner, game_state)
    owner.feature_previous_action = action
    owner.feature_wait_streak = (
        int(owner.feature_wait_streak) + 1 if action == "WAIT" else 0
    )
    from ..team_agent.feature_system.continuous_v2 import selected_coin_target
    owner.feature_previous_position = tuple(int(value) for value in game_state["self"][3])
    target = selected_coin_target(game_state)
    owner.feature_previous_coin_target = (
        None if target is None else tuple(int(value) for value in target))
    if action == "BOMB" and bool(game_state["self"][2]):
        owner.feature_own_bomb_position = tuple(
            int(value) for value in game_state["self"][3])
        owner.feature_own_bomb_pending = True
        owner.feature_own_bomb_placed_step = int(game_state["step"])


def action_history_state(owner) -> dict[str, object]:
    return {
        "previous_action": getattr(owner, "feature_previous_action", None),
        "wait_streak": int(getattr(owner, "feature_wait_streak", 0)),
        "round": getattr(owner, "feature_history_round", None),
        "own_bomb_position": getattr(owner, "feature_own_bomb_position", None),
        "own_bomb_pending": bool(getattr(owner, "feature_own_bomb_pending", False)),
        "own_bomb_placed_step": getattr(owner, "feature_own_bomb_placed_step", None),
        "previous_position": getattr(owner, "feature_previous_position", None),
        "previous_coin_target": getattr(owner, "feature_previous_coin_target", None),
        "position_history": list(getattr(owner, "feature_position_history", ())),
        "position_history_key": getattr(owner, "feature_position_history_key", None),
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
    placed_step = state.get("own_bomb_placed_step")
    owner.feature_own_bomb_placed_step = None if placed_step is None else int(placed_step)
    previous_position = state.get("previous_position")
    owner.feature_previous_position = (
        None if previous_position is None else tuple(previous_position))
    previous_target = state.get("previous_coin_target")
    owner.feature_previous_coin_target = (
        None if previous_target is None else tuple(previous_target))
    owner.feature_position_history = [
        tuple(position) for position in state.get("position_history", ())
    ][-POSITION_HISTORY_LIMIT:]
    history_key = state.get("position_history_key")
    owner.feature_position_history_key = (
        None if history_key is None else tuple(history_key))


def own_bomb_history_for_state(owner, game_state: dict) -> dict[str, object]:
    """Return factual state for the most recently placed own bomb."""
    return bomb_history(project_history(snapshot_history(owner), game_state), game_state)

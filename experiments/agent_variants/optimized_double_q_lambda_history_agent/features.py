"""State-specific continuous-v2 adapter with real action-history inputs."""

from __future__ import annotations

from agent_code.learning_common.action_history import project_history, snapshot_history
from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import FEATURE_ID, extract


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)
_CACHE_LIMIT = 2


def _cache_for_round(owner, round_index):
    """Return the agent-local two-state cache, resetting it between rounds."""
    if getattr(owner, "history_input_cache_round", None) != round_index:
        owner.history_input_cache_round = round_index
        owner.history_input_cache = {}
    cache = getattr(owner, "history_input_cache", None)
    if cache is None:
        cache = owner.history_input_cache = {}
    return cache


def features_for_state(
    owner, game_state, previous_action=None, wait_streak=0, *,
    previous_position=None, previous_coin_target=None,
):
    """Encode an observation using the history that belongs to that state."""
    if game_state is None:
        return None
    key = (game_state.get("round"), game_state.get("step"))
    cache = _cache_for_round(owner, game_state.get("round"))
    if key in cache:
        return cache[key]

    if hasattr(owner, "feature_history_round"):
        history = project_history(snapshot_history(owner), game_state)
        previous_action = history.previous_action
        wait_streak = history.wait_streak
        previous_position = history.previous_position
        previous_coin_target = history.previous_coin_target

    # Accepted for compatibility with the shared adapter; continuous-v2 does
    # not encode the streak itself.
    del wait_streak
    value = extract(
        game_state,
        previous_action=previous_action,
        previous_position=previous_position,
        previous_coin_target=previous_coin_target,
    )
    cache[key] = value
    while len(cache) > _CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    return value

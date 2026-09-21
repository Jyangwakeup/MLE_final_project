"""Self-contained corrected continuous-v2 history adapter."""

from agent_code.learning_common.action_history import project_history, snapshot_history
from agent_code.team_agent.feature_system import feature_schema_contract
from agent_code.team_agent.feature_system.continuous_v2 import FEATURE_ID, extract


FEATURE_SCHEMA = feature_schema_contract(FEATURE_ID)


def features_for_state(
    owner, game_state, previous_action=None, wait_streak=0, *,
    previous_position=None, previous_coin_target=None,
):
    if game_state is None:
        return None
    round_index = game_state.get("round")
    if getattr(owner, "history_input_cache_round", None) != round_index:
        owner.history_input_cache_round = round_index
        owner.history_input_cache = {}
    cache = getattr(owner, "history_input_cache", None)
    if cache is None:
        cache = owner.history_input_cache = {}
    key = (round_index, game_state.get("step"))
    if key in cache:
        return cache[key]
    if hasattr(owner, "feature_history_round"):
        history = project_history(snapshot_history(owner), game_state)
        previous_action = history.previous_action
        previous_position = history.previous_position
        previous_coin_target = history.previous_coin_target
    del wait_streak
    value = extract(
        game_state, previous_action=previous_action,
        previous_position=previous_position,
        previous_coin_target=previous_coin_target)
    cache[key] = value
    while len(cache) > 2:
        cache.pop(next(iter(cache)))
    return value


__all__ = ["FEATURE_SCHEMA", "features_for_state"]

"""Read-only Q-learning decision traces for frozen evaluation runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_code.learning_common.linear_agent import repeated_cycle_mask
from agent_code.learning_common.runtime import effective_legal_mask
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.team_agent.feature_system.common import build_context
from agent_code.team_agent.safety import mask_for_decision


def q_learning_decision_record(
    owner: Any, game_state: dict[str, Any], action: str,
) -> dict[str, Any] | None:
    """Describe a completed frozen decision without modifying the owner.

    The linear agent caches the feature object it used immediately before
    choosing an action.  Reading that cache lets this diagnostic report the
    exact Q values and tile activations without recomputing or advancing
    action history.
    """
    config = getattr(owner, "linear_config", None)
    model = getattr(owner, "model", None)
    item = getattr(owner, "_feature_cache_value", None)
    if config is None or model is None or item is None:
        return None
    if tuple(getattr(config, "ACTIONS", ())) != ACTIONS:
        return None
    vector = np.asarray(item.vector, dtype=np.float32)
    physical = effective_legal_mask(
        item.legal_mask, ACTIONS, owner.curriculum_allows_bomb)
    values = np.asarray(model.q_values(vector), dtype=np.float64)
    raw_indices = np.flatnonzero(physical)
    if not len(raw_indices):
        return None
    raw_best = float(values[raw_indices].max())
    raw_ties = [int(index) for index in raw_indices if values[index] == raw_best]
    legal, fallback = mask_for_decision(
        game_state, physical, owner.safety_spec,
        allow_bomb=owner.curriculum_allows_bomb, exploring=False)
    legal = repeated_cycle_mask(item, legal, owner.reward_id)
    legal_indices = np.flatnonzero(legal)
    legal_best = float(values[legal_indices].max())
    legal_ties = [int(index) for index in legal_indices if values[index] == legal_best]
    selected = ACTIONS.index(action) if action in ACTIONS else None
    projections = getattr(model, "action_feature_indices", None)
    tiles = (
        [model.coder.encode(vector[indices]).astype(int).tolist()
         for indices in projections]
        if projections is not None else
        [model.coder.encode(vector).astype(int).tolist() for _ in ACTIONS]
    )
    context = build_context(game_state)
    history = tuple(getattr(owner, "feature_position_history", ()))
    position = tuple(int(value) for value in game_state["self"][3])
    return {
        "schema_version": "q-learning-decision-v1",
        "round": int(game_state.get("round", 0)),
        "step": int(game_state.get("step", 0)),
        "action": action,
        "selected_index": selected,
        "q_values": values.tolist(),
        "physical_mask": physical.astype(bool).tolist(),
        "legal_mask": legal.astype(bool).tolist(),
        "safety_fallback": bool(fallback),
        "raw_argmax_ties": raw_ties,
        "legal_argmax_ties": legal_ties,
        "raw_argmax_vetoed": bool(not legal[raw_ties[0]]),
        "in_current_danger": bool(context.normal_danger[1, position[0], position[1]]),
        "visible_coin_count": len(game_state.get("coins", ())),
        "reachable_coin_exists": bool(context.reachable_coin_exists),
        "reachable_crate_frontier_exists": bool(context.reachable_crate_frontier_exists),
        "crates_in_blast": int(context.crates_in_blast),
        "physical_useful_bomb": bool(
            physical[ACTIONS.index("BOMB")] and context.crates_in_blast > 0),
        "position": list(position),
        "position_history": [
            [int(value) for value in item] for item in history
        ],
        "position_repeat_count": int(sum(
            tuple(int(value) for value in item) == position for item in history)),
        "wait_streak_after_selection": int(
            getattr(owner, "feature_wait_streak", 0)),
        "tile_codes_by_action": tiles,
    }

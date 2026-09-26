"""Continuous-v3 plus bounded WAIT and periodic-position history."""

from __future__ import annotations

from math import log1p

import numpy as np

from . import continuous_v3
from .common import ACTIONS, MOVE_ACTIONS, action_destination, build_context
from .types import FeatureSchema, VectorFeatures


FEATURE_ID = "continuous-v4"
MAX_WAIT_STREAK = 8
MAX_CYCLE_LENGTH = 8
MAX_CYCLE_REPEATS = 4
HISTORY_FIELDS = ("wait_streak",) + tuple(
    f"{action.lower()}_{field}"
    for action in ACTIONS
    for field in ("revisits_recent_position", "cycle_length", "cycle_repeats")
)
VECTOR_FIELDS = continuous_v3.VECTOR_FIELDS + HISTORY_FIELDS
FEATURE_DIM = 126


def _cycle_detail(
    position_history: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    destination: tuple[int, int],
) -> tuple[bool, int, int]:
    """Describe the best periodic suffix after appending ``destination``.

    A first closure counts as one completed cycle. Further repeats require the
    complete intervening position block to match, rather than only its endpoint.
    """
    history = [tuple(position) for position in position_history]
    if not history or destination not in history[:-1]:
        return False, 0, 0
    sequence = history + [tuple(destination)]
    candidates = []
    for period in range(2, MAX_CYCLE_LENGTH + 1):
        if len(sequence) < period + 1 or sequence[-1] != sequence[-1 - period]:
            continue
        repeats = 1
        while repeats < MAX_CYCLE_REPEATS:
            right_end = len(sequence) - repeats * period
            left_end = right_end - period
            if left_end < 0:
                break
            if sequence[left_end:right_end] != sequence[right_end:right_end + period]:
                break
            repeats += 1
        candidates.append((repeats, period))
    if not candidates:
        return True, 0, 0
    repeats, period = max(candidates, key=lambda item: (item[0], -item[1]))
    return True, period, repeats


def schema(board_shape=None) -> FeatureSchema:
    return FeatureSchema(
        feature_id=FEATURE_ID,
        output_kind="vector",
        action_order=ACTIONS,
        vector_fields=VECTOR_FIELDS,
        vector_shape=(FEATURE_DIM,),
        normalization={
            **continuous_v3.schema().normalization,
            "wait_streak": "log1p(min(consecutive WAIT,8))/log1p(8)",
            "revisits_recent_position": "binary for physical move candidates",
            "cycle_length": "period in 2..8 divided by 8; zero if non-periodic",
            "cycle_repeats": "completed repeats capped at 4 and divided by 4",
        },
    )


def extract(
    game_state: dict,
    previous_action: str | None = None,
    wait_streak: int = 0,
    own_bomb: dict[str, object] | None = None,
    previous_position: tuple[int, int] | None = None,
    previous_coin_target: tuple[int, int] | None = None,
    position_history: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
) -> VectorFeatures | None:
    if game_state is None:
        return None
    base = continuous_v3.extract(
        game_state,
        previous_action=previous_action,
        wait_streak=wait_streak,
        own_bomb=own_bomb,
        previous_position=previous_position,
        previous_coin_target=previous_coin_target,
    )
    context = build_context(game_state)
    history_values = [
        log1p(min(max(int(wait_streak), 0), MAX_WAIT_STREAK))
        / log1p(MAX_WAIT_STREAK)
    ]
    for index, action in enumerate(ACTIONS):
        if action not in MOVE_ACTIONS or not bool(context.legal_mask[index]):
            history_values.extend((0.0, 0.0, 0.0))
            continue
        destination = action_destination(context.position, action)
        revisits, period, repeats = _cycle_detail(position_history, destination)
        history_values.extend((
            float(revisits),
            period / float(MAX_CYCLE_LENGTH),
            min(repeats, MAX_CYCLE_REPEATS) / float(MAX_CYCLE_REPEATS),
        ))
    vector = np.concatenate((
        base.vector, np.asarray(history_values, dtype=np.float32),
    )).astype(np.float32, copy=False)
    return VectorFeatures(FEATURE_ID, vector, base.legal_mask.copy(), base.context)

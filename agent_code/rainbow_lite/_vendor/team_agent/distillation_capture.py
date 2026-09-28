"""Optional frozen-policy fact capture used to build Task 3 teacher data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from agent_code.rainbow_lite._vendor.learning_common.action_history import own_bomb_history_for_state
from .feature_system.continuous_phase_v1 import extract as extract_phase
from .phase import ensure_phase_history, phase_facts_for_owner


CAPTURE_ENV = "BOMBERMAN_DISTILLATION_CAPTURE"


def maybe_capture_teacher_row(owner, game_state, q_values, legal_mask) -> None:
    path = os.getenv(CAPTURE_ENV)
    if not path:
        return
    ensure_phase_history(owner, game_state)
    phase = extract_phase(
        game_state,
        getattr(owner, "feature_previous_action", None),
        int(getattr(owner, "feature_wait_streak", 0)),
        own_bomb_history_for_state(owner, game_state),
        getattr(owner, "feature_previous_position", None),
        getattr(owner, "feature_previous_coin_target", None),
        initial_crates=owner.phase_initial_crates,
        initial_opponents=owner.phase_initial_opponents,
        phase_values=phase_facts_for_owner(owner, game_state),
    )
    row = {
        "environment_seed": int(os.environ["BOMBERMAN_CAPTURE_ENVIRONMENT_SEED"]),
        "task_id": int(os.environ["BOMBERMAN_CAPTURE_TASK_ID"]),
        "state": phase.vector.astype(np.float32).tolist(),
        "legal_mask": np.asarray(legal_mask, dtype=bool).tolist(),
        "teacher_q": np.asarray(q_values, dtype=np.float32).tolist(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, separators=(",", ":")) + "\n")

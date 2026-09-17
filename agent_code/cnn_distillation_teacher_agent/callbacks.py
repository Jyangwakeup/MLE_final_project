"""Run continuous-v2 unchanged while capturing aligned student observations."""

from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.double_dqn_continuous_v2_agent import callbacks as teacher
from agent_code.learning_common.runtime import effective_legal_mask
from agent_code.team_agent.feature_system import ACTIONS
from agent_code.cnn_distilled_double_dqn_agent.features import (
    FEATURE_ID as STUDENT_FEATURE_ID, features_for_state, initialize_history,
    record_position,
)


ALGORITHM = teacher.ALGORITHM
MODEL_FILE = teacher.MODEL_FILE
HYPERPARAMETERS = teacher.HYPERPARAMETERS
NETWORK_SPEC = teacher.NETWORK_SPEC
AGENT_METADATA = teacher.AGENT_METADATA
_ROWS = []
_REGISTERED = False
_TEACHER_HASH = None


def _flush():
    destination = os.getenv("BOMBERMAN_CNN_TEACHER_CAPTURE")
    if not destination or not _ROWS:
        return
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        boards=np.stack([row[0] for row in _ROWS]).astype(np.float32),
        legal_masks=np.stack([row[1] for row in _ROWS]).astype(bool),
        teacher_q=np.stack([row[2] for row in _ROWS]).astype(np.float32),
        environment_seeds=np.asarray([row[3] for row in _ROWS], dtype=np.int64),
        steps=np.asarray([row[4] for row in _ROWS], dtype=np.int16),
        feature_id=np.asarray(STUDENT_FEATURE_ID),
        teacher_checkpoint_sha256=np.asarray(_TEACHER_HASH),
    )


def setup(self):
    global _REGISTERED, _TEACHER_HASH
    teacher.setup(self)
    with Path(self.model_file).open("rb") as file:
        _TEACHER_HASH = hashlib.sha256(file.read()).hexdigest()
    self._cnn_student_history = SimpleNamespace()
    initialize_history(self._cnn_student_history)
    if not _REGISTERED:
        atexit.register(_flush)
        _REGISTERED = True


def act(self, game_state):
    teacher_features = teacher._features_for(self, game_state)
    physical = effective_legal_mask(
        teacher_features.legal_mask, ACTIONS, self.curriculum_allows_bomb)
    teacher_q = self.model.q_values(teacher_features.vector)
    student = features_for_state(self._cnn_student_history, game_state)
    _ROWS.append((
        student.board.copy(), physical.copy(), teacher_q.copy(),
        int(os.environ["BOMBERMAN_CAPTURE_ENVIRONMENT_SEED"]),
        int(game_state["step"]),
    ))
    action = teacher.act(self, game_state)
    record_position(self._cnn_student_history, game_state)
    return action


state_to_features = teacher.state_to_features
legal_actions = teacher.legal_actions

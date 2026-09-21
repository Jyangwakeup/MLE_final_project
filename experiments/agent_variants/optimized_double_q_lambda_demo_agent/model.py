"""Tile control with a trace-free update for frozen demonstrations."""

from __future__ import annotations

import numpy as np

from agent_code.learning_common.tile_coding import TraceControl


class DemonstrationTraceControl(TraceControl):
    def observe_demonstration(self, transition, *, rng, learning_rate=0.02):
        """Apply one 1-step Double-Q update without touching online state."""
        state = np.asarray(transition.state)
        projected = (
            state if self.action_feature_indices is None
            else state[self.action_feature_indices[transition.action]]
        )
        active = self.coder.encode(projected)
        estimator = int(rng.integers(2))
        current = float(self.weights[estimator, transition.action, active].sum())
        target = float(transition.reward)
        if not transition.done:
            legal = np.flatnonzero(np.asarray(transition.next_legal, dtype=bool))
            if not len(legal):
                raise ValueError("demonstration transition has no legal next action")
            selector = self.q_values(transition.next_state, estimator)
            selected = int(legal[np.argmax(selector[legal])])
            target += self.gamma * float(
                self.q_values(transition.next_state, 1 - estimator)[selected]
            )
        delta = target - current
        self.weights[estimator, transition.action, active] += (
            float(learning_rate) / self.coder.tilings
        ) * delta
        return abs(delta)

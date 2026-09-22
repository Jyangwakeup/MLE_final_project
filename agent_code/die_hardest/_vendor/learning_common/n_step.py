"""Deterministic round-local n-step return accumulator."""

from __future__ import annotations

from collections import deque


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        if n_step < 1:
            raise ValueError("n_step must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.pending = deque()

    def append(self, transition):
        self.pending.append(transition)
        emitted = []
        if transition.done:
            while self.pending:
                emitted.append(self._emit_one())
        elif len(self.pending) >= self.n_step:
            emitted.append(self._emit_one())
        return emitted

    def _emit_one(self):
        first = self.pending[0]
        reward = 0.0
        last = first
        steps = 0
        for candidate in list(self.pending)[:self.n_step]:
            reward += (self.gamma ** steps) * float(candidate.reward)
            steps += 1
            last = candidate
            if candidate.done:
                break
        self.pending.popleft()
        values = first._asdict()
        values["reward"] = reward
        values["next_state"] = last.next_state
        values["done"] = last.done
        values["next_legal"] = last.next_legal
        if "steps" in values:
            values["steps"] = steps
        return type(first)(**values)

    def state_dict(self) -> dict:
        return {"n_step": self.n_step, "gamma": self.gamma, "pending": list(self.pending)}

    def load_state_dict(self, state: dict | None) -> None:
        self.pending.clear()
        if not state:
            return
        if int(state["n_step"]) != self.n_step or float(state["gamma"]) != self.gamma:
            raise ValueError("n-step accumulator contract changed")
        self.pending.extend(state.get("pending", ()))

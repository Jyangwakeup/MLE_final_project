"""Deterministic hashed tile coding and trace-based control learners."""
from __future__ import annotations
import numpy as np

class TileCoder:
    def __init__(
        self, dimensions, tilings=8, bins=8, memory_size=32768, seed=0,
        signed_indices=(), ternary_indices=(),
    ):
        self.dimensions, self.tilings, self.bins, self.memory_size = int(dimensions), int(tilings), int(bins), int(memory_size)
        rng = np.random.default_rng(seed)
        self.offsets = rng.random((self.tilings, self.dimensions), dtype=np.float32)
        self.primes = rng.integers(1, 2**31-1, size=self.dimensions, dtype=np.int64) | 1
        self.signed_indices = np.asarray(signed_indices, dtype=np.int64)
        self.ternary_indices = np.asarray(ternary_indices, dtype=np.int64)
    def encode(self, state):
        clipped = np.asarray(state, dtype=np.float32).copy()
        clipped[self.ternary_indices] = np.sign(clipped[self.ternary_indices])
        clipped[self.signed_indices] = (clipped[self.signed_indices] + 1.) / 2.
        clipped = np.clip(clipped, 0., 1.)
        coordinates = np.floor(clipped[None, :] * self.bins + self.offsets).astype(np.int64)
        hashes = (coordinates * self.primes).sum(axis=1) + np.arange(self.tilings, dtype=np.int64) * 2654435761
        return np.mod(hashes, self.memory_size).astype(np.int64)
    def state_dict(self): return {"offsets": self.offsets, "primes": self.primes}
    def load_state_dict(self, state):
        if np.asarray(state["offsets"]).shape != self.offsets.shape: raise ValueError("tile coder contract changed")
        self.offsets = np.asarray(state["offsets"], dtype=np.float32); self.primes = np.asarray(state["primes"], dtype=np.int64)

class TraceControl:
    def __init__(self, dimensions, actions, *, seed, hyperparameters, algorithm):
        self.actions, self.algorithm = int(actions), algorithm
        self.gamma, self.alpha, self.lam = hyperparameters["gamma"], hyperparameters["learning_rate"], hyperparameters["lambda"]
        self.final_alpha = float(hyperparameters.get("learning_rate_final", self.alpha))
        self.alpha_decay_steps = int(hyperparameters.get("learning_rate_decay_steps", 1))
        if not 0 < self.final_alpha <= self.alpha or self.alpha_decay_steps < 1:
            raise ValueError("learning-rate decay must satisfy 0 < final <= start and steps >= 1")
        projections = hyperparameters.get("action_feature_indices")
        if projections is None:
            self.action_feature_indices = None
            projected_dimensions = dimensions
            projected_signed = hyperparameters.get("signed_indices", ())
            projected_ternary = hyperparameters.get("ternary_indices", ())
        else:
            if len(projections) != self.actions:
                raise ValueError("one action feature projection is required per action")
            self.action_feature_indices = tuple(
                np.asarray(indices, dtype=np.int64) for indices in projections)
            projected_dimensions = len(self.action_feature_indices[0])
            if projected_dimensions < 1 or any(
                len(indices) != projected_dimensions
                or np.any(indices < 0) or np.any(indices >= dimensions)
                for indices in self.action_feature_indices
            ):
                raise ValueError(
                    "action feature projections must be equal, non-empty, and in range")
            first_projection = self.action_feature_indices[0]
            signed = set(hyperparameters.get("signed_indices", ()))
            ternary = set(hyperparameters.get("ternary_indices", ()))
            projected_signed = tuple(
                index for index, source in enumerate(first_projection)
                if int(source) in signed)
            projected_ternary = tuple(
                index for index, source in enumerate(first_projection)
                if int(source) in ternary)
        self.coder = TileCoder(
            projected_dimensions, hyperparameters["tilings"],
            hyperparameters["bins"], hyperparameters["memory_size"], seed,
            projected_signed, projected_ternary)
        estimators = 2 if algorithm == "double_q_lambda" else 1
        self.weights = np.zeros((estimators, self.actions, hyperparameters["memory_size"]), dtype=np.float32)
        self.traces = np.zeros_like(self.weights); self.updates = 0; self.epsilon = 0.
        self.watkins_trace_cut = bool(hyperparameters.get("watkins_trace_cut", False))
        self._behavior_was_greedy = True
        self.rng = np.random.default_rng(seed)
    def q_values(self, state, estimator=None):
        if self.action_feature_indices is None:
            active = self.coder.encode(state)
            if estimator is None: return self.weights[:, :, active].sum(axis=(0,2)) / self.weights.shape[0]
            return self.weights[estimator][:, active].sum(axis=1)
        values = np.empty(self.actions, dtype=np.float32)
        estimators = range(self.weights.shape[0]) if estimator is None else (estimator,)
        scale = self.weights.shape[0] if estimator is None else 1
        for action, indices in enumerate(self.action_feature_indices):
            active = self.coder.encode(np.asarray(state)[indices])
            values[action] = sum(
                float(self.weights[item, action, active].sum())
                for item in estimators) / scale
        return values
    def set_epsilon(self, epsilon): self.epsilon = float(epsilon)
    def set_behavior_greedy(self, greedy):
        """Record whether the behavior action follows the current greedy policy."""
        self._behavior_was_greedy = bool(greedy)
    def observe(self, transition):
        state = np.asarray(transition.state)
        active = self.coder.encode(
            state if self.action_feature_indices is None
            else state[self.action_feature_indices[transition.action]])
        if self.algorithm == "expected_sarsa_lambda":
            estimator = 0; current = float(self.weights[0, transition.action, active].sum()); target = float(transition.reward)
            if not transition.done:
                values = self.q_values(transition.next_state); legal = np.flatnonzero(transition.next_legal)
                greedy = legal[np.flatnonzero(values[legal] == values[legal].max())]
                probabilities = np.zeros(self.actions); probabilities[legal] = self.epsilon / len(legal); probabilities[greedy] += (1.-self.epsilon)/len(greedy)
                target += (self.gamma ** transition.steps) * float(np.dot(probabilities, values))
        else:
            estimator = int(self.rng.integers(2)); current = float(self.weights[estimator, transition.action, active].sum()); target = float(transition.reward)
            if not transition.done:
                selector = self.q_values(transition.next_state, estimator); legal = np.flatnonzero(transition.next_legal)
                selected = int(legal[np.argmax(selector[legal])])
                target += (self.gamma ** transition.steps) * float(self.q_values(transition.next_state, 1-estimator)[selected])
        delta = target-current
        if self.watkins_trace_cut and not self._behavior_was_greedy:
            self.traces.fill(0.)
        self.traces *= self.gamma * self.lam
        self.traces[estimator, transition.action, active] = 1.
        progress = min(self.updates / self.alpha_decay_steps, 1.0)
        effective_alpha = self.alpha + (self.final_alpha - self.alpha) * progress
        self.weights[estimator] += (effective_alpha / self.coder.tilings) * delta * self.traces[estimator]
        self.updates += 1
        if transition.done: self.traces.fill(0.)
        return abs(delta)
    def retrospective_penalty(self, state, action, reward):
        """Apply direct credit to an earlier action without reviving old traces."""
        state = np.asarray(state)
        projected = (
            state if self.action_feature_indices is None
            else state[self.action_feature_indices[action]])
        active = self.coder.encode(projected)
        estimator = 0 if self.algorithm == "expected_sarsa_lambda" else int(self.rng.integers(2))
        current = float(self.weights[estimator, action, active].sum())
        delta = float(reward) - current
        progress = min(self.updates / self.alpha_decay_steps, 1.0)
        effective_alpha = self.alpha + (self.final_alpha - self.alpha) * progress
        self.weights[estimator, action, active] += (
            effective_alpha / self.coder.tilings) * delta
        self.updates += 1
        return abs(delta)
    def checkpoint(self):
        return {"weights":self.weights, "traces":self.traces, "tile_coder":self.coder.state_dict(), "updates":self.updates, "learner_rng_state":self.rng.bit_generator.state, "behavior_was_greedy":self._behavior_was_greedy}
    def load_checkpoint(self, checkpoint, training=False, **_):
        self.weights = np.asarray(checkpoint["weights"], dtype=np.float32).copy(); self.coder.load_state_dict(checkpoint["tile_coder"]); self.updates=int(checkpoint.get("updates",0))
        self._behavior_was_greedy=bool(checkpoint.get("behavior_was_greedy",True))
        if training:
            self.traces=np.asarray(checkpoint["traces"], dtype=np.float32).copy(); self.rng.bit_generator.state=checkpoint["learner_rng_state"]
        else: self.traces.fill(0.)

__all__ = ["TileCoder", "TraceControl"]

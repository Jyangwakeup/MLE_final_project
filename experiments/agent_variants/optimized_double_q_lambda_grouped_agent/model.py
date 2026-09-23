"""Double Q(lambda) with independent base and history tile groups."""

from __future__ import annotations

import numpy as np

from agent_code.learning_common.tile_coding import TileCoder


class GroupedTraceControl:
    """Sum two independently tiled value components for every action."""

    def __init__(self, dimensions, actions, *, seed, hyperparameters, algorithm):
        if algorithm != "double_q_lambda":
            raise ValueError("grouped control only supports double_q_lambda")
        self.actions = int(actions)
        self.algorithm = algorithm
        self.gamma = float(hyperparameters["gamma"])
        self.alpha = float(hyperparameters["learning_rate"])
        self.final_alpha = float(hyperparameters["learning_rate_final"])
        self.alpha_decay_steps = int(hyperparameters["learning_rate_decay_steps"])
        self.lam = float(hyperparameters["lambda"])
        self.watkins_trace_cut = bool(hyperparameters["watkins_trace_cut"])
        self.group_action_feature_indices = tuple(
            tuple(np.asarray(group, dtype=np.int64) for group in action_groups)
            for action_groups in hyperparameters["group_action_feature_indices"]
        )
        if len(self.group_action_feature_indices) != self.actions:
            raise ValueError("one grouped projection is required per action")
        group_count = len(self.group_action_feature_indices[0])
        if group_count != 2 or any(len(groups) != group_count for groups in self.group_action_feature_indices):
            raise ValueError("exactly two feature groups are required")
        tilings = tuple(int(value) for value in hyperparameters["group_tilings"])
        memories = tuple(int(value) for value in hyperparameters["group_memory_sizes"])
        if len(tilings) != group_count or len(memories) != group_count:
            raise ValueError("group coder contract length mismatch")
        if any(np.any(indices < 0) or np.any(indices >= dimensions)
               for groups in self.group_action_feature_indices for indices in groups):
            raise ValueError("grouped feature index is out of range")

        signed_sources = set(hyperparameters.get("signed_indices", ()))
        ternary_sources = set(hyperparameters.get("ternary_indices", ()))
        self.coders = []
        for group in range(group_count):
            first = self.group_action_feature_indices[0][group]
            if any(len(groups[group]) != len(first) for groups in self.group_action_feature_indices):
                raise ValueError("each action must use equal dimensions within a group")
            signed = tuple(i for i, source in enumerate(first) if int(source) in signed_sources)
            ternary = tuple(i for i, source in enumerate(first) if int(source) in ternary_sources)
            self.coders.append(TileCoder(
                len(first), tilings[group], hyperparameters["bins"], memories[group],
                seed + 104729 * group, signed, ternary))
        self.coders = tuple(self.coders)
        self.total_tilings = sum(coder.tilings for coder in self.coders)
        self.weights = tuple(
            np.zeros((2, self.actions, coder.memory_size), dtype=np.float32)
            for coder in self.coders)
        self.traces = tuple(np.zeros_like(weights) for weights in self.weights)
        self.updates = 0
        self.epsilon = 0.0
        self._behavior_was_greedy = True
        self.rng = np.random.default_rng(seed)

    def active_tiles(self, state, action):
        vector = np.asarray(state)
        return tuple(
            coder.encode(vector[self.group_action_feature_indices[action][group]])
            for group, coder in enumerate(self.coders))

    def q_values(self, state, estimator=None):
        values = np.zeros(self.actions, dtype=np.float32)
        estimators = range(2) if estimator is None else (int(estimator),)
        scale = 2 if estimator is None else 1
        for action in range(self.actions):
            active = self.active_tiles(state, action)
            values[action] = sum(
                float(self.weights[group][item, action, active[group]].sum())
                for group in range(len(self.coders)) for item in estimators
            ) / scale
        return values

    def set_epsilon(self, epsilon):
        self.epsilon = float(epsilon)

    def set_behavior_greedy(self, greedy):
        self._behavior_was_greedy = bool(greedy)

    def observe(self, transition):
        state = np.asarray(transition.state)
        active = self.active_tiles(state, transition.action)
        estimator = int(self.rng.integers(2))
        current = sum(float(self.weights[group][estimator, transition.action, tiles].sum())
                      for group, tiles in enumerate(active))
        target = float(transition.reward)
        if not transition.done:
            selector = self.q_values(transition.next_state, estimator)
            legal = np.flatnonzero(transition.next_legal)
            selected = int(legal[np.argmax(selector[legal])])
            target += (self.gamma ** transition.steps) * float(
                self.q_values(transition.next_state, 1 - estimator)[selected])
        delta = target - current
        if self.watkins_trace_cut and not self._behavior_was_greedy:
            for traces in self.traces:
                traces.fill(0.0)
        progress = min(self.updates / self.alpha_decay_steps, 1.0)
        effective_alpha = self.alpha + (self.final_alpha - self.alpha) * progress
        for group, tiles in enumerate(active):
            self.traces[group][...] *= self.gamma * self.lam
            self.traces[group][estimator, transition.action, tiles] = 1.0
            self.weights[group][estimator] += (
                effective_alpha / self.total_tilings) * delta * self.traces[group][estimator]
        self.updates += 1
        if transition.done:
            for traces in self.traces:
                traces.fill(0.0)
        return abs(delta)

    def retrospective_penalty(self, state, action, reward):
        active = self.active_tiles(state, action)
        estimator = int(self.rng.integers(2))
        current = sum(float(self.weights[group][estimator, action, tiles].sum())
                      for group, tiles in enumerate(active))
        delta = float(reward) - current
        progress = min(self.updates / self.alpha_decay_steps, 1.0)
        effective_alpha = self.alpha + (self.final_alpha - self.alpha) * progress
        for group, tiles in enumerate(active):
            self.weights[group][estimator, action, tiles] += (
                effective_alpha / self.total_tilings) * delta
        self.updates += 1
        return abs(delta)

    def checkpoint(self):
        weights = list(self.weights)
        traces = list(self.traces)
        coders = [coder.state_dict() for coder in self.coders]
        return {
            # The generic resume validator requires the three legacy keys.
            # They alias the same grouped objects in the pickle payload.
            "weights": weights, "traces": traces,
            "tile_coder": {"grouped_tile_coders": coders},
            "grouped_weights": weights,
            "grouped_traces": traces,
            "grouped_tile_coders": coders,
            "updates": self.updates,
            "learner_rng_state": self.rng.bit_generator.state,
            "behavior_was_greedy": self._behavior_was_greedy,
        }

    def load_checkpoint(self, checkpoint, training=False, **_):
        weights = checkpoint.get("grouped_weights")
        coders = checkpoint.get("grouped_tile_coders")
        if weights is None or coders is None or len(weights) != 2 or len(coders) != 2:
            raise ValueError("checkpoint does not contain grouped tile-coding state")
        for group in range(2):
            restored = np.asarray(weights[group], dtype=np.float32)
            if restored.shape != self.weights[group].shape:
                raise ValueError("grouped checkpoint weight shape changed")
            self.weights[group][...] = restored
            self.coders[group].load_state_dict(coders[group])
        self.updates = int(checkpoint.get("updates", 0))
        self._behavior_was_greedy = bool(checkpoint.get("behavior_was_greedy", True))
        if training:
            traces = checkpoint.get("grouped_traces")
            if traces is None or len(traces) != 2:
                raise ValueError("training checkpoint has no grouped traces")
            for group in range(2):
                restored = np.asarray(traces[group], dtype=np.float32)
                if restored.shape != self.traces[group].shape:
                    raise ValueError("grouped checkpoint trace shape changed")
                self.traces[group][...] = restored
            self.rng.bit_generator.state = checkpoint["learner_rng_state"]
        else:
            for traces in self.traces:
                traces.fill(0.0)

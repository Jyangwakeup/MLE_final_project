---
status: accepted
---

# Use a veto-only survival mask

The Agent applies one versioned survival mask to exploration, learned greedy behavior, frozen inference, and Double DQN bootstrap targets. The mask removes only physical legal actions proven unable to survive the prediction horizon; the learned Q-function still ranks every remaining action, and an empty safe set falls back to the learned physical-action argmax. This boundary reduces avoidable deaths without embedding a rule-selected “best” action, preserving the course requirement that decisions are learned while accepting conservative errors from the finite-horizon dynamics model.

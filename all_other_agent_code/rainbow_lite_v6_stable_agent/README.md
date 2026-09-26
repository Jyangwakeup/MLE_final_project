# Rainbow-lite V6 stable continuation

This experiment-only agent preserves the continuous-v6 opponent-tracking
network while reducing continuation drift: learning rate `1e-4`, one learner
update per four observations, a 100,000-transition prioritized replay, and a
10,000-transition protected quota for returns of at least 4.0.

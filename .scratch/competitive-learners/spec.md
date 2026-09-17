# Competitive learners

Status: ready-for-agent

Implement three independently trainable Task 2 candidates under the existing
`continuous-v2`, reward, exploration, and survival-mask contracts:

1. `rainbow_lite_agent`: Dueling Double DQN with 3-step returns and prioritized replay.
2. `expected_sarsa_lambda_agent`: linear tile coding with Expected SARSA(lambda).
3. `double_q_lambda_agent`: a Q-learning improvement using two linear tile-coded
   estimators and estimator-local eligibility traces.

All agents must expose the framework callbacks, carry versioned hyperparameter and
feature contracts in exact-resume checkpoints, work through `experiments/run.py`,
and remain packageable without framework modifications. Tests cover learner math,
checkpoint round trips, contracts, and callback smoke behavior.

## Fair comparison contract

- Feature: `continuous-v2`
- Default reward for formal Task 2 runs: `r7_safe_credit_sparse`
- Safety: explicitly configure `survival-mask-v1/all`, horizon 7
- Evaluation: use the same seeds and role-specific metrics as the Task 2 winner
- The reserved final-test seeds `20000..20099` remain unused until selection

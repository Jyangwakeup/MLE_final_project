# Double Q(lambda) Agent

Archived source: the runner and submission builder do not load this agent from
`all_other_agent_code/`. Its experiment contract remains available for
historical metadata resolution.

An improvement over the tabular Q-learning baseline: two independently updated
tile-coded estimators reduce maximization bias, while estimator-local eligibility
traces propagate delayed bomb and collection rewards.

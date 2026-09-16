# No-safety causal bomb credit

Improve the Expected SARSA(lambda) Task 2 ablation without enabling a survival
mask or removing any physical legal action. A self-kill must assign additional
credit to the earlier bomb-placement state, while crate credit is delayed until
the bomb has resolved and the agent can place another bomb.

The change uses a new reward identity so existing r10 evidence remains frozen.

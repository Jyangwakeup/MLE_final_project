---
status: accepted
---

# Preserve redundant routes during the own-bomb cycle

Task 3 seed 22 died to its own bomb in seven of twenty development games even
though the existing H=7 mask rejected immediately fatal actions. The failures
appeared after the bomb was placed: learned choices progressively consumed
escape space until no safe action remained.

We add `survival-mask-v3` around the unchanged 84-dimensional
`continuous-v2` Double DQN and unchanged `r7_safe_credit_sparse` reward. Before
placing a bomb, the mask vetoes `BOMB` when a safe non-bomb alternative exists
and the hypothetical bomb leaves fewer than two internally vertex-disjoint
time-expanded escape paths. While the own bomb or its flame remains, the mask
keeps only v1-safe actions that preserve two routes when such actions exist.
It falls back first to the complete v1 set and only then to physical-action Q
selection.

This is deliberately a veto, not an action recommendation: learned Q-values
continue to rank the remaining actions. Holding features and rewards fixed
makes the frozen A/B test identify the effect of escape redundancy. A
specialized replay mix is preregistered as the only fallback and may run only
if frozen v3 passes but newly trained models still fail a safety gate.

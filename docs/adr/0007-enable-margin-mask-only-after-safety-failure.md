---
status: accepted
---

# Enable survival-mask-v2 only after a preregistered safety failure

The default remains survival-mask-v1. If the highest-ranked first-round arm exceeds 5% suicide on either training seed, v2 may reject actions whose H=7 escape area is below 75% of the best v1-safe action while the agent still owes escape from its own bomb. Empty refinement falls back to the complete v1 set; only an empty v1 set falls back to physical Q. This preserves the learner's ranking among allowed actions and avoids introducing an untested shield into every arm.

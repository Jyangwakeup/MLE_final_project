---
status: accepted
---

# Enumerate opponent transitions for own-bomb safety

The two-route mask reduced seven known Task 3 self-deaths to one, but the
remaining failure placed a bomb immediately before an opponent/environment
transition removed every H=7 escape. Treating each opponent's current tile as
a permanent obstacle did not represent that transition.

During bomb placement and the own-bomb escape obligation, we enumerate every
joint physical opponent action, opponent bomb placement, and official action
execution order. An action passes the new boundary only when every distinct
result keeps at least one H=7 continuation. The previous two-route boundary
remains the nominal test, and fallback is v4 to v3 to v1 to physical Q.

We rejected a learned opponent predictor because it would provide no safety
claim for unseen opponents. We also rejected blocking the union of every
possible future opponent tile because it combines mutually exclusive worlds
and can suppress useful bombs. Exhaustive one-step scenarios are more costly,
so the implementation deduplicates equal results and must still pass the
50 ms P95 engineering gate.

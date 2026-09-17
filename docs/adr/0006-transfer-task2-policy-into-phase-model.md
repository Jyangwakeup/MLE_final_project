---
status: accepted
---

# Treat Task 2 to phase-model initialization as transfer

The Task 2 parents use 84 inputs and v7 state; Task 3 uses 117 inputs and v8. The explicit transfer copies policy and target, zeros all 33 new input columns, and resets optimizer and replay. A frozen, independently collected Task 1/2 teacher dataset protects old behavior. Ordinary resume rejects this boundary because claiming exact continuation would hide the changed feature, reward, optimizer, and replay contracts.

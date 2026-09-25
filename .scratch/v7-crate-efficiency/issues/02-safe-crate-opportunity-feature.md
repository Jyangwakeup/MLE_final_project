# Add a safe crate-opportunity interaction feature

Type: prototype
Status: claimed

Create a new feature contract derived from V7 that adds an observable scalar
for the conjunction of physical BOMB legality, horizon survivability, and
crate blast utility. It must remain a factual input rather than an action rule,
and the survival mask must remain veto-only.

Blocked by: 01

## Acceptance criteria

- A versioned feature ID and schema preserve all V7 fields in order.
- The new scalar is nonzero only when BOMB is physically legal,
  horizon-survivable, and would hit at least one crate.
- Migration zero-extends the V7 policy without changing its initial outputs.
- Unit tests cover feature semantics and migration equivalence.
- Frozen C400 remains the baseline for the subsequent controlled training run.

## Comments

# Domain docs

Rules for engineering skills that need this repository's domain documentation.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- If `CONTEXT-MAP.md` exists, follow it to the relevant context-specific `CONTEXT.md` files instead.
- Read relevant decisions in `docs/adr/` before changing the affected area.
- If these files do not exist, proceed silently. Domain-modeling work creates them only when terminology or decisions need to be recorded.

## Layout

This is a single-context repository. Domain documentation, when needed, lives at:

```text
CONTEXT.md
docs/adr/
```

## Vocabulary and decisions

Use terminology defined in `CONTEXT.md` for issue titles, proposals, hypotheses, and tests. If a proposed change conflicts with an existing ADR, surface the conflict explicitly rather than silently overriding the decision.

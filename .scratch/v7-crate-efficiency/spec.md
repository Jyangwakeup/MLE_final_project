# V7 crate-efficiency diagnosis

Status: active

## Goal

Determine whether the frozen Rainbow-lite V7 C400 checkpoint loses crate-clearing
performance because it fails to reach crate frontiers or because it declines
horizon-survivable, crate-hitting bomb opportunities.

## Baseline

Use the existing fixed ten-seed Task 4 evaluations for C100-C400. Treat game
score, coins, kills, deaths, and first-place rate as outcome metrics. Treat
crates per bomb, zero-utility bomb rate, bomb rate, wait rate, and the new
crate-opportunity diagnostics as behavioral evidence only.

## First experiment

Add read-only evaluation diagnostics for states where a bomb is physically
legal, hits at least one crate, and is horizon-survivable. Measure how often
the frozen policy selects BOMB in those states. Do not change policy inputs,
action masks, rewards, weights, or RNG use.

## Decision rule

- Low opportunity frequency: investigate crate-frontier navigation features.
- High opportunity frequency with low BOMB selection: investigate explicit
  safe crate-opportunity features or training credit.
- High selection and low crate yield: investigate bomb placement utility.

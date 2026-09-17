# Bound Task 3 optimization before independent confirmation

Accepted. The failed plateau confirmation remains failed. The lifecycle repair
campaign first tests the corrected original package, then permits at most two
independent changes (fixed 16/32/16 Replay and kill reward 5→15) and their qualified
combination. The first package whose three training seeds pass development enters
independent confirmation; confirmation or main failure ends the campaign without
trying another candidate. All gates and the original plateau rule remain fixed.

The registered 60/100/100 development/confirmation/main worlds are disjoint and
audited against all worktrees. Used confirmation worlds are not blind validation
again. The 24-hour budget starts with implementation; new arms stop at hour18 and
new evaluation batches at hour23. Timeout is incomplete evidence, not success.
See experiments/task3_lifecycle_campaign.json for the exact frozen contract.

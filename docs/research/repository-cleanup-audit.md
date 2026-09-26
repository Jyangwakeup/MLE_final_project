# Repository Cleanup Audit

Audit date: 2026-09-24. Baseline commit:
`da6df5aee2e004d4c5bc51fd8674fc6b812370b6`.

## Decision

This cleanup improves the public review path without rewriting research
history. The selected Agent, tracked experiment evidence, recovery archive, and
runtime runs remain in place. Only generated `output/` and three completed
one-off `.scratch/` work areas enter reversible local quarantine.

## Baseline inventory

The inventory reports logical bytes separately from allocated bytes because
many experiment artifacts are sparse or linked. Counts include regular files
and symbolic links but not directories.

| Path | Files | Logical size | Allocated size | Classification | Action |
|---|---:|---:|---:|---|---|
| `agent_code/die_hardest` | 53 | 55.8 MiB | 11.1 MiB | Active surface | Freeze |
| `experiments/results` | 16,946 | 277.4 MiB | 110.0 MiB | Versioned experiment evidence | Freeze paths and content |
| `archive` | 261,591 | 74.0 GiB | 9.5 GiB | Recovery archive | Preserve in place |
| `runs` | 32,762 | 17.6 GiB | 2.5 GiB | Runtime artifacts with path coupling | Inventory only |
| `output` | 1,107 | 90.9 MiB | 37.3 MiB | Generated output | Quarantine |
| `.scratch/archive-consolidation` | 24 | 21.6 KiB | 35.0 KiB | Completed one-off work | Quarantine |
| `.scratch/branch-integration` | 361 | 266.2 MiB | 47.1 MiB | Completed one-off work | Quarantine |
| `.scratch/die-hardest-submission` | 501 | 117.5 MiB | 24.1 MiB | Completed one-off work | Quarantine |

The Git object database occupied approximately 89 MiB before cleanup. This
confirms that the main issue is navigation and local artifact lifecycle, not
repository-history size.

## Risk findings

- Experiment controllers and analyses directly reference `runs/<run-id>`, so
  moving `runs/` before the report deadline would break active and historical
  workflows.
- Manifests and tests directly reference `experiments/results/`; moving only
  selected result files would invalidate their recorded paths and hashes.
- `archive/` is the sole local recovery location for parts of the consolidated
  worktree history. It has stronger retention requirements than quarantine.
- Formal reports already preserve the conclusions from the three completed
  scratch areas. Their detailed local diagnostics remain recoverable through a
  quarantine manifest rather than staying in the active issue workspace.

## Quarantine controls

The maintenance tool accepts only the four approved paths, refuses tracked
content and destination conflicts, detects active users and source changes,
records file hashes and symbolic links, and supports verified restoration. It
does not expose deletion or purge behavior.

## Result

The first quarantine operation completed with manifest
`.local-artifacts/manifests/20260924T162239Z-da6df5ae.json`. It moved 1,993
files without deletion. A restore preview verified all four original paths;
restoration was deliberately not executed.

The post-operation inventory reports `output/` and the three completed scratch
areas as absent from the active workspace. `runs/`, `archive/`,
`experiments/results/`, and `agent_code/die_hardest/` retained exactly their
baseline file counts and byte totals. The selected checkpoint still has
SHA-256
`ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a`.

Twenty-two quarantine and archive-boundary tests pass, including a repeated
quarantine no-op and restore preview. The complete suite in the
declared Python 3.10 environment ran 599 tests: 589 passed, five skipped, and
five existing historical tests errored because their frozen manifests contain
absolute paths to worktrees removed by the earlier archive consolidation. All
five missing paths are under `MLE_final_project_task4_exploration` or
`MLE_final_project_task4_frozen`; none points to a quarantined target. Updating
those preserved historical contracts is outside this low-risk cleanup.

One no-GUI round with `die_hardest` and three rule-based opponents completed
successfully. The ALSA warnings emitted on the headless host are unrelated to
Agent execution.

# Repository Layout

This guide is the shortest path through the project. It separates the files a
reviewer normally needs from research evidence and machine-local artifacts.

## Active surface

- `README.md` introduces the project and selected competition Agent.
- `agent_code/die_hardest/` is the frozen competition Agent. Its own README
  documents usage, provenance, dependencies, and known limitations.
- `main.py` and the root framework modules run the Bomberman environment.
- `experiments/` contains experiment orchestration, registered configurations,
  and the internal experiment guide.
- `tests/` contains the regression suite.
- `docs/research/` contains curated reports and experiment indexes.

Start with the root quick-start command when checking that the selected Agent
runs. Use the experiment guide only when reproducing or extending research.

## Versioned experiment evidence

`experiments/results/` contains tracked evidence referenced by manifests,
tests, and research reports. It includes unsuccessful experiments because the
course assessment requires the development history and model-selection basis
to remain visible. Existing paths in this tree are frozen; it is not a cache or
a general output directory.

The Agent directories other than `die_hardest` preserve developed models and
baselines. The [Q-learning and CNN index](../agent_code/Q_CNN_AGENT_INDEX.md)
and [Rainbow Lite index](research/rainbow-lite-index.md) provide narrower entry
points than browsing every variant.

## Recovery archive

`archive/` contains machine-local recovery material from consolidated historical
worktrees. Only its guide is tracked by Git; an ordinary clone does not contain
the full archive. The archive has its own registries, hashes, and recovery
procedure and must not be treated as disposable generated output. See the
[archive guide](../archive/README.md).

## Runtime and quarantined artifacts

`runs/` is the fixed runtime location used by experiment controllers and
historical analysis. It is ignored by Git and remains in place until a separate
post-report retention review.

`output/` and completed one-off work under `.scratch/` are generated local
artifacts. After verification they can be moved into the ignored
`.local-artifacts/` quarantine by
`scripts/quarantine_local_artifacts.py`. Quarantine is reversible and is not a
permanent archive. The tool deliberately provides no deletion command.

Use these read-only and reversible interfaces from the repository root:

```bash
python scripts/quarantine_local_artifacts.py audit
python scripts/quarantine_local_artifacts.py quarantine
python scripts/quarantine_local_artifacts.py quarantine --execute
python scripts/quarantine_local_artifacts.py restore --manifest <manifest>
```

Both mutating commands require `--execute`; without it they only preview the
operation. Permanent deletion requires a separate, explicit review.

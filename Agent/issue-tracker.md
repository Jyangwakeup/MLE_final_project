# Issue tracker: Local Markdown

Issues and specs for this repository live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The specification is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`; do not combine tickets into one file.
- Record a triage state on a `Status:` line near the top of each issue file. See `triage-labels.md` for the role strings.
- Append comments and discussion under a `## Comments` heading at the bottom of the file.

## Publishing and reading tickets

When a skill says to publish to the issue tracker, create a file under `.scratch/<feature-slug>/`, creating the directory as needed.

When a skill says to fetch a ticket, read the referenced file path. The user will normally provide the path or issue number.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`; it contains notes, decisions so far, and open questions.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with a `Type:` line (`research`, `prototype`, `grilling`, or `task`) and a `Status:` line (`claimed` or `resolved`).
- Blocking: record dependencies as `Blocked by: NN, NN`; a ticket is unblocked when every listed ticket is resolved.
- Frontier: select the lowest-numbered open, unblocked, unclaimed ticket.
- Claim: set `Status: claimed` before starting work.
- Resolve: add an `## Answer` section, set `Status: resolved`, and add a context pointer to the map's decisions section.

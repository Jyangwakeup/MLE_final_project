# Historical Rainbow Lite scripts

These scripts launched or monitored completed Rainbow Lite experiments. They
are retained for research auditability, not as current entry points:

- `train/` contains one-off training and chained evaluation launchers;
- `evaluate/` contains frozen checkpoint evaluation launchers;
- `monitor/` contains dashboards and polling helpers tied to historical run IDs.

Run them only from a checkout with the expected ignored `runs/` data and local
Python environment. For new work, prefer `python -m experiments.run --config
<path>` and record a new experiment contract instead of editing these files.

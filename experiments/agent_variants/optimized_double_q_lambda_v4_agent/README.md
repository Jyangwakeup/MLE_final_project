# Optimized Double Q(lambda) v4 agent

This isolated candidate keeps the validated Watkins Double Q(lambda) learner
and replaces the 84-value `continuous-v2` input with the 126-value
`continuous-v4` history representation.  Long-running results and checkpoint
hashes are recorded in `EXPERIMENT_LOG.md`.

The tournament runtime loads `final.pkl` relative to this directory.  It does
not require the experiment runner or a training run directory.

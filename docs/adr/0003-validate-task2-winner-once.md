---
status: accepted
---

# Validate the Task 2 winner once

A Task 2 configuration must pass the development gates independently for training seeds 11, 22, and 33 and in their 60-round aggregate before one checkpoint is selected by the preregistered ranking. Only that checkpoint receives one 100-seed main validation; failure does not permit testing the runner-up or further tuning. This avoids adapting the selection to the validation set at the cost of rejecting the whole family when one replication is unstable.

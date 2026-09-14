---
status: accepted
---

# Use frozen score for Task 1 convergence

Task 1 stops on three consecutive exploration-free 20-seed assessments with mean score at least 48, not on an action-count target or shaped training reward. Assessments begin after 200 cumulative rounds and repeat every 50 newly trained rounds, with a 1000-round lineage cap. Monitoring uses seeds 9000–9019; promotion uses the independent seeds 10000–10019. Action counts remain part of exploration scheduling and diagnostics. This costs periodic evaluation time, but measures the learned policy directly and avoids treating efficient early completion as insufficient training.

# Keep the frozen competition Agent on the active surface

The selected Die Hardest submission is a frozen competition Agent and its
complete runtime package lives at `agent_code/die_hardest/`. Keep callbacks,
vendored dependencies, the submission manifest, and `final.pt` together and
versioned there so the official loader can run the selected candidate from the
documented active directory. The frozen file manifest defines the expected
package contents and checkpoint identity.

`all_other_agent_code/die_hardest/` is not a second active copy. Historical
evaluation records and recovery material remain in their evidence and archive
locations; moving the runnable package does not turn those records into a new
performance evaluation. Future references to the runnable candidate should
point to `agent_code/die_hardest/`, while historical claims continue to cite
their original experiment evidence.

This follows the course submission contract, which requires the trained
parameters inside `agent_code/<agent_name>/`, and the repository convention
that documented active paths identify the source of truth for current runtime
work.

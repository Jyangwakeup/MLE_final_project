# Separate active code, evidence, recovery archives, and quarantine

The repository keeps its reviewer-facing active surface small without moving
versioned experiment evidence or treating the local recovery archive as a
cache. Generated one-off artifacts may enter a separate ignored quarantine
only through a verified, reversible manifest; quarantine never counts as
archival preservation and is not automatically deleted. This preserves stable
research references before the report deadline while making the intended entry
points explicit.

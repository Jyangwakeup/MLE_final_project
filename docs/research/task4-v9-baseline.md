# Task4 with a recertified v9 parent

This protocol supersedes the v5-reference A/B campaign only for the new task4-campaign-v4 run. Existing records and contracts remain valid descriptions of their own experiments. No Task3 retraining occurs.

The original Task3 checkpoint is evaluated with a separate v9 reference config; child training uses explicit migration and only B. The parent and children share strict100/300ms act and400ms search limits. Evaluation proceeds: historical paired retention identity audit (or repeat), new100-world Task1–4 safety admission, three20-round training diagnostics,60-world development,100-world three-seed confirmation,100-world unique-candidate main validation. Old candidate-only evaluation must finish first; it cannot select new checkpoints.

Use `python experiments/task4_background.py --manifest experiments/task4_v9_reference_campaign.json` from this worktree with the registered environment. Run IDs include manifest and source identity. Status, PID, logs and commands live under the manifest campaign's runs directory. Only same-identity infrastructure interruptions may use `--resume`; terminal failures cannot resume. The runtime is unchanged; source/config/test/evidence hashes are frozen before launch. Budget begins at implementation, not launch.

Raw Replay limitations, world allocation, checkpoint hashes, pairing intervals and actual completed stages belong in the final local report. Only a passed main validation sets Task4 qualified.

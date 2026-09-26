# Die Hardest

Frozen competition candidate: Task4 B seed33/c200, Double DQN, survival-mask-v9.
Tournament registration name: **Die Hardest**. Module and default framework display name: `die_hardest`.

Install `requirements.txt` in the course environment. Copy this entire directory into the original framework's `agent_code/`, then run from the framework root:

```bash
python main.py play --agents die_hardest rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --no-gui
```

Inference loads the included `final.pt` relative to this directory, uses CPU and one Torch thread, and needs no sibling project agents. Run without `--train`; the training module is retained from the verified source, but this frozen copy is intended for inference. For the verified default configuration, do not set `BOMBERMAN_*` overrides.

The original B33 checkpoint is unchanged. Renaming updates only package references and documentation. The algorithm, feature, reward and checkpoint contract identifiers are unchanged. Requirements retain the validated NumPy 2.2.6 and Torch 2.5.1 versions. See SUBMISSION_MANIFEST.json for provenance and hashes.

The original B33 completed a separate 1000-world evaluation versus three rule-based opponents: mean score 4.231, first including ties 50.2%, sole first 37.8%, survival 94.7%, no observed timeouts or safety alerts. These are results of the source candidate, not a new 1000-world evaluation after renaming.

Known limitation: the historical 27155 safety-contract counterexample remains unresolved. Packaging/renaming checks are not full safety certification. Docker and official tournament hardware were not verified here. No isolated repair prototype is included.

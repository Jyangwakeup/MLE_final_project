# Fix Rainbow causal survival credit and train

Type: task
Status: claimed

Implement a new reward identity and no-safety Rainbow configuration, connect
Rainbow to versioned temporal rewards, add retrospective bomb credit, test the
contracts, and train/evaluate against `experiments/task2_quality_gate.json`.

## Comments

- Frozen inference and all training decisions must keep safety mode `off`.
- Do not modify or resume existing r10/r11 evidence.
- The fixed r250 r13 evaluation reached 0% suicide and 100% bomb survival,
  but 89.68% of bombs had zero utility. Its unconditional +3 resolved-bomb
  reward was therefore rejected. r14 clears causal bomb blame on successful
  resolution and grants positive retrospective credit only to useful bombs.
- r14 improved the fixed 5-seed checkpoint from 2.4 coins/49.6 crates/20%
  suicide at round 55 to 3.6/54.0/0% at round 100, then regressed at round 150.
  r15 starts from r100, applies direct retrospective penalties to bombs that
  resolve with zero destroyed crates, and raises reachable-coin priority.

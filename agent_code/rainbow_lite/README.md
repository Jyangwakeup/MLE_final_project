# Rainbow Lite

This short name exposes the Rainbow Lite V5/R18 full-retraining chain as
`agent_code/rainbow_lite`.

- Default checkpoint: `final.pt` (Task 4 c500)
- Feature contract: `continuous-v5` (140 inputs)
- Reward contract: `r18_wait_attractor_escape`
- Safety contract: `survival-mask-v5`

The retained full-retraining checkpoints are:

| Task | Checkpoint | Source run |
|---|---|---|
| Task 1 | `checkpoints/task1/final.pt` | `final_rainbow_v5_r18_maskv5_s11_t1_c0500` |
| Task 2 | `checkpoints/task2/final.pt` | `final_rainbow_v5_r18_maskv5_s11_t2_c0800` |
| Task 3 | `checkpoints/task3/final.pt` | `final_rainbow_v5_r18_maskv5_s11_t3_from_t2c0800_c0300` |
| Task 4 | `checkpoints/task4/final.pt` | `final_rainbow_v5_r18_maskv5_s11_t4_from_t3c0300_c0500` |

Task 2's mean training reward over its final 100 rounds was
`61.91074489196847`. The root `final.pt` is byte-identical to the Task 4
checkpoint, which is the paper-selected final full-retraining candidate.

Training reward is not a frozen evaluation score and must not be presented
as promotion evidence. This directory is independently runnable: callback,
model, feature and training implementations are local, while supporting
feature, reward and safety utilities are bundled under `_vendor/`. It can be
copied by itself into another Bomberman framework's `agent_code/` directory.

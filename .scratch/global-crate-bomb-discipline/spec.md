# Global crate navigation and bomb discipline

Add a checkpoint-incompatible `continuous-v5` contract rather than mutating
continuous-v4. It exposes directional global crate density and a bounded
pre-bomb crate-frontier objective so learned behavior can resume navigation
after mandatory escape. `r17_global_crate_bomb_discipline` penalizes
zero-utility placement immediately and again when the owned bomb resolves
without crate utility. The survival mask remains veto-only per ADR 0001.

Legacy r11-r15 IDs remain readable for checkpoint reproducibility. New
no-safety experiments use the frozen `r16_no_safety_locked` copy of r15.

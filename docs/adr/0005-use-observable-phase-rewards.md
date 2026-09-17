---
status: accepted
---

# Use observable, smoothly mixed match phases

Task 3 uses situation progress computed from crate depletion, opponent depletion, and round progress. Continuous triangular weights mix early resource, middle combat, and late mobility terms. This avoids a brittle turn-number switch and exposes facts rather than a recommended action. Directly scaling coin, crate, and own-kill events changes the learned objective, so every choice is made from exploration-free official game metrics, never training reward. Geometric edge occupancy remains diagnostic because an edge tile is not inherently unsafe.

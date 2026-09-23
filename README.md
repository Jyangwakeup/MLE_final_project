# Bomberman Reinforcement Learning

Final project for **Machine Learning Essentials**, **Summer Semester 2026 (SS 2026)**, **Heidelberg University**.

This project develops reinforcement learning agents for Bomberman. The experiments progress from coin collection and navigation to crate destruction, interaction with opponents, and full competitive matches. Agents must balance collecting points with surviving their own bombs and those placed by other players.

The repository contains Q-learning, Double Q(λ), Double DQN, and CNN-based experiments. Across these approaches, we investigate state representations, reward shaping, safety constraints on available actions, and transfer between tasks. Frozen evaluations separate learned policy performance from training rewards, while experiment records preserve unsuccessful attempts and known limitations alongside improvements.

## Selected Competition Agent: Die Hardest

**Die Hardest** is the currently selected and frozen competition model, originating from **Task4 B33 seed33/c200**. It uses Double DQN with an 84-dimensional feature representation and safety action filtering. Its framework loading name is `die_hardest`; inference loads the included checkpoint on CPU using one Torch thread.

The original B33 candidate was evaluated in 1,000 local random worlds in the `classic` scenario against three rule-based agents:

| Metric | Result |
|---|---:|
| Games evaluated | 1,000 |
| Opponents per game | Three rule-based agents |
| Mean official score | 4.231 |
| First place, including ties | 50.2% |
| Sole first place | 37.8% |
| Survival rate | 94.7% |
| Observed framework timeouts | 0 |

Official score is **coins collected + 5 × opponents killed**. These are local benchmark results, not official tournament results. They come from B33 before renaming; Die Hardest subsequently passed behavioral equivalence checks covering 180 historical observations and paired matches. The selection does not establish superiority over every model in this repository under a single common evaluation protocol.

The historical **27155 safety counterexample remains unresolved**. Zero observed timeouts or safety alerts in this benchmark do not provide a complete safety guarantee. See the [Agent README](agent_code/die_hardest/README.md) for the benchmark summary and the [packaging and equivalence report](docs/research/die-hardest-submission/REPORT.md) for provenance, checks, and limitations.

## Quick Start

Use Python **3.10**, matching the validated environment. From the repository root, install the Agent's declared dependencies and run one match:

```bash
python -m pip install -r agent_code/die_hardest/requirements.txt
python main.py play --agents die_hardest rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --no-gui
```

This uses the existing weights for inference with the default model configuration. Do not add `--train` or set `BOMBERMAN_*` configuration overrides when reproducing the frozen Agent setup. The dependency file pins NumPy 2.2.6 and Torch 2.5.1.

For an optional GUI, install `pygame` and remove `--no-gui` in an environment with a graphical display.

## Repository Guide

- [Die Hardest](agent_code/die_hardest/README.md): frozen Agent, usage, provenance, and limitations.
- [Experiment runner](experiments/README.md): configuration and execution of research workflows.
- [Task3 experiment index](docs/research/task3-experiment-index.md): historical safety experiments, outcomes, and source references.
- [Q-learning and CNN model index](agent_code/Q_CNN_AGENT_INDEX.md): model variants and their experiment records.
- [Archive and recovery guide](archive/README.md): artifact locations and instructions for recovering historical experiments.

Large raw logs and some experimental artifacts are stored only in the local archive. An ordinary clone does not contain the complete archive; the linked documentation distinguishes public repository records from local artifact locations.

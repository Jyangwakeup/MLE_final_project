# Bomberman Learning Context

This context defines the project-specific language used to separate game legality, predicted survival, and learned action choice.

## Language

**Physical legal action**:
An action the current game state permits under collision, occupancy, bomb availability, and curriculum rules. Physical legality makes no claim about future survival.
_Avoid_: Valid move, safe move

**Horizon-survivable action**:
A physical legal action for which at least one continuation remains alive through the complete configured prediction horizon.
_Avoid_: Good action, optimal action

**Avoidable fatal action**:
A physical legal action that is not horizon-survivable in a state containing at least one horizon-survivable alternative.
_Avoid_: Bad move, unsafe action

**Survival mask**:
The subset of physical legal actions that are horizon-survivable. It is a veto boundary, not a ranking or recommendation among the remaining actions.
_Avoid_: Safety policy, rule-based controller

**Safety intervention**:
A decision where the learner's highest-valued physical legal action is vetoed by the survival mask and another learned action is selected.
_Avoid_: Rule action, corrected action

**No-safe-action fallback**:
A decision state with no horizon-survivable physical legal action, in which the learner selects its highest-valued physical legal action without a safety veto.
_Avoid_: Random fallback, rescue action

**Coin capability assessment**:
A frozen, exploration-free Task 1 evaluation over the dedicated convergence seeds. Its primary result is mean game score; Task 1 requires every episode's score to equal its collected-coin count.
_Avoid_: Training reward, recent training score

**Task 1 score convergence**:
Three consecutive coin capability assessments whose mean score is at least 48, with assessments separated by 50 newly trained rounds.
_Avoid_: Action budget reached, training completed

**Curriculum promotion qualification**:
Task 1 score convergence followed by a passing, independent stage-gate evaluation and all engineering checks. Convergence alone does not authorize promotion.
_Avoid_: Converged, completed

**Task 2 winner**:
The single Task 2 checkpoint whose configuration passed independent development gates for all three training seeds and then passed the one permitted main validation. A development-selected checkpoint is only a Task 2 winner candidate.
_Avoid_: Best training run, winner configuration, winner candidate

**Task 3 parent baseline**:
A frozen evaluation of one training seed's direct Task 2 parent on the Task 1, Task 2, and Task 3 development worlds before Task 3 learning starts.
_Avoid_: Global baseline, published winner baseline

**Task 3 pilot checkpoint**:
A Task 3 checkpoint produced at a preregistered 500-round assessment boundary. It remains experimental until its training seed passes every capability, retention, safety, and engineering gate.
_Avoid_: Converged model, Task 3 winner

**Task 3 pilot candidate**:
The single development-selected checkpoint from a Task 3 configuration that passed every gate independently for all three training seeds. It is not qualified for Task 4 until a later independent main validation passes.
_Avoid_: Task 3 winner, Task 4 parent

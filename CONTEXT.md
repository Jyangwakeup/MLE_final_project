# Bomberman Learning Context

This context defines the project-specific language used to separate game legality, predicted survival, and learned action choice.

**Decision history snapshot**:
The immutable action history and admitted action set associated with one actual
decision. Reconstructing its learning transition cannot advance or rewind the
Agent's live history.
_Avoid_: Current history for old states, mutable replay history

**Task 3 qualified model**:
A frozen model from a package whose three training seeds passed independent
confirmation and whose single selected candidate passed main validation under
the preregistered capability, retention, safety, and engineering gates.
_Avoid_: Best development checkpoint, guaranteed strong opponent

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

**Own-bomb escape obligation**:
The interval from selecting `BOMB` until that bomb is gone and every tile in
its corresponding flame has cleared. During this interval the Agent must keep
the strongest available escape redundancy; the obligation resets at a round
boundary.
_Avoid_: Bomb timer only, permanent caution mode

**Robust-survivable action**:
A horizon-survivable action with at least two internally vertex-disjoint paths
through the time-expanded safety graph. It is a veto criterion, never a
recommended movement direction.
_Avoid_: Best escape action, scripted route

**Avoidable escape collapse**:
A transition into post-bomb no-safe-action fallback after an earlier decision
in the same own-bomb cycle had at least one safe alternative. It is counted
once per bomb cycle.
_Avoid_: Every death, unavoidable trap

**Opponent transition scenario**:
One possible next game state produced when every Agent chooses from its
physical legal actions and the environment applies one permitted execution
order, followed by the official world update. It includes resulting positions,
new bombs, explosions, destroyed crates, and deaths.
_Avoid_: Predicted opponent move, most likely move

**Controllable survival set**:
The actions for which a feedback strategy remains alive throughout the current
own-bomb danger interval under every modeled opponent transition. Later actions
may depend on the newly observed state; this is not one fixed route.
_Avoid_: Predetermined escape path, recommended action

**Opponent-robust-survivable action**:
An action admitted by the controllable survival set. v5 proves this against a
conservative superset of official opponent occupancy and bomb danger, so it may
reject extra actions but may not accept an unproved action.
_Avoid_: Likely-safe action, opponent prediction

**Robust bomb-placement veto**:
The refusal to place a bomb when placement lacks a completed controllable
survival proof and a v1-safe non-bomb alternative exists. It assigns the safety
decision to the action that creates the future obligation.
_Avoid_: Bomb penalty, rule-selected movement

**Own-bomb danger interval**:
The placement transition through the last dangerous flame produced by that
bomb, while also respecting all other timed explosions. The official framework
does not implement chain reactions, so the model must not invent them.
_Avoid_: Countdown only, chain-reaction window

**Fail-closed robust evaluation**:
If the survival proof exceeds its internal budget or cannot be completed, the
candidate is not marked robust. A timeout is recorded and fails experiment
admission even if rejecting the action happened to prevent death.
_Avoid_: Timeout success, safe by default

**Robust guarantee loss**:
During an active own-bomb obligation, no action passes v5 and execution falls
back to the v1 survival set (or ultimately physical Q). It is distinct from a
death and must be zero at the safety gates.
_Avoid_: Ordinary intervention, every fallback

**Safety regression corpus**:
The immutable eleven environment seeds containing seven legacy Task 3
self-deaths and four n-step-5 Task 3 self-deaths. It is used before any fresh
evaluation seeds, never for model selection.
_Avoid_: Development set, training seeds

**Frozen safety admission**:
The one-time 100-seed evaluation of unchanged weights under v5. Passing requires
the fixed safety, capability-retention, bomb-use, invalid-action, and latency
gates before any new Task 3 training is allowed.
_Avoid_: Training result, Task 3 winner

**Immediate-parent distillation teacher**:
The frozen policy from the directly preceding curriculum stage. A Task 3 child
uses its Task 2 parent's policy, never the older teacher embedded in that parent.
_Avoid_: Inherited teacher, original teacher

**Earliest passing curriculum checkpoint**:
The first preregistered checkpoint that simultaneously passes capability,
retention, safety, and engineering gates.
_Avoid_: Final checkpoint, longest-trained checkpoint

This remains the name of the completed retention-prefix experiment. New Task 3
plateau experiments use a plateau-best qualified checkpoint instead.

**Frozen multi-task assessment**:
An exploration-free evaluation of one immutable checkpoint on Task 1, Task 2,
and Task 3 using the same dedicated development worlds as its parent baseline.
It is the only observation used by Task 3 plateau stopping.
_Avoid_: Training reward curve, online training score

**Performance plateau**:
A preregistered sequence of frozen multi-task assessments with no meaningful
official-score improvement and no qualifying gate progress for the applicable
patience window. It can only be observed after checkpoints exist; it does not
predict a future decline.
_Avoid_: Convergence proof, loss plateau, predicted degradation

**Plateau-best qualified checkpoint**:
The best checkpoint that passes every capability, retention, safety, and
engineering gate before a Task 3 performance plateau stops training. Later
training checkpoints may be discarded without deleting their evidence.
_Avoid_: Latest checkpoint, first passing checkpoint

**Budget-truncated qualified checkpoint**:
A best qualified checkpoint selected when the preregistered round cap is
reached while meaningful improvement is still occurring. It may enter
independent confirmation, but it is not described as plateau-converged.
_Avoid_: Converged checkpoint, training failure

**Coin capability assessment**:
A frozen, exploration-free Task 1 evaluation over the dedicated convergence seeds. Its primary result is mean game score; Task 1 requires every episode's score to equal its collected-coin count.
_Avoid_: Training reward, recent training score

**Task 1 score convergence**:
Three consecutive coin capability assessments whose mean score is at least 48, with assessments separated by 50 newly trained rounds.
_Avoid_: Action budget reached, training completed

**Curriculum promotion qualification**:
Task 1 score convergence followed by a passing, independent stage-gate evaluation and all engineering checks. Convergence alone does not authorize promotion.
_Avoid_: Converged, completed

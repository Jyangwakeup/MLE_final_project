# Rainbow no-safety Task 2

Train and evaluate Rainbow-lite for Task 2 with `survival-mask-v1/mode=off`.
No horizon-survivable action may be removed during exploration, greedy action
selection, bootstrap, or frozen inference.

Improve learning credit by enabling the configured temporal reward path,
assigning self-kill credit to the causal bomb-placement transition, and adding
a bounded state-potential term for the fraction of physically legal actions
that remain horizon-survivable. Existing reward identities and evidence remain
unchanged.


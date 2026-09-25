# Agent-centred quadrant context

Add eight factual inputs to the frozen continuous-v7 representation: normalized
crate and opponent density in the north-west, north-east, south-west, and
south-east quadrants relative to the Agent's current position.

Cells on the Agent's horizontal or vertical axis contribute half to each
adjacent quadrant; the Agent cell itself contributes equally to all quadrants
for denominator accounting but can contain neither a crate nor an opponent.
Density denominators count non-stone cells, so border walls do not distort the
signal.  The features do not change rewards, masks, or action selection rules.

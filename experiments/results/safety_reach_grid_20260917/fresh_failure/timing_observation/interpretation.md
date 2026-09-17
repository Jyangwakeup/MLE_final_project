# Timing observation, not a root-cause claim

500 fixed-core repetitions of the captured failure state completed without a timeout. Complete reference and current proofs match exactly: RIGHT, LEFT, WAIT, BOMB; 840 scenarios and 616,094 logical states. Offline reference budget was 60 seconds and does not alter runtime thresholds.

The slowest observed runtime call took 393.213ms, close to the unchanged 400ms budget. Several other slow calls include generation-2 garbage collections lasting roughly 88–105ms. CPU time closely follows wall time in those samples, so scheduler descheduling alone does not explain those particular slow calls. This does not identify the cause of the original 408.911ms whole-act failure, whose GC/CPU timing was not recorded.

Next retain whole-worker context while recording slow-call GC and CPU timings; do not disable collection or change safety thresholds on this evidence alone. The original 384-completed-game sweep remains terminal failure.

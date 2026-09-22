# CNN Path uses the winner's r7 safety contract

The CNN Path agent retains its 17-channel spatial representation and residual
network.  It uses `r7_safe_credit_sparse` with `survival-mask-v1` in `all`
mode, horizon seven, and `physical_q` fallback.

The mask vetoes only actions shown by the environment reachability analysis to
be avoidably fatal.  It is applied both to behavior and to the Double-DQN
bootstrap action set, rather than supplying a target direction or an optimal
action.  In Task 1 BOMB remains disabled by the curriculum.

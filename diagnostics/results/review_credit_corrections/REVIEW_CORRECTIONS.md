# Review corrections

Existing 1000 states and 23065 branches reused; no retraining/replay.

Prior task rewards depend on the forced action in 43 states (45 prior tasks). Including pending outcomes changes the sampled optimal set in 32 states.

The new all-outcomes event return includes own, future-decision and preexisting pending outcomes. Policy centering removes action-independent constants; prior-task rewards are not presumed unaffected.

The historical non-conflict group accounts for the positive PPO-minus-SafeMin regret gap; the conflict group offsets part of it. See performance_gap_decomposition.csv. These are fixed-continuation contrasts, not causal attribution of training failure.

Top-1 now refers to the explicit selected lowest-index tied-best action in ranking tables; best-set intersection is a separate metric. Actual stochastic choices remain in the deployment-action tables.

One continuation per action is insufficient for an expected-Q oracle. Repeat continuations with paired exogenous seeds and estimate state/action means and uncertainty before selecting a causal root cause. The historical decision-vs-event comparison does not test omitted pending-outcome credit.

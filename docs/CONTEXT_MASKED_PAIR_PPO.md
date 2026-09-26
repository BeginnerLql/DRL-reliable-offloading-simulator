# Context Masked Pair PPO v2

This version addresses reproducible issues found in the code review. It is an
independent agent and entry point, not a replacement of the historical baseline.
Legacy `pair_scoring`, masked agent defaults, entry points and checkpoints retain
exact behavior. No task reward formula, execution, safe-mask or reliability
formula changed. The observation stays at 35 dimensions and actions at 28.

## Actor representation

`pair_context` uses complete node tuples, static pair rho, current task features
and the full observed node context. Its shared scorer averages the two ordered
node scores. Swapping whole node tuples preserves the score; swapping only their
backlog coordinates no longer destroys the association of backlog with CPU and
uplink. Global observed context permits relative pair preferences to depend on
other nodes. This adds no hidden effective-hazard or in-flight observations.
Old Actor checkpoints cannot be loaded into this different architecture.

## Causal event-interval credit

Task reward logging still records the reward of each originating task exactly
once. The new outcome hook also records its true first-result timestamp, even
when bookkeeping happens at a later task arrival. Before training, v2 constructs

```
r_interval[k] = sum(gamma ** (outcome_time - decision_time[k]) * task_reward)
```

for outcomes in `[decision_time[k], decision_time[k+1])`; the final interval
includes the terminal endpoint. Equal-time decisions have empty zero-duration
intervals. This includes an earlier task's outcome when a later decision affects
it through upload/queue overtaking. It does not duplicate that reward in the
originating transition. Within-interval discounting is applied once; existing
`gamma ** delta_t` GAE propagates returns between intervals. Each episode checks
that its discounted interval return equals its discounted outcome-event return.

This is a **versioned temporal objective change**, not a relabeling of the old
origin-task GAE or a claim of measured performance improvement. `--credit-mode
origin_task` retains the historical task-credit objective for ablation.

## Optimization

Actor and Critic are clipped independently by default. `--gradient-clipping
joint` reproduces the historical global clipping operation. Gradient norms are
logged before clipping. Adam partly compensates gradient scaling, so shared
clipping alone was not proven to explain the old performance gap.

## Running

From the repository root with the project Python environment:

```bash
python tools/run_context_masked_pair_ppo.py \
  --train-episodes 3 --eval-episodes 3 \
  --output-dir diagnostics/results/my_context_v2_smoke

python tools/run_context_masked_pair_ppo.py \
  --checkpoint-dir diagnostics/results/my_context_v2_smoke \
  --eval-episodes 3 --output-dir diagnostics/results/my_context_v2_deployment
```

Deployment defaults to masked stochastic; `--deployment greedy` is explicit.
A new output directory is required. Checkpoint evaluation restores its saved
actor/credit/clipping modes. Architecture, credit and clipping can be ablated
independently using `--actor-mode pair_scoring`, `--credit-mode origin_task` and
`--gradient-clipping joint`. With all three legacy choices, regression tests
verify identical task logs and trained Actor/Critic parameters to the old agent.

## Diagnostic corrections

`diagnostics/review_credit_diagnostics.py` reads full-precision historical
provenance and writes a separate supplement. It adds policy-centered pending-task
outcomes to the prior own-plus-future analysis. Historical results are not
rewritten. In 1,000 states / 23,065 branches, 43 states contain action-dependent
prior-task rewards; incorporating pending outcomes changes 32 sampled optimal
sets.

Ranking Top-1 now refers to the explicitly selected tied-best action. Best-set
intersection is separately named. Actual stochastic deployment metrics are not
interchanged with greedy ranking metrics. Automatic single-continuation causal
classification is disabled: these contrasts remain sample-path observations,
not expected-Q estimates. Repeated paired continuations and uncertainty estimates
are still required before an algorithmic root-cause or effectiveness claim.

## Validation at integration

- Functional smoke: 3 training episodes, 3 frozen stochastic evaluation episodes,
  200 tasks each; this is not a formal multi-seed benchmark.
- Evaluation: mean reward 60.2216, mean latency 2.0604 seconds, overall RSR 98%,
  highest-tier RSR 92%, conditional RSR 100%, avoidable violations 0.
- Initial minibatch ratio error: 0 for all 3 training episodes; all recorded
  ratios finite, range about 0.9909–1.0092.
- Maximum discounted-return conservation residual: 5.12e-13.
- Regression: 377 tests and 90 subtests pass, 2 DQN/DDPG tests deselected;
  one existing tensor-construction performance warning.

The smoke results establish functioning training/deployment and credit
bookkeeping only. They must not be compared with historical 10-seed results as
an effectiveness claim; the training lengths and evaluation samples differ.

# Coframe Personalization Handoff Summary

## Direct Answer

Yes: a policy can reliably pick a better variant per visitor than the best single global variant.

The strongest fully verified run I completed was:

| policy | average captured headroom |
| --- | ---: |
| random | 0.2% |
| best_fixed / my_policy | 0.3% |
| example | 18.3% |
| seg_country | 42.7% |
| seg_eb | 62.7% |
| gbm_tlearner | 65.4% |
| hybrid_eb_gbm | 69.6% |

The best story is: direct reward modeling wins, but the functional form matters. The empirical-Bayes model is a strong conservative baseline. A boosted-tree T-learner captures extra nonlinear and time-related signal. A support-aware hybrid is best because it uses EB shrinkage when data is thin and boosted trees when support is large.

## Where The Code Is

Repo:

```text
/Users/guclaw/repos/applied-science-take-home/personalization
```

Important files:

| file | purpose |
| --- | --- |
| `solution/eb_policy.py` | Empirical-Bayes additive T-learner. Registers as `seg_country`, `seg_eb`, `seg_eb_recency`. |
| `solution/gbm_policy.py` | sklearn histogram gradient-boosted T-learner plus `hybrid_eb_gbm`. This produced the best verified full-run score. |
| `solution/lgbm_policy.py` | LightGBM T-learner and `hybrid_eb_lgbm` experiment. Faster backend, but only quick-tested before handoff. |
| `solution/__init__.py` | Policy registration. |
| `solution/observability.py` | Generates data observability plots. |
| `solution/eval_observability.py` | Generates harness/evaluation plots. |
| `solution/arm_sensitivity.py` | Truth-only diagnostic for arm-choice sensitivity. Not used for training. |
| `solution/research_log.md` | Longer causal/ML strategy notes. |
| `solution/report.html` | Static HTML report draft for GitHub viewing. |
| `solution/index.html` | Redirects to `report.html`. |
| `solution/submission_checklist.md` | Deliverables/video checklist. |

Dependencies added:

```text
scikit-learn>=1.4
lightgbm>=4.0
```

## Important Caveat For The Next Agent

The last full 12-policy eval was stopped because it was taking too long. I killed only the run in this repo. A separate `/tmp/coframe-th` run from the other agent was left alone.

Also: quick LightGBM tests overwrote `solution/results/summary.json` after the best full run. The best verified full-run numbers above came from the completed 10-policy run before the quick LightGBM experiments. The next agent should rerun or restore the full final eval before submitting plots/results.

Recommended final eval command:

```bash
uv run run_eval.py --policies random,my_policy,best_fixed,example,feature_demo,seg_country,seg_eb,seg_eb_recency,gbm_tlearner,hybrid_eb_gbm --workers 1 --threads 2
uv run plot_results.py
uv run python solution/observability.py
uv run python solution/eval_observability.py
uv run python solution/arm_sensitivity.py
```

If including LightGBM in the final comparison:

```bash
uv run run_eval.py --policies random,my_policy,best_fixed,example,feature_demo,seg_country,seg_eb,seg_eb_recency,gbm_tlearner,hybrid_eb_gbm,lgbm_tlearner,hybrid_eb_lgbm --workers 1 --threads 2
```

## Problem Framing

This is a contextual bandit, not reinforcement learning.

At each impression:

```text
context x = visitor/session features
action a = one available variant
reward r = binary conversion
target = choose argmax_a E[reward | x, a]
```

There is no multi-step state transition, no Bellman equation, and no online exploration in the scored task. The harness does walk-forward evaluation:

```text
time 11: train on all data through time 10, score on time 11
time 15: train on all data through time 14, score on time 15
```

That lets the policy keep learning as history grows without leaking future outcomes.

## Randomization

The data appears randomized.

Four datasets look like equal-split randomized tests:

```text
atlas, meadow, vega, zephyr
```

Two datasets have explicit propensities:

```text
helios: non-uniform random assignment with propensities like 0.05, 0.25, 0.85
rotation: random over a changing live action set, propensities about 1/8 or 1/9
```

Interpretation:

- We should respect propensities for causal reasoning and diagnostics.
- We should not feed propensity as an ordinary user feature at serve time.
- Raw inverse-propensity weighting is high variance when propensities are small.
- Direct reward modeling plus shrinkage is a good scored-policy approach.

## Reward Function

The reward is binary conversion.

The policy estimates:

```text
mu(x, a) = P(conversion = 1 | context x, variant a)
```

Then serves:

```text
argmax_a mu_hat(x, a)
```

restricted to `available_variants`.

The harness evaluates against hidden truth files, not noisy off-policy estimates:

```text
captured% = (policy_cvr - best_fixed_cvr) / (oracle_cvr - best_fixed_cvr) * 100
```

## Models Tried

### 1. Starter / Baselines

- `random`
- `my_policy`
- `best_fixed`
- provided `example`
- `feature_demo`

These establish that global best is basically worthless and the problem is genuinely contextual.

### 2. Empirical-Bayes T-Learner

Implemented in:

```text
solution/eb_policy.py
```

Policies:

```text
seg_country
seg_eb
seg_eb_recency
```

This models each arm's reward surface with shrunk segment effects:

```text
score(x, a)
  = global arm prior
    + country/language/platform/device effects
    + selected country cross effects
```

Why it worked:

- strong protection against sparse cells;
- interpretable;
- leak-free;
- very good on `vega`;
- strong baseline at 62.7% average captured headroom.

Concern:

- additive/logit functional form may miss nonlinear interactions.

### 3. Gradient-Boosted T-Learner

Implemented in:

```text
solution/gbm_policy.py
```

Policy:

```text
gbm_tlearner
```

This directly fits one model per arm:

```text
for each arm a:
  fit P(reward | x, a) using rows where that arm was shown
```

Result:

- better than EB on large/drifting datasets;
- worse on sparse `vega`;
- average 65.4% captured headroom;
- slower than EB.

### 4. Support-Aware Hybrid

Implemented in:

```text
solution/gbm_policy.py
```

Policy:

```text
hybrid_eb_gbm
```

Rule:

```text
if not enough data or any arm has weak support:
  use seg_eb
else:
  use gbm_tlearner
```

This was the best verified full-run policy:

```text
69.6% average captured headroom
```

### 5. LightGBM Experiment

Implemented in:

```text
solution/lgbm_policy.py
```

Policies:

```text
lgbm_tlearner
hybrid_eb_lgbm
```

Reason:

The user correctly pointed out sklearn is not the fastest boosted-tree package. LightGBM is a more realistic boosted-tree backend.

Status:

- installed and registered;
- quick-tested only;
- faster than sklearn GBM in quick tests;
- first-pass hyperparameters did not beat `hybrid_eb_gbm`;
- worth tuning if time allows.

## Stale Data / Continuous Learning

We are not doing RL or online exploration in the take-home.

We are doing walk-forward retraining:

- train only on past data;
- recommend on the next future window;
- fold that window into history;
- retrain later.

The stale-data issue is real: if `P(reward | context, action)` changes over time, old rows can hurt. `zephyr` is the clearest drift/stale-data dataset.

Approaches tried:

- `seg_eb_recency`: exponential time decay, helps `zephyr`;
- `gbm_tlearner`: includes timestamp-derived features, does much better on `zephyr`;
- `hybrid_eb_gbm`: gets most of the flexible-model benefit while protecting sparse data.

Production next step:

Use adaptive recency or drift detection, not a hard-coded dataset rule.

## Observability / Displays

Generated data displays:

```text
solution/results/observability/
```

Key plots:

- `overview.png`
- `assignment_mix.png`
- `propensity_diagnostics.png`
- `action_availability.png`
- `segment_signal_heatmap.png`
- `temporal_reward.png`
- `temporal_assignment_mix.png`

Generated evaluation displays:

```text
solution/results/eval_observability/
```

Key plots:

- `policy_scoreboard_heatmap.png`
- `captured_convergence_grid.png`
- `harness_walk_forward_windows.png`
- `final_value_components.png`
- `policy_runtime.png`
- `final_recommendation_mix.png`
- `choice_violations.png`

Generated arm-sensitivity displays:

```text
solution/results/arm_sensitivity/
```

Key plots:

- `oracle_winner_share.png`
- `top_second_gap.png`
- `leave_one_arm_out_loss.png`

Note: arm-sensitivity uses truth files for analysis only, not training.

## How To Communicate The Solution

Simple explanation:

> We used the randomized logs to learn, for each visitor context and each variant, the expected conversion probability. Then we chose the live variant with the highest predicted conversion. We evaluated this honestly with walk-forward training so the model only learned from the past. The final method is support-aware: it uses a conservative empirical-Bayes model when data is sparse and a boosted-tree model when enough data exists.

Video structure:

1. Explain context, action, reward.
2. Show randomization and propensities.
3. Show the walk-forward harness.
4. Show model ladder: random -> best fixed -> segment EB -> boosted T-learner -> hybrid.
5. Show final captured-headroom numbers.
6. Discuss stale data / drift and production OPE.

## Themes From The User's Prompts

Your prompts pushed the work in the right direction because they kept forcing the solution away from a shallow leaderboard chase and toward a defensible applied-science story.

Main themes:

1. **Causal identification and randomization**
   You asked whether assignment was random, whether propensities matter, and whether we need to condition on assignment chance. That led to the right Rosenbaum-Rubin framing and diagnostics.

2. **Variance and small propensities**
   You emphasized that low propensities create high-variance estimates. That pushed the model toward shrinkage and away from blindly using raw inverse-propensity weights.

3. **State/action/reward clarity**
   You asked what the state, action, context, and reward are. That clarified this as a contextual bandit rather than RL.

4. **No leakage, walk-forward learning**
   You asked about training at time 11 on time 10 and before, then retraining later. That is exactly the harness logic and the right production analogy.

5. **Stale data and drift**
   You asked whether relationships can change over time. That led to recency weighting, timestamp features, and the zephyr drift story.

6. **Functional-form sensitivity**
   You challenged the additive logit EB form and asked for boosting. That was correct: GBM improved the average and led to the final hybrid.

7. **Speed and production realism**
   You challenged sklearn and pointed toward faster GBM packages. That led to the LightGBM experiment and a better production discussion.

8. **Observability and communication**
   You asked for displays and an outsider-friendly explanation. That led to the HTML report, observability plots, and video-ready narrative.

## What The Next Agent Should Do

1. Rerun the final full eval because quick LightGBM tests overwrote `summary.json`.
2. Decide whether to submit `hybrid_eb_gbm` as final or tune `hybrid_eb_lgbm`.
3. Update `solution/report.html` with the final scoreboard after the rerun.
4. Run:

```bash
uv run plot_results.py
uv run python solution/observability.py
uv run python solution/eval_observability.py
uv run python solution/arm_sensitivity.py
```

5. Commit `solution/`, generated results, `pyproject.toml`, and `uv.lock`.


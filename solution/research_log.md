# Personalization Policy Research Log

> **Final status (read this first).** The numbered sections below are the original
> chronological exploratory notes — kept on purpose, because the journey is part of
> the deliverable. This top block is the **final summary**: what was actually built,
> the verified results, what drove each decision, what was rejected, and every AI
> tool and test harness used.

## 0. Final summary

### What was built — the policy ladder (verified full walk-forward, 50 iterations)

| policy | what it is | helios | meadow | rotation | vega | zephyr | **avg** |
|---|---|---|---|---|---|---|---|
| `random` | uniform | −0.4 | 0.2 | −0.1 | 1.3 | 0.1 | **0.2** |
| `best_fixed` / `my_policy` | best single global arm | −0.1 | 0.2 | −0.1 | −0.3 | 1.6 | **0.3** |
| `example` (provided) | one-feature smoothed table | 19.8 | 26.2 | 13.0 | 13.8 | 18.9 | **18.3** |
| `seg_country` | EB table, country only | 55.6 | 47.0 | 41.6 | 42.3 | 27.2 | **42.7** |
| `seg_eb` | EB additive T-learner (mains + country crosses) | 77.3 | 72.2 | 61.2 | 55.3 | 47.4 | **62.7** |
| `seg_eb_recency` | seg_eb + exponential time decay (drift ablation) | 75.1 | 70.0 | 57.5 | 48.2 | 58.9 | **61.9** |
| `gbm_tlearner` | gradient-boosted T-learner (one booster/arm) | 78.3 | 78.7 | 61.4 | 32.7 | 75.9 | **65.4** |
| **`hybrid_eb_gbm`** | **support-aware: EB when thin, GBM when rich** | **79.7** | 78.4 | **61.9** | **55.3** | 72.7 | **69.6** |
| `lgbm_tlearner` | LightGBM backend (faster) | 76.3 | 76.6 | 57.7 | 29.5 | 74.4 | **62.9** |
| `hybrid_eb_lgbm` | LightGBM hybrid | 78.2 | 77.5 | 61.1 | 55.3 | 73.1 | **69.1** |

`captured%` = share of the best-fixed→oracle gap captured. `atlas` is omitted (no
headroom; captured% is n/a — see edge cases). **Shipped policy: `hybrid_eb_gbm`, 69.6%.**

### What drove each decision
- **Direct reward model, not uplift contrasts.** Estimate `μ(x,a)=P(convert|x,a)`,
  argmax over available arms. The per-row baseline cancels in the argmax, so modeling
  levels is sufficient and lower-variance than estimating K−1 contrasts.
- **Empirical-Bayes shrinkage is the backbone** — it is the do-no-harm mechanism
  (sparse / zero-signal cells collapse to the arm prior), which makes `seg_eb` strong
  and safe, and is exactly why the hybrid stays safe on `vega`.
- **Gradient boosting was added because the additive form left signal on the table**
  on the large datasets (meadow +9pp, helios). But pure GBM *overfits sparse vega*
  (32.7% vs EB's 55.3%).
- **The hybrid resolves it by support**: EB when a segment/arm is thin, GBM when
  rich → keeps EB's vega protection (55.3%) and most of GBM's big-data/drift gains →
  **69.6%**. An **arm-sensitivity analysis** (how often the forms agree on the chosen
  arm, and whether their disagreements cost conversion) confirmed the rule: GBM's
  disagreements *add* value where data is rich, *destroy* it on sparse vega, and are
  *benign* on atlas.
- **Recency is adaptive, not global.** Global time-decay helps zephyr (the concept-
  drift dataset, +11.5pp) but hurts the four stationary datasets, so it is an ablation,
  not the default; the production answer is change-point-gated decay.

### What was rejected, and why
- **Raw inverse-propensity weighting (IPW)** — assignment is unconfounded given the
  observed features (Rosenbaum–Rubin: conditioning on X is the adjustment), so IPW
  adds variance with no bias payoff. Kept only as a diagnostic / OPE topic. (Helios
  ESS drops 150k → ~40k under IPW — the variance trap.)
- **Uplift / CATE contrast modeling (X/R/DR-learner, causal forests)** — the right
  tool for budgeted allocation or a designated control, neither of which this has.
- **Pure GBM as the shipped policy** — overfits sparse vega.
- **Global recency weighting** — hurts the stationary datasets.
- **LightGBM first-pass hyperparameters** — faster than sklearn but did not beat the
  sklearn hybrid; kept as the production-speed backend pending tuning.

### Every AI tool and test harness used
- **Claude Code (Anthropic, Opus)** — the agent that built this, run as **two parallel
  threads** on the same take-home (a primary modeling thread and a parallel docs /
  observability thread), with findings cross-checked between them. A **multi-agent
  workflow** of subagents profiled the harness and data in parallel; a **review
  subagent** audited this report before submission.
- **The provided evaluation harness** — `run_eval.py` (walk-forward, leak-free,
  truth-scored captured headroom) and `plot_results.py`.
- **Custom test / observability harnesses added here** — `solution/observability.py`
  (data displays), `solution/eval_observability.py` (harness / eval dashboards), and
  `solution/arm_sensitivity.py` (truth-only arm-choice sensitivity diagnostics).
- **Modeling libraries** — `numpy` / `pandas` (the empirical-Bayes table),
  `scikit-learn` (first gradient-boosted T-learner), then **LightGBM** (faster
  backend, added on the user's prompt). Environment managed with `uv`.

---

## 1. Problem framing

This harness is a contextual bandit / treatment assignment problem, not a
full reinforcement learning problem.

At each impression:

```text
context x        = visitor and session features
action a         = one available variant_id
observed reward  = conversion for the variant that was shown
policy target    = choose the available variant with the highest expected conversion
```

There is no multi-step action sequence in the benchmark. The only sequence is
the evaluation protocol:

```text
past logged data -> fit policy
next time window -> recommend variant for each context
hidden truth     -> evaluator scores recommendations
window becomes history -> repeat
```

The policy is allowed to learn from past rows containing `context`,
`variant_id`, `reward`, `timestamp`, and sometimes `propensity`. At serve time,
`recommend()` receives only the context columns plus timestamp and the currently
available action set. It does not receive the logged action, reward, or
propensity for the evaluation window.

The evaluator scores against hidden counterfactual truth:

```text
captured% = (policy_cvr - best_fixed_cvr) / (oracle_cvr - best_fixed_cvr) * 100
```

That means the benchmark score is not an off-policy estimate. The truth files
are used by the evaluator only. A valid policy should not read them.

## 2. Causal refresher: assignment, propensities, and variance

Each row has potential outcomes:

```text
Y(variant_1), Y(variant_2), ..., Y(variant_K)
```

The log only reveals the outcome for the assigned action:

```text
observed reward = Y(A)
```

If assignment is randomized conditional on observed information, then the
logged action is ignorable given that information:

```text
A independent of potential outcomes | X, e(X)
```

where `e(X)` is the propensity score or assignment probability. This is the
Rosenbaum-Rubin idea: if the chance of assignment is known and captures the
assignment mechanism, treatment comparisons can be adjusted using that
propensity rather than pretending the observed action mix was naturally
representative.

For this project:

- We should not feed `propensity` into the model as an ordinary user feature.
- We can use `propensity` as an estimation correction or reliability signal.
- Small propensities create large weights and high-variance estimates.

The classic inverse propensity weight is:

```text
weight_i = 1 / p(A_i | X_i)
```

If `p = 0.05`, the row gets weight `20x`. That can correct assignment skew, but
it also means one noisy conversion can dominate a small cell. This is why the
final policy should use shrinkage and avoid overfitting thin action/context
cells.

Useful variance intuition:

```text
unweighted Bernoulli SE ~= sqrt(p_hat * (1 - p_hat) / n)
weighted effective sample size = (sum w)^2 / sum(w^2)
weighted SE ~= sqrt(p_hat * (1 - p_hat) / ESS)
```

The effective sample size is the number of independent equal-weight rows that
would carry similar information. When weights are uneven, ESS can be much
smaller than raw row count.

## 3. Data overview

All datasets cover May 2, 2025 through May 31, 2025.

| dataset | rows | variants | observed reward rate | features | propensity logged |
| --- | ---: | ---: | ---: | ---: | --- |
| atlas | 80,000 | 4 | 6.07% | 9 | no |
| helios | 150,000 | 4 | 5.83% | 8 | yes |
| meadow | 150,000 | 4 | 8.02% | 9 | no |
| rotation | 120,000 | 10 | 6.01% | 7 | yes |
| vega | 6,000 | 4 | 5.30% | 9 | no |
| zephyr | 200,000 | 4 | 7.94% | 8 | no |

Feature profile:

| dataset | feature sketch |
| --- | --- |
| atlas | platform:3; device_type:3; country:40; language:24; referrer_id:399; color_scheme:3; is_returning; visits_count; days_since_last_visit |
| helios | platform:3; device_type:3; country:40; language:24; referrer_id:482; color_scheme:3; is_returning; days_since_last_visit |
| meadow | platform:3; device_type:3; country:40; language:24; referrer_id:511; color_scheme:3; is_returning; visits_count; days_since_last_visit |
| rotation | platform:3; device_type:3; country:40; language:24; referrer_id:447; color_scheme:3; is_returning |
| vega | platform:3; device_type:3; country:40; language:24; referrer_id:109; color_scheme:3; is_returning; visits_count; days_since_last_visit |
| zephyr | platform:3; device_type:3; country:40; language:24; referrer_id:589; color_scheme:3; is_returning; visits_count |

## 4. What randomization looks like

Variant assignment is not identical across datasets.

| dataset | assignment pattern |
| --- | --- |
| atlas | near uniform: each variant about 25% |
| meadow | near uniform: each variant about 25% |
| vega | near uniform: each variant about 25% |
| zephyr | near uniform: each variant about 25% |
| helios | non-uniform randomized assignment with logged propensities |
| rotation | changing action set over time with logged propensities |

Variant shares:

| dataset | variant shares |
| --- | --- |
| atlas | v1 24.90%, v2 24.89%, v3 24.99%, v4 25.21% |
| helios | v1 6.75%, v2 54.13%, v3 27.70%, v4 11.42% |
| meadow | v1 24.96%, v2 25.07%, v3 25.09%, v4 24.87% |
| rotation | v01 12.28%, v02 12.30%, v03 12.29%, v04 12.06%, v05 12.25%, v06 12.19%, v07 4.91%, v08 8.47%, v09 8.46%, v10 4.79% |
| vega | v1 24.37%, v2 25.27%, v3 25.17%, v4 25.20% |
| zephyr | v1 25.07%, v2 25.03%, v3 25.04%, v4 24.87% |

Propensity diagnostics:

| dataset | propensity values | min | median | max | note |
| --- | --- | ---: | ---: | ---: | --- |
| atlas | not logged | | | | assignment appears near-uniform |
| helios | 0.05, 0.25, 0.85 | 0.05 | 0.85 | 0.85 | 20x IPW weights; ESS about 39,818 / 150,000 |
| meadow | not logged | | | | assignment appears near-uniform |
| rotation | 0.1111, 0.125 | 0.1111 | 0.125 | 0.125 | weights are mild; ESS about 119,723 / 120,000 |
| vega | not logged | | | | assignment appears near-uniform but small data |
| zephyr | not logged | | | | assignment appears near-uniform |

The key warning case is `helios`. Its raw action distribution is highly skewed,
and rows with propensity 0.05 have weight 20 under inverse propensity weighting.
That creates exactly the high-variance regime where shrinkage matters.

`rotation` has propensities too, but its weights are much gentler. Its bigger
issue is changing action availability:

```text
variant_07: 2025-05-02 -> 2025-05-13
variant_08: 2025-05-02 -> 2025-05-22
variant_09: 2025-05-11 -> 2025-05-31
variant_10: 2025-05-20 -> 2025-05-31
```

The policy must always restrict choices to `available_variants`.

## 5. Support and variance by action

Unweighted standard errors are small for large balanced datasets, but much
larger for `vega` and for rare actions in `helios` and `rotation`.

| dataset | variant | n | observed reward | unweighted SE | weighted ESS | weighted reward | weighted SE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlas | variant_1 | 19,924 | 6.25% | 0.17% | | | |
| atlas | variant_2 | 19,914 | 5.87% | 0.17% | | | |
| atlas | variant_3 | 19,991 | 6.17% | 0.17% | | | |
| atlas | variant_4 | 20,171 | 5.99% | 0.17% | | | |
| helios | variant_1 | 10,125 | 7.13% | 0.26% | 7,835 | 7.33% | 0.29% |
| helios | variant_2 | 81,191 | 5.93% | 0.08% | 18,180 | 8.20% | 0.20% |
| helios | variant_3 | 41,550 | 5.04% | 0.11% | 10,397 | 8.43% | 0.27% |
| helios | variant_4 | 17,134 | 6.51% | 0.19% | 8,157 | 8.16% | 0.30% |
| meadow | variant_1 | 37,444 | 7.94% | 0.14% | | | |
| meadow | variant_2 | 37,609 | 7.92% | 0.14% | | | |
| meadow | variant_3 | 37,635 | 8.20% | 0.14% | | | |
| meadow | variant_4 | 37,312 | 8.04% | 0.14% | | | |
| rotation | variant_07 | 5,897 | 6.14% | 0.31% | 5,882 | 6.12% | 0.31% |
| rotation | variant_10 | 5,745 | 5.92% | 0.31% | 5,730 | 5.92% | 0.31% |
| vega | variant_1 | 1,462 | 5.75% | 0.61% | | | |
| vega | variant_2 | 1,516 | 4.55% | 0.54% | | | |
| vega | variant_3 | 1,510 | 5.30% | 0.58% | | | |
| vega | variant_4 | 1,512 | 5.62% | 0.59% | | | |
| zephyr | variant_1 | 50,133 | 7.97% | 0.12% | | | |
| zephyr | variant_4 | 49,737 | 8.04% | 0.12% | | | |

The `helios` weighted estimates are also a useful warning: naive rates say
variant_1 looks best, while weighted rates suggest variants 2, 3, and 4 are
competitive or better after assignment correction. This is the kind of dataset
where "global best by observed reward" can be misleading.

## 6. Initial signal checks

I ran a simple diagnostic: within one feature at a time, estimate
variant-by-segment reward rates with additive shrinkage toward global variant
rates, then ask how much that would improve in-sample over the global best
variant. This is only a signal diagnostic, not a valid final policy estimate.

Top feature signals:

| dataset | feature | smoothed in-sample lift proxy vs global best |
| --- | --- | ---: |
| atlas | country | 0.481pp |
| atlas | language | 0.353pp |
| atlas | referrer_id | 0.183pp |
| helios | country | 5.032pp |
| helios | language | 3.474pp |
| helios | referrer_id | 2.334pp |
| meadow | country | 4.495pp |
| meadow | language | 2.916pp |
| meadow | platform | 2.460pp |
| rotation | country | 5.852pp |
| rotation | language | 3.506pp |
| rotation | referrer_id | 3.009pp |
| vega | country | 2.093pp |
| vega | language | 1.946pp |
| vega | platform | 1.281pp |
| zephyr | country | 2.961pp |
| zephyr | platform | 2.143pp |
| zephyr | language | 1.839pp |

This points toward a hierarchical contextual policy:

```text
global variant prior
-> single-feature segment effects
-> possibly selected low-order interactions
-> shrink aggressively based on cell support / ESS
```

## 7. What we have tried so far

Completed evaluator runs currently in `solution/results/`:

| policy | description |
| --- | --- |
| random | uniformly random among available variants |
| my_policy | starter policy: chooses best historical global variant rate |
| example | provided worked example: smoothed per-segment variant rates using first categorical feature |
| feature_demo | provided feature-pipeline example: smoothed time-of-day by variant |

Full walk-forward results:

| dataset | random | my_policy | example | feature_demo |
| --- | ---: | ---: | ---: | ---: |
| atlas | n/a | n/a | n/a | n/a |
| helios | -0.4% | -0.1% | 19.8% | 1.9% |
| meadow | 0.2% | 0.2% | 26.2% | 4.0% |
| rotation | -0.1% | -0.1% | 13.0% | 1.4% |
| vega | 1.3% | -0.3% | 13.8% | 4.8% |
| zephyr | 0.1% | 1.6% | 18.9% | 6.0% |
| average over headroom datasets | 0.2% | 0.3% | 18.3% | 3.6% |

Interpretation:

- The starter global-best policy barely improves over random.
- The simple segment example is already much stronger, which means the problem
  is genuinely contextual.
- Time of day helps, but less than direct visitor/context segmentation.
- `atlas` has no headroom: best fixed equals oracle, so the right behavior is
  to do no harm.
- `helios` is a propensity/variance warning case.
- `rotation` is an action-set-change warning case.
- `vega` is sparse, so high-dimensional models can easily chase noise.

## 8. ML strategy

The policy should estimate:

```text
mu(x, a) = E[conversion | context x, variant a]
```

Then choose:

```text
argmax_a mu_hat(x, a)
```

among the variants available in the current window.

The main risk is not that the target is hard to write down. The risk is that
the estimates are noisy, especially in sparse cells, small datasets, and
low-propensity actions. The strategy should therefore be a reliability-weighted
hierarchical estimator rather than a huge unconstrained model.

### Candidate final policy

Use layered smoothed estimates:

```text
score(x, a)
  = global_variant_rate[a]
  + reliability(country, a)      * country_adjustment[x.country, a]
  + reliability(language, a)     * language_adjustment[x.language, a]
  + reliability(platform, a)     * platform_adjustment[x.platform, a]
  + reliability(device_type, a)  * device_adjustment[x.device_type, a]
  + reliability(color_scheme, a) * color_adjustment[x.color_scheme, a]
  + reliability(time_bucket, a)  * time_adjustment[x.hour_bucket, a]
```

Each adjustment is shrunk toward a safer parent estimate:

```text
smoothed_rate(segment, a)
  = (weighted_reward_sum(segment, a) + k * parent_rate[a])
    / (weighted_count_or_ESS(segment, a) + k)
```

For propensity datasets:

- Use assignment probabilities only for estimation weights or ESS.
- Prefer stabilized/clipped weights rather than raw `1 / p` everywhere.
- Let ESS, not raw row count, control shrinkage.
- Avoid trusting small cells even when the raw weighted mean is high.

For non-propensity near-uniform datasets:

- Treat assignment as approximately uniform randomized.
- Use ordinary counts for shrinkage.
- Still shrink sparse segments, especially `vega` and high-cardinality
  `referrer_id`.

For action-set changes:

- Score every known variant, but mask to `available_variants`.
- If an available variant has little or no history, fall back to parent/global
  priors rather than fabricating confidence.

### Why not just brute-force or use a huge model?

The benchmark can be run quickly, but brute force is the wrong mindset. The
submission needs to show a production-shaped policy learning approach. A very
flexible model can overfit noisy logged outcomes and overreact to rare
assignment cells. The causal issue is not compute. It is support, overlap, and
variance.

A tree/boosting model may still be useful later, but only if wrapped in:

- strict walk-forward training,
- variant-as-action scoring,
- propensity-aware weighting where appropriate,
- regularization,
- and fallback/shrinkage for low-support decisions.

### Production connection

In production we would not have hidden truth files. We would need:

- randomized exploration or controlled experimentation to maintain support,
- logged propensities for every decision,
- off-policy evaluation with IPS / SNIPS / doubly robust estimators,
- variance checks and confidence intervals,
- guardrails for sparse segments and new variants,
- monitoring for drift, action availability, and policy concentration.

The Fisher-information intuition fits here: the best logging policy is not
just the policy with the best immediate reward. It is also the policy that
collects information where future policy parameters are uncertain and
decision-relevant. That is probably beyond this take-home's implementation
scope, but it is central to productionizing the system.

## 9. Next steps

1. Implement `hierarchical_policy_v1`.
   - Global variant prior.
   - Single-feature segment tables for country, language, platform, device,
     color scheme, returning status, and time bucket.
   - Additive smoothing toward global variant rates.
   - ESS-based smoothing for propensity datasets.

2. Implement `hierarchical_policy_v2`.
   - Add selected interactions only where support is large enough, for example
     `platform x country`, `platform x device_type`, or `country x language`.
   - Use stricter shrinkage for `vega` and high-cardinality `referrer_id`.

3. Implement `advantage_policy`.
   - Model variant advantage over the best fixed/global baseline rather than
     raw conversion.
   - This better matches the business decision: not "who converts", but "which
     variant improves conversion for this context".

4. Compare policies in the harness.
   - Run quick mode first for iteration speed.
   - Run full walk-forward on all datasets.
   - Plot over-time convergence and dataset-specific failures.

5. Keep every policy registered.
   - The final submission should show the research path, not only the final
     number.

## 10. Current conclusion

The problem is best approached as causal policy learning from randomized logs:
estimate per-context action value, account for assignment mechanisms where
propensities are logged, and shrink hard wherever support is thin.

The strongest immediate signal is contextual segmentation. The strongest risk
is overfitting noisy action/context cells, especially in `helios` because of
20x inverse-propensity weights and in `vega` because the dataset is small.

The next implementation should be a leak-free hierarchical shrinkage policy
that uses the context signal found in the diagnostics while respecting
variance, overlap, and changing action sets.

## 11. Review-note triage

An external review raised several useful implementation and modeling warnings.
I checked the ones that depend on the actual harness.

### `score_variants` column order

This is a real silent-bug risk. `ScoredPolicy.recommend()` expects
`score_variants()` to return one column per `meta.variant_ids`, in exactly that
global metadata order. It then masks unavailable variants by comparing
`meta.variant_ids` to `available_variants` and returns
`meta.variant_ids[argmax]`.

Therefore, a policy must not return score columns in `available_variants` order.
That would silently score the wrong arms whenever the available action set is a
subset or is ordered differently from `meta.variant_ids`. The safe contract is:

```text
scores.shape == (len(contexts), len(meta.variant_ids))
scores[:, j] corresponds to meta.variant_ids[j]
```

The custom policy implementation should include this as a comment or assertion.

### Atlas, vega, and shrinkage

The warning about overfitting is directionally right:

- `atlas` has no measured headroom, so any apparent personalization is noise.
- `vega` has only 6,000 rows, so thin segment/action cells need heavy pooling.
- Per-row argmax over noisy cell estimates is dangerous unless estimates are
  shrunk toward a stable prior.

This strengthens the case for empirical-Bayes shrinkage as the default
guardrail rather than hard segment rules.

### Rotation oracle note

Verified: in `rotation`, stored `oracle_value` differs from the row-wise max of
the per-arm probabilities in 15.17% of rows.

Important correction: in this checkout, `engine.scoring.OracleScoring` loads
`truth.oracle` from the stored `oracle_value` column and uses that value in
`cumulative_oracle`. It does not recompute the row-wise max during scoring.

So the practical guidance is:

- do not train or calibrate a policy on truth files;
- if doing diagnostics, know that the official captured-headroom denominator is
  the harness's stored `oracle_value`, not a recomputed oracle;
- policy value for chosen variants is still scored from the per-arm probability
  of the chosen arm.

### T-learner vs S-learner

The review recommends a T-learner, but the current strongest low-risk path is
still the EB table:

- A T-learner gives explicit arm-specific response surfaces.
- An S-learner shares strength across arms by using `variant_id` as a feature.
- The EB table is a simpler, faster version of arm-specific surfaces with
  explicit shrinkage.

For this harness, the table should be the shipped workhorse. A T-learner or
GBM can be an experiment after the table establishes a strong, explainable
baseline.

### Propensity usage

The review says not to use propensities in the scored policy. I mostly agree
with the caution, but the reason should be stated carefully.

If the goal is to estimate `E[Y | X, A]` and assignment is randomized
conditional on observed context, outcome modeling does not require IPW for
unbiasedness under a correctly specified model. Raw IPW can add variance,
especially in `helios` where weights reach 20x and effective sample size falls
from 150,000 rows to about 39,818.

However, because our table conditions on coarser segments than the full context,
propensity can still be useful as a diagnostic or as clipped/stabilized
estimation support. The policy should not blindly use raw `1 / p` weights. The
candidate implementation should start unweighted or lightly stabilized, then
compare a propensity-aware variant in the harness.

### Recency / change-points

The review claims `zephyr` has a regime flip and needs adaptive time decay.
That is plausible from the baseline over-time curves, but it should be treated
as an experiment rather than a hard-coded assumption. A safe implementation is:

- default to expanding-window history;
- add a candidate policy with adaptive decay/change-point logic;
- keep decay dataset-agnostic in code, so the solution is not just memorizing a
  dataset name.

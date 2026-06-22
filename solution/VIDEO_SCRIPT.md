# Video Script — Coframe Personalization Take-Home
*Jonathan Gu · ~5–6 min · read naturally; [bracketed] lines are screen cues.*

---

## 0 · Cold open (~15s)
The question: can a policy pick a better variant per visitor than the single best
variant — and how much extra conversion does that capture? My answer: **yes** — the
final policy captures **~70% of the available headroom** on average, and does **no
harm** where there isn't any.

## 1 · Demo (~75s)
[terminal] Everything runs through the provided walk-forward harness.

[run] `uv run run_eval.py --policies random,my_policy,seg_eb,gbm_tlearner,hybrid_eb_gbm`

The metric is **captured headroom**: 0% = the best single variant (the A/B winner),
100% = the perfect per-visitor choice. [as it prints] random ≈ 0; empirical-Bayes
**62.7%**; gradient-boosted **65.4%**; the hybrid **69.6%** — each step beats the last
and the baseline.

[run] `uv run plot_results.py` → open `results/plots/summary.png`. There's the scoreboard.

## 2 · Solution & research log (~110s)
[report / research log] It's an **offline one-shot contextual bandit** — not RL. Each
impression is one decision: pick `argmax` over arms of `P(convert | context, arm)`.
We're scored against the true oracle, so **greedy is the right call** — there's no
exploration value when our picks don't generate new data.

I profiled the six datasets first. Four are clean equal-split A/B tests; two
(`helios`, `rotation`) log propensities. I **model reward directly** — `P(convert|x,a)`,
levels not contrasts — and argmax.

The ladder: best-fixed (~0, which proves the problem is contextual) → `seg_country`
(country EB, 43%) → `seg_eb` (empirical-Bayes additive T-learner, 62.7%) →
`gbm_tlearner` (boosted trees, 65.4%) → **`hybrid_eb_gbm` (69.6%, shipped)**.

The hybrid is the headline: it's a **per-window router** — empirical-Bayes when training
support is thin, gradient-boosted trees when it's large. That's *why* it wins: it takes
boosting's gains on the big datasets without paying its variance cost on sparse `vega`,
where it falls back to shrinkage. It gates on **support, not a dataset name**, so it's
leak-free.

Tools: built with **Claude Code across two parallel agent threads** that cross-checked
each other; the provided eval harness; my observability and arm-sensitivity scripts;
numpy/pandas, then scikit-learn and LightGBM.

## 3 · Edge cases (~80s)
Each dataset is a different edge case:
- **atlas — no headroom.** Every variant is identical, so shrinkage collapses to best-fixed; do no harm.
- **vega — sparse** (6k rows). Boosting overfits to 33%; the hybrid stays on EB at 55%.
- **zephyr — concept drift**; the best arm flips mid-stream. The boosted model's time feature tracks it (47% → 73%).
- **rotation — changing action set**; arms enter and leave. We mask to the available variants and prior-fill brand-new arms.
- **helios — randomized but not equal-split.** Variants were logged with **unequal probabilities** (some as low as 0.05), so raw arm averages are biased — an arm can look good because of *who* it was shown to. But assignment is random *conditional on context*, so within a comparable context bucket the visitors who saw each variant are exchangeable, and reward modeling is valid. We avoid raw IPW (0.05 → weight 20 → high variance) and keep propensities for overlap/variance diagnostics.
- **cold start** — the first window is empty; the policy abstains to random until data arrives.

## 4 · Productionization (~70s)
In production there's no oracle:
- **OPE:** IPS / SNIPS / doubly-robust with confidence intervals. helios' effective sample size drops ~150k → 40k under IPW — which is why you prefer doubly-robust over raw weights.
- **Exploration:** greedy is right for the score, but in production it starves the arms it disfavors, so you add Thompson / ε-greedy to keep overlap and keep OPE valid.
- **Retraining:** keyed to drift detection, not a fixed clock (zephyr is the example).
- **Serving:** the EB table is a microsecond lookup; the boosted model is one evaluation per arm (10× on rotation). The hybrid serves the cheap path most of the time.

This is the kind of system I built at Instacart for growth ML — causal targeting,
incrementality, off-policy evaluation.

## Close (~10s)
The repo has every policy I tried, the research log, the plots, and this write-up as a
hosted page. Thanks — happy to go deeper on any piece of it.

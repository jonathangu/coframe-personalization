# Video Script — Coframe Personalization Take-Home
*Jonathan Gu · target ~7 minutes · read naturally; bracketed lines are screen cues, not spoken.*

---

## 0 · Cold open (~20s)

Hi, I'm Jonathan. The question this take-home asks is the one a client asks right before they turn personalization on: **can a policy reliably pick a better variant per visitor than the single best variant — and how much extra conversion does that actually capture?**

Short answer from my work: **yes, reliably.** My final policy captures about **70% of the available headroom** on average, and on the one dataset where there's nothing to gain, it does **no harm**. Let me show you how I got there.

---

## 1 · Demo — run the harness (~90s)

[Screen: terminal in the repo]

Everything runs through the provided harness with `uv`. I'll run the eval across the ladder of policies I built:

[run] `uv run run_eval.py --policies random,best_fixed,seg_eb,gbm_tlearner,hybrid_eb_gbm`

The metric is **captured headroom**. Zero percent means you did as well as the best *single* variant — the A/B-test winner. A hundred percent means you matched the per-impression *oracle*, the perfect choice for every visitor. So it's a clean 0-to-100 scale between "don't personalize" and "personalize perfectly."

[as it prints] Watch `random` sit at zero — and watch the personalized policies climb. `seg_eb`, my empirical-Bayes model, lands around **63%**. The boosted-tree version, **65%**. And the hybrid, **~70%**.

[run] `uv run plot_results.py`  [open `results/plots/summary.png`]

Here's the scoreboard as a picture — every policy, every dataset. And the over-time plots show how each policy *learns* as training history grows.

---

## 2 · Solution & research log (~2.5 min)

[Screen: the hosted report / research log]

**First, what kind of problem is this.** It's an **offline, one-shot contextual bandit** — not reinforcement learning. There's no state, no sequence of actions, no exploration in the scored task. Each impression is one independent decision: given the visitor's context, pick the variant with the highest expected conversion. The optimal policy is just `argmax over arms of P(convert | context, arm)`. And because we're scored against the *true* oracle, greedy exploitation is provably optimal — exploration is a production concern, not a scoring lever.

**Then I profiled the data.** Six datasets. Four are clean **equal-split A/B tests**. Two — helios and rotation — ship **propensities**: helios assigns variants with unequal, logged probabilities, and rotation rotates its action set over time. I verified empirically that assignment is independent of the visitor's features. That matters: by Rosenbaum–Rubin, since we condition on the full context in our model, **the propensity is redundant for the score** — and inverse-propensity weighting would only add variance, exactly the high-variance trap when some probabilities are as low as 0.05.

**How I specify the reward.** Conversion is binary. I model `P(convert | context, arm)` directly — the levels, not an uplift contrast — and take the argmax. No contrast is needed because the per-visitor baseline cancels in the argmax.

**The model ladder — this is the iteration story:**
- `random` and `best_fixed` — the baselines. Best-fixed is basically worthless here, which proves the problem is genuinely contextual.
- `seg_country` — an empirical-Bayes table on just the country signal. Already ~43%.
- `seg_eb` — the full **empirical-Bayes additive T-learner**: per-arm conversion surfaces, shrunk toward stable priors, with a few country interactions. **62.7%.** Leak-free, interpretable, and the shrinkage is what protects sparse data.
- `gbm_tlearner` — **gradient-boosted trees**, one per arm. I tested whether the additive form was leaving signal on the table. It was: GBM adds about 9 points on the big datasets by capturing interactions automatically. **65.4%.**
- `hybrid_eb_gbm` — the winner. **Support-aware**: empirical-Bayes when data is thin, boosted trees when support is large. **69.6%.**

The reason for the hybrid is a **sensitivity analysis**: I measured how often the different functional forms even *agree* on which arm to show, and whether their disagreements cost conversion. GBM's disagreements *add* value where data is rich, *destroy* it on sparse vega, and are completely *benign* on atlas. So the right answer isn't "pick a model" — it's "pick the model the data can support."

[Tools] I built this with **Claude Code**, running a primary and a parallel agent, and cross-checked findings between them — for example, the parallel track caught a subtle detail about how the harness computes its oracle denominator.

---

## 3 · Edge cases (~90s)

[Screen: edge-cases section / per-dataset plots]

Each dataset is really a different edge case, and the policy handles each deliberately:
- **Cold start** — the first window has no training data, so the policy abstains to random until data arrives.
- **atlas — no headroom.** Every variant is identical, so there's nothing to personalize. The empirical-Bayes shrinkage collapses every arm to the same estimate, so we fall back to best-fixed and **do no harm**. You literally can't lose there.
- **vega — sparse.** Six thousand rows, about one row per cell early on. This is where boosted trees overfit, and the hybrid correctly falls back to the shrinkage model.
- **helios — biased logging.** A confounder that's a function of the observed features; conditioning on context removes the bias, and we ignore the propensity.
- **rotation — changing action set.** Arms enter and leave over time. We score all arms, mask to the ones available, and give brand-new arms a prior until they have data.
- **zephyr — concept drift.** The best arm flips halfway through the timeline. Recency-weighting recovers it — but I showed that a *global* recency setting hurts the stationary datasets, so the production answer is **adaptive** recency, gated by drift detection.

---

## 4 · Productionization (~90s)

In production there's **no oracle**, and that's where everything I skipped for the score comes back:
- **Off-policy evaluation** — you'd measure policies with IPS, SNIPS, or **doubly-robust** estimators, with confidence intervals. Helios is the cautionary tale: its effective sample size drops from 150,000 rows to about 40,000 under inverse-propensity weighting, which is exactly why you reach for doubly-robust over raw IPW.
- **Exploration vs. exploitation** — greedy is optimal for *this* score, but in production greedy starves the arms it disfavors, so future models degrade. You explore — Thompson sampling or epsilon-greedy — to keep overlap alive. That's the Fisher-information, value-of-information idea: collect data where the policy is uncertain *and* it matters.
- **Retraining cadence** — keyed to drift detection, not a hard-coded schedule. Zephyr shows why.
- **Serving latency** — the empirical-Bayes table is a hash lookup, microseconds per request; a per-arm boosted model is N model evaluations per impression — ten times on rotation. The hybrid is mostly the cheap path, which is a real production virtue.

This is squarely the kind of system I built at Instacart for growth ML — causal targeting from randomized logs, incrementality, budgeted allocation, and off-policy evaluation — so the productionization story here is the one I'd actually ship.

---

## Close (~15s)

The repo has **every policy I tried**, the research log, the observability plots, and this whole write-up as a hosted page. Thanks for watching — I'm happy to go deeper on any piece of it.

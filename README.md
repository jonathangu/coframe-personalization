# Coframe Personalization Take-Home — Solution

**Author:** Jonathan Gu · **Live report:** https://jonathangu.github.io/coframe-personalization/

> Can a policy reliably pick a better variant per visitor than the single best
> variant — and how much extra conversion does that capture? **Yes.** The shipped
> policy captures **69.6% of the available headroom** on average, and does **no
> harm** on the one dataset where there is none.

This is an **offline, one-shot contextual bandit**: for each visitor's context,
choose the variant with the highest expected conversion, learned from logged
randomized history, evaluated walk-forward against hidden per-variant truth.

## Result — verified full walk-forward (captured headroom %)

| policy | helios | meadow | rotation | vega | zephyr | **avg** |
|---|---|---|---|---|---|---|
| `random` | −0.4 | 0.2 | −0.1 | 1.3 | 0.1 | **0.2** |
| `example` (provided) | 19.8 | 26.2 | 13.0 | 13.8 | 18.9 | **18.3** |
| `seg_country` | 55.6 | 47.0 | 41.6 | 42.3 | 27.2 | **42.7** |
| `seg_eb` (empirical-Bayes T-learner) | 77.3 | 72.2 | 61.2 | 55.3 | 47.4 | **62.7** |
| `gbm_tlearner` (gradient-boosted) | 78.3 | 78.7 | 61.4 | 32.7 | 75.9 | **65.4** |
| **`hybrid_eb_gbm`** (shipped) | **79.7** | 78.4 | **61.9** | **55.3** | 72.7 | **69.6** |

`0%` = best single variant (the A/B winner); `100%` = the per-impression oracle.
`atlas` is omitted — it has **no headroom** (every variant is identical), so the
policy correctly does nothing there.

**The shipped policy is support-aware:** an empirical-Bayes shrinkage table when a
segment is data-thin (which protects sparse `vega`, where pure boosting overfits to
32.7%), switching to gradient-boosted trees when support is large (which captures
the interaction and drift signal on the big datasets).

## Read this first

- **[`solution/report.html`](solution/report.html)** — the full write-up (also the [live page](https://jonathangu.github.io/coframe-personalization/)).
- **[`solution/research_log.md`](solution/research_log.md)** — research log: what was tried, what drove each decision, what was rejected, and every AI tool used.
- **[`solution/DELIVERABLES_COVERAGE.md`](solution/DELIVERABLES_COVERAGE.md)** — maps each required topic to where it's answered.
- **[`solution/VIDEO_SCRIPT.md`](solution/VIDEO_SCRIPT.md)** — the video walkthrough script.
- **[`solution/CHAT_TRANSCRIPT.md`](solution/CHAT_TRANSCRIPT.md)** — how the work was driven with AI.

## Run it

```bash
# uv provisions Python + deps automatically (numpy/pandas, scikit-learn, lightgbm)
uv run run_eval.py --policies random,my_policy,seg_eb,gbm_tlearner,hybrid_eb_gbm
uv run plot_results.py
uv run python solution/observability.py        # data displays
uv run python solution/eval_observability.py   # harness/eval dashboards
uv run python solution/arm_sensitivity.py      # arm-choice sensitivity (truth-only diagnostic)
```

Every policy is registered in [`solution/__init__.py`](solution/__init__.py).

## Repo map

| path | what |
|---|---|
| `solution/eb_policy.py` | empirical-Bayes additive T-learner (`seg_country`, `seg_eb`, `seg_eb_recency`) |
| `solution/gbm_policy.py` | gradient-boosted T-learner + support-aware hybrid (`gbm_tlearner`, `hybrid_eb_gbm`) |
| `solution/lgbm_policy.py` | LightGBM backend (`lgbm_tlearner`, `hybrid_eb_lgbm`) |
| `solution/observability.py`, `eval_observability.py`, `arm_sensitivity.py` | display/diagnostic harnesses |
| `solution/results/` | generated `summary.json`, `eval.log`, plots, and observability figures |
| `engine/`, `policies/`, `run_eval.py`, `plot_results.py` | the provided harness (see `README_HARNESS.md`) |
| `data/` | logged datasets + hidden truth files |

The evaluation harness is Coframe's, preserved in
[`README_HARNESS.md`](README_HARNESS.md) and unmodified; everything under
`solution/` is the submission.

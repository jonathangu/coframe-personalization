# Deliverables Coverage — the four required video topics

A checklist mapping every required sub-point to where it is answered (file ·
section · figure) and the one-line talking point. Use this as the video prep sheet.

Shipped policy: **`hybrid_eb_gbm` — 69.6% average captured headroom** (support-aware:
empirical-Bayes when data is thin, gradient-boosted trees when support is large).

---

## 1 · Demo — run the harness and walk the eval plots

| Sub-point | Where | Talking point |
|---|---|---|
| Run the harness | `VIDEO_SCRIPT.md` §1 · `report.html` Runbook | `uv run run_eval.py --policies random,my_policy,seg_eb,gbm_tlearner,hybrid_eb_gbm` |
| Policies side by side | `report.html` Results table (all 12) · `results/eval_observability/policy_scoreboard_heatmap.png` | one row per policy, per dataset |
| Each step beat the last **and** random | Results table avg column · `results/eval_observability/captured_convergence_grid.png` | ladder climbs 0.2 → 18.3 → 42.7 → 62.7 → 65.4 → **69.6** |
| Read the captured-headroom report as you go | `results/eval.log` · `results/summary.json` | the printed per-dataset table + averages |
| Render plots | `uv run plot_results.py` → `results/plots/summary.png` + `<dataset>_over_time.png` | |
| Loom preferred | `VIDEO_SCRIPT.md` is the read-aloud Loom script | |

## 2 · Solution & research log — what you built and how you iterated

| Sub-point | Where | Talking point |
|---|---|---|
| What you built | `research_log.md` §0 · `report.html` Implementation + "What kind of ML model" | direct reward model `μ(x,a)=P(convert\|x,a)`, argmax; EB → GBM → hybrid |
| How you iterated | `research_log.md` §0 ladder + chronological §1–§11 | each rung fixes the prior's failure |
| How you profiled the datasets | `results/observability/*` (assignment_mix, propensity_diagnostics, segment_signal_heatmap, temporal_*) · `research_log.md` §3–§6 | randomization, propensity, signal, support, drift |
| What drove each decision | `research_log.md` §0 "What drove each decision" | why EB, why GBM, why hybrid, the arm-sensitivity check |
| What you rejected | `research_log.md` §0 "What was rejected, and why" | raw IPW, uplift contrasts, pure GBM, global recency, untuned LightGBM |
| **Every AI tool and test harness** | `research_log.md` §0 "Every AI tool…" · `CHAT_TRANSCRIPT.md` | Claude Code (2 parallel threads + a workflow of subagents + a review subagent); the provided `run_eval.py`/`plot_results.py`; custom `observability.py`/`eval_observability.py`/`arm_sensitivity.py`; numpy/pandas → scikit-learn → LightGBM; `uv` |

## 3 · Edge cases — which datasets exhibit them, and how the policy holds up

| Edge case | Dataset that exhibits it | How the policy holds up | Where |
|---|---|---|---|
| Cold start | every dataset's first window (empty train) | abstain → uniform random until data arrives | `report.html` Edge cases · `eb_policy.py` |
| Sparse / underpowered | **vega** (6k rows, ~1 row/cell early) | heavy EB shrinkage; hybrid falls back to EB (55.3% vs pure-GBM 32.7%) | Edge cases · `arm_sensitivity` |
| Non-stationarity | **zephyr** (best arm flips at t≈0.5) | recency weighting recovers it (+11.5pp); made adaptive, not global | Edge cases · `<zephyr>_over_time.png` |
| Changing action sets | **rotation** (arms 07/08 leave, 09/10 enter) | score all arms, mask to available, prior for brand-new arms | Edge cases · `action_availability.png` |
| Do no harm (no headroom) | **atlas** (every arm identical) | EB collapses all arms to one estimate → best-fixed; captured% n/a, can't lose | Edge cases |
| Non-uniform assignment (bonus) | **helios** (skewed shares; propensities to 0.05) | raw arm averages are biased, but reward modeling is valid conditional on context; avoid raw IPW (0.05 → weight 20), keep propensity for diagnostics/OPE | Randomization section · `propensity_diagnostics.png` |

> "Part of the exercise is discovering which datasets exhibit these" — we discovered
> them by profiling (the observability plots), then designed the policy around them.

## 4 · Productionization — evolving to a real deployment

| Sub-point | Where | Talking point |
|---|---|---|
| OPE when counterfactuals are unavailable | `report.html` Productionization + "What is next" · `research_log.md` §8 | IPS / SNIPS / **doubly-robust** + confidence intervals; helios ESS drops 150k → ~40k under IPW → why DR over raw IPW |
| Exploration vs. greedy exploitation | same | greedy is optimal for *this* score; in prod, explore (Thompson / ε-greedy) to keep overlap — Fisher-information / value-of-information |
| Retraining cadence | same | keyed to **drift detection** (zephyr), adaptive recency, not a hard-coded schedule |
| Serving latency | `results/eval_observability/policy_runtime.png` | EB table = microsecond hash lookup; GBM = `n_arms` model evals/impression (10× on rotation); the hybrid is mostly the cheap path |

---

### Where each lives
- **`report.html`** (this folder) — the full evaluator-facing write-up; open via the hosted page or locally.
- **`research_log.md`** — the research log (final summary §0 + chronological notes).
- **`VIDEO_SCRIPT.md`** — the read-aloud video script, structured to the four topics above.
- **`CHAT_TRANSCRIPT.md`** / **`CHAT_TRANSCRIPT_FOR_AGENT.md`** — how the work was driven (AI collaboration).
- **`results/`** — `summary.json`, `eval.log`, `plots/`, `observability/`, `eval_observability/`, `arm_sensitivity/`.

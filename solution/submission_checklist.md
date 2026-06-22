# Submission Checklist

## Code deliverable

Submit the completed `solution/` package in a public GitHub repository.

The package should include:

- policy implementations in `solution/`;
- every policy tried, registered in `solution/__init__.py`;
- generated evaluation outputs under `solution/results/`;
- comparison plots under `solution/results/plots/`;
- data observability plots under `solution/results/observability/`;
- eval observability plots under `solution/results/eval_observability/`;
- a short research log explaining what was tried, what worked, what failed, and
  why.

Current supporting files:

- `research_log.md` - strategy, data notes, randomization, variance, and next steps.
- `observability.py` - script that regenerates the data observability plots.
- `results/observability/README.md` - visual index of data displays.
- `eval_observability.py` - script that regenerates harness/eval dashboards.
- `results/eval_observability/README.md` - visual index of eval displays.

## Video deliverable

Record a walkthrough covering:

1. Demo: run the eval harness, show `summary.json`, `eval.log`, and plots.
2. Data: explain assignment, propensities, support, variance, and action-set changes.
3. Policy: explain the final policy and the policies tried along the way.
4. Results: compare final policy against `random`, `my_policy`, and intermediate policies.
5. Edge cases: cold start, `atlas` no-headroom, `vega` sparsity, `helios`
   propensities, `rotation` changing arms, and any detected non-stationarity.
6. Productionization: OPE, exploration, propensities, monitoring, retraining, and
   serving latency.

## Regeneration commands

```bash
uv run run_eval.py --policies random,my_policy,example,feature_demo
uv run plot_results.py
uv run python solution/observability.py
uv run python solution/eval_observability.py
```

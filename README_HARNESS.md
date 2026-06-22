# Coframe Applied Science — Take-Home

## Background

Coframe ships winning web experiences for clients. An A/B test finds the single best variant *for everyone* — but different visitors respond to different things. The next lever is **personalization**: given who is visiting (their context), choose the variant most likely to convert *for them*.

The catch is that you learn from a **log of past decisions**, not a clean experiment. Each row records the context of an impression, the variant that was shown, and whether it converted — and you only ever see the outcome of the variant that *was* shown, never the **counterfactual**: what each of the *other* variants would have returned for that same visitor. The question this take-home asks is the one a client asks before turning personalization on: *can a policy reliably pick a better variant per visitor than the best single variant — and how much extra conversion does that actually capture?*

---

## Your task

Build a **personalization policy**: given a visitor's context, choose which variant to show, learned from the logged history.

The harness, the data, and the evaluation are all provided. The baseline to beat is the **`random` policy**; a runnable but non-personalizing **`my_policy` starter** ships in `solution/` for you to build on. Your job: capture as much of the available personalization headroom as you can, and do **no harm** on traffic where personalization cannot help.

You build and submit your work in the **`solution/`** package (see the developer workflow below) — commit and push the policies you tried (not just the final one) plus a short write-up of how you got there.

**How you build it is up to you.** What matters is that the policy is **leak-free** (it learns only from the past) and that you can show, honestly, how much extra conversion it captures.

---

## Quickstart

With [uv](https://docs.astral.sh/uv/) (it provisions the right Python and all dependencies for you; nothing else to install):

```bash
uv run run_eval.py                              # all datasets, the random baseline
uv run run_eval.py --quick                      # fast loop on a small subset
uv run run_eval.py --policies random,my_policy  # baseline + the starter in solution/
uv run plot_results.py                          # render comparison plots to solution/results/plots/
```

The first `uv run` creates `.venv`, installs the dependencies, and pins Python (see `.python-version`).

`run_eval.py` writes, under `solution/results/` (inside the package you submit — nothing to move before you commit):

- `solution/results/<dataset>/<policy>.jsonl` — one file **per policy**, one line per iteration, with the captured-headroom metric and per-iteration train/inference timing. Per-policy files are independent, so re-running one policy never clobbers another's results.
- `solution/results/summary.json` — final scores + speed per `(policy, dataset)` and the per-policy average across datasets.
- `solution/results/eval.log` — the per-dataset and average score report (also printed).

`plot_results.py` plots every policy it finds; pass `--policies` to select a subset (e.g. `--policies random,my_policy`).

---

## How you're scored: captured headroom

Every dataset ships with the **true conversion probability of every variant** for every row (`data/<name>_truth.parquet`). The harness scores each recommendation against this hidden ground truth — the per-impression **oracle** — rather than a noisy off-policy estimate, so the signal is clean:

```
captured% = (policy_cvr - best_fixed) / (oracle_cvr - best_fixed) * 100
```

- **0%** — the best *non-personalized* variant (the A/B-test winner).
- **100%** — the per-impression oracle (always show each row's best variant).
- **NaN** — a dataset where the best fixed variant already *is* the per-impression oracle, so there is no headroom to capture and the ratio is undefined; a number there would mean the metric broke, not that the dataset was solved. (You'll see it in the score table once you run — figuring out what it implies is part of the task.)

`policy_cvr` and `oracle_cvr` are **true** expected conversion rates from the hidden per-variant probabilities, so the score is exact (up to the dataset's own sampling), not an off-policy guess.

---

## The data

Each dataset is a log of past decisions over one shared web-visitor schema: feature columns describing the visitor and context (`platform`, `device_type`, `country`, `language`, `referrer_id`, `color_scheme`, `is_returning`, `visits_count`, `days_since_last_visit`, `timestamp`), the shown `variant_id`, the binary `reward`, and — on some datasets — a `propensity`.

---

## Evaluation: walk-forward (the way production works)

The timeline of each dataset is split into chronological **iterations**. For each:

1. **train** your policy on everything *before* the iteration (the past),
2. **recommend** a variant for every impression *in* the iteration (the unseen future),
3. **score** those recommendations against the hidden truth, then fold the window into history and repeat.

Training always precedes the data it is judged on — no leakage. The first iteration is a **true cold start** (empty training history). Window sizes grow geometrically at first (so you can watch convergence as `n_train` goes from ~30 rows to thousands), then become equal-width — plot quality against `n_train` to read convergence speed. The **action set can change over time**: `recommend` is given the variants live in each window and must only return those.

---

## Developer workflow — iterate on policy & features

Your work goes in the **`solution/`** package — that's the folder you commit and push. It ships with a runnable starter (`my_policy`, a non-personalizing baseline). The two baselines that ship with the harness are `random` and the best-fixed-variant bar. Improve from there:

1. **Write your policy** in `solution/policy.py` (or add more modules). Copy `policies/example_policy.py` — a worked personalized policy — as a template. Implement either `Policy.fit` + `Policy.recommend`, or `ScoredPolicy.fit` + `ScoredPolicy.score_variants` (you get `recommend` — argmax over the available action set, random cold start — for free).
2. **Register it** in `solution/__init__.py` (already wired for `my_policy`):
   ```python
   from solution.policy import MyPolicy
   register("my_policy", MyPolicy)
   ```
   The harness auto-imports `solution/`, so registered policies are immediately available to `run_eval.py` and `plot_results.py`.
3. **Run & compare**:
   ```bash
   uv run run_eval.py --policies random,my_policy --quick   # fast feedback
   uv run run_eval.py --policies random,my_policy           # full protocol
   uv run plot_results.py --policies random,my_policy
   ```
   Need an extra package? `uv add <name>` adds it to your solution's dependencies.

**Keep the variants you tried.** Register as many policies as you like — add a module per approach and `register(...)` each under its own name in `solution/__init__.py`, then compare them in one run (`uv run run_eval.py --policies random,my_policy,my_policy_v2`). We'd rather see the approaches you explored across the challenge than only the final one.

### Feature engineering without lookahead

`engine/features.py` gives a `FeaturePipeline` contract that splits *what you learn from data* (`fit`, on training history only) from *how a row becomes features at serve time* (`transform`, a pure, row-local function of fitted state). Anything data-dependent — vocabularies, frequencies, scalers, target stats — must be learned in `fit`, never from the frame you are scored on. Writing your own is small: subclass `FeaturePipeline` and implement `fit` + `transform` (plus optional `categorical_outputs`) — see `solution/feature_demo.py` for a worked pipeline that a policy actually consumes. The harness only ever hands `recommend`/`transform` the feature columns + timestamp — never `reward`, `variant_id`, or `propensity`.

---

## Layout

```
personalization/
  README.md  .gitignore
  pyproject.toml        uv project + dependencies; requires Python 3.11+
  uv.lock               pinned dependency graph (reproducible `uv run` / `uv sync`)
  .python-version       the Python uv provisions for this project
  data/                 <name>.parquet (logged log) + <name>_truth.parquet (per-variant truth)
  engine/               the isolated engine
    datasets.py         DatasetMeta, load_dataset, list_datasets
    features.py         FeaturePipeline contract (subclass it) + clean_categorical helper
    policy.py           Policy / ScoredPolicy + random / best-fixed baselines
    evaluation.py       walk_forward protocol (cold start, expanding history)
    scoring.py          oracle scoring -> captured-headroom %
  policies/
    __init__.py         the registry (register / get_policy); ships {random, example}, auto-imports solution/
    example_policy.py   a worked personalized policy (registered as `example`); copy as a template
  solution/             <- YOUR submission (commit & push this)
    __init__.py         registers your policies with the harness (one or many)
    policy.py           the `my_policy` starter — add a module per policy as you iterate
    results/            (generated) eval output, committed with your submission: <dataset>/<policy>.jsonl + summary.json + eval.log + plots/
  run_eval.py           run the eval -> writes jsonl + summary.json + logs to solution/results/
  plot_results.py       matplotlib over-time + comparison plots -> solution/results/plots/
```

## One jsonl line (`solution/results/<dataset>/<policy>.jsonl`)

```json
{
  "dataset": "meadow",
  "policy": "random",
  "iteration": 12,
  "window_start": "...",
  "window_end": "...",
  "n_train": 18421,
  "n_window": 3000,
  "train_seconds": 0.0,
  "inference_seconds": 0.01,
  "raw_cvr": 0.081,
  "cumulative_cvr": 0.080,
  "raw_oracle": 0.179,
  "cumulative_oracle": 0.179,
  "cumulative_best_fixed": 0.080,
  "cumulative_captured_pct": 0.4,
  "available_variants": ["variant_1", "..."],
  "new_variants": [],
  "choice_violations": 0,
  "recommendation_mix": {
    "variant_1": 0.25,
    "...": 0.25
  }
}
```

---

## Deliverables

Email back a **recorded video** and your **code**.

Your code is the `solution/` package — include the **policies you tried across the challenge** (register each under its own name), not just the final one. Alongside it, share a short **write-up / research log** in whatever form suits you (Markdown, a notebook, inline notes): what you tried, what you measured on the data, what you learned, and what you discarded and why. We care as much about *how you got there* as the final number.

The video should cover:

1. **Demo** — run the harness and walk us through the **eval plots**: put the policies you iterated on side by side (`uv run run_eval.py --policies random,my_policy,my_policy_v2`, then `uv run plot_results.py`) and show how each step improved on the last and on the `random` baseline, reading the captured-headroom report as you go. Loom demo format is preferred.
2. **Solution & research log** — what you built and **how you iterated to it**: a walk through your research log — how you profiled the datasets, what you tried, and what drove each decision. Include every AI tool and test harness you used during the build.
3. **Edge cases** — the edge cases you found in the data and how your policy holds up on each (e.g. the cold start, sparse/underpowered data, non-stationarity, changing action sets, and doing no harm where there's no headroom). Part of the exercise is discovering which datasets exhibit these.
4. **Productionization** — how this evolves to a real deployment: off-policy evaluation when counterfactuals are *not* available, exploration vs. greedy exploitation, retraining cadence, and serving latency.

A **public GitHub repository is preferred** for the code. A gist or zip is acceptable if that isn't possible.

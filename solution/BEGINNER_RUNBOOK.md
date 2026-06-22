# Beginner Runbook — How to Run This (step by step)

Written for someone who has never run this kind of code. Follow it top to bottom.
Everything happens in the **Terminal** app on a Mac.

---

## What you need first (one-time)

This project uses **`uv`** — a tool that automatically installs the right Python and
all the libraries for you. Check if you have it:

```bash
uv --version
```

- If you see a version number (e.g. `uv 0.5.x`), you're set.
- If you see `command not found`, install it once with:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  then **close and reopen Terminal** and run `uv --version` again.

You do **not** need to install Python, numpy, scikit-learn, or anything else by hand —
`uv` does it the first time you run a command (that first run is a little slower while it
downloads things; later runs are fast).

---

## Step 1 — Go to the project folder

Copy-paste this and press Enter:

```bash
cd /Users/guclaw/repos/applied-science-take-home/personalization
```

You're now "inside" the project. (To double-check, run `ls` — you should see folders like
`engine`, `solution`, `data`, and files like `run_eval.py`.)

---

## Step 2 — Run the evaluation

This trains every policy and scores it. Copy-paste the whole line:

```bash
uv run run_eval.py --policies random,my_policy,best_fixed,example,feature_demo,seg_country,seg_eb,seg_eb_recency,gbm_tlearner,hybrid_eb_gbm --workers 1 --threads 2
```

**What to expect:**
- It prints a block per dataset (`atlas`, `helios`, …) as it goes.
- It takes a **few minutes** (the gradient-boosted policies are the slow part — that's
  normal; you'll see `gbm_tlearner` and `hybrid_eb_gbm` take the longest).
- When it finishes you'll see a big table titled **`CAPTURED HEADROOM`**, ending with an
  **`AVERAGE`** block. The last line should read about:

  ```text
  hybrid_eb_gbm  mean captured=  69.6%  (n=5 datasets)
  ```

  That's the headline result: the shipped policy captured **69.6%** of the available
  personalization headroom.

> **Faster version (optional, ~1–2 min)** — if you just want a quick look, run fewer
> policies:
> ```bash
> uv run run_eval.py --policies random,seg_eb,gbm_tlearner,hybrid_eb_gbm
> ```

**How to read a row** (e.g. `helios   hybrid_eb_gbm   79.7%   17.04%   8.00%   19.34%`):
- `captured = 79.7%` → captured 79.7% of the gap between "best single variant" and "perfect".
- `value = 17.04%` → the policy's true conversion rate.
- `bestfix = 8.00%` → the best single variant (the A/B-test winner) — this is the **0%** line.
- `oracle = 19.34%` → the perfect per-visitor choice — this is the **100%** line.
- `atlas` shows `n/a` on purpose: every variant is identical there, so there's nothing to
  personalize (the policy correctly does no harm).

---

## Step 3 — Make the plots

```bash
uv run plot_results.py
```

This draws the comparison charts from the run you just did.

---

## Step 4 — Find the outputs

Everything lands in `solution/results/`:

| file | what it is |
|---|---|
| `solution/results/eval.log` | the full printed report (the `CAPTURED HEADROOM` table) |
| `solution/results/summary.json` | the scores as data |
| `solution/results/plots/summary.png` | **the one-glance scoreboard** (all policies, all datasets) |
| `solution/results/plots/<dataset>_over_time.png` | how each policy learns as data grows |

Open the main scoreboard image:

```bash
open solution/results/plots/summary.png
```

---

## Step 5 — (Optional) regenerate the report's figures

These rebuild the data displays and the eval dashboards used in the HTML report:

```bash
uv run python solution/observability.py        # data displays
uv run python solution/eval_observability.py   # harness / eval dashboards
uv run python solution/arm_sensitivity.py      # arm-choice sensitivity diagnostic
```

---

## Step 6 — Open the full write-up

- **Live (nicest):** <https://www.jonathangu.com/coframe-personalization/>
- **Locally:**
  ```bash
  open solution/report.html
  ```

---

## For the Loom recording

A clean 60–90 second demo:

1. Show the Terminal. Say: *"This runs through the provided walk-forward harness."*
2. Run:
   ```bash
   uv run run_eval.py --policies random,my_policy,seg_eb,gbm_tlearner,hybrid_eb_gbm
   uv run plot_results.py
   ```
3. As the `CAPTURED HEADROOM` table prints, read the ladder out loud:
   *"random ≈ 0%, the empirical-Bayes policy 62.7%, gradient-boosted 65.4%, and the
   support-aware hybrid 69.6% — each step beats the last and the random baseline."*
4. Open the scoreboard:
   ```bash
   open solution/results/plots/summary.png
   ```
   Say: *"The strongest verified policy is the support-aware hybrid — empirical-Bayes when
   data is thin, gradient-boosted trees when there's enough support."*

(The full script is in `VIDEO_SCRIPT.md`.)

---

## If something goes wrong

- **`uv: command not found`** → install uv (top of this file), reopen Terminal.
- **`No such file or directory`** → you're not in the project folder; re-run the `cd` in Step 1.
- **It seems stuck** → it isn't; the gradient-boosted policies just take a couple of
  minutes. Let it finish until you see the `CAPTURED HEADROOM` table.
- **Want it faster** → use the shorter `--policies random,seg_eb,gbm_tlearner,hybrid_eb_gbm`.

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

Record = dict[str, Any]
Results = dict[str, dict[str, list[Record]]]

def _read_results(
    results_dir: Path,
    datasets: list[str] | None,
    policies: set[str] | None,
) -> Results:
    out: Results = {}
    for jsonl in sorted(results_dir.glob("*/*.jsonl")):
        ds, policy = jsonl.parent.name, jsonl.stem
        if datasets and ds not in datasets:
            continue
        if policies and policy not in policies:
            continue
        recs = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        recs.sort(key=lambda r: r["iteration"])
        if recs:
            out.setdefault(ds, {})[policy] = recs
    return out

def _plot_over_time(ds: str, by_policy: dict[str, list[Record]], out_dir: Path) -> Path:
    fig, (ax_cvr, ax_cap) = plt.subplots(1, 2, figsize=(14, 5))
    ref = next(iter(by_policy.values()))
    iters = [r["iteration"] for r in ref]
    ax_cvr.plot(
        iters, [r["cumulative_oracle"] * 100 for r in ref],
        color="#2ca02c", ls=":", lw=1.8, label="oracle (100%)",
    )
    ax_cvr.plot(
        iters, [r["cumulative_best_fixed"] * 100 for r in ref],
        color="#d62728", ls="--", lw=1.8, label="best fixed (0%)",
    )
    for policy, recs in sorted(by_policy.items()):
        xs = [r["iteration"] for r in recs]
        cvr = [r["cumulative_cvr"] * 100 for r in recs]
        cap = [r["cumulative_captured_pct"] for r in recs]
        ax_cvr.plot(xs, cvr, marker="o", ms=3, label=policy)
        ax_cap.plot(xs, cap, marker="o", ms=3, label=policy)
    ax_cvr.set_title(f"{ds}: cumulative true CVR")
    ax_cvr.set_ylabel("CVR (%)")
    ax_cap.set_title(f"{ds}: captured headroom")
    ax_cap.set_ylabel("captured (%)")
    ax_cap.axhline(0, color="#d62728", ls="--", lw=1.2)
    ax_cap.axhline(100, color="#2ca02c", ls=":", lw=1.2)
    if iters:
        ax_cap.set_xlim(iters[0], iters[-1])
    for ax in (ax_cvr, ax_cap):
        ax.set_xlabel("iteration (chronological)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if all(math.isnan(r["cumulative_captured_pct"]) for r in ref):
        ax_cap.text(
            0.5, 0.5, "no personalization headroom\n(captured % undefined)",
            ha="center", va="center", transform=ax_cap.transAxes, fontsize=11, color="gray",
        )
    fig.tight_layout()
    path = out_dir / f"{ds}_over_time.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

def _final_captured(
    results: Results,
) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    policies = sorted({p for bp in results.values() for p in bp})
    final: dict[tuple[str, str], float] = {}
    headroom_ds: list[str] = []
    for ds, bp in results.items():
        caps = {p: recs[-1]["cumulative_captured_pct"] for p, recs in bp.items() if recs}
        if caps and not all(math.isnan(v) for v in caps.values()):
            headroom_ds.append(ds)
        for p, v in caps.items():
            final[(ds, p)] = v
    return sorted(headroom_ds), policies, final

def _plot_summary(results: Results, out_dir: Path) -> Path:
    datasets, policies, final = _final_captured(results)
    fig, (ax_bar, ax_speed) = plt.subplots(
        1, 2, figsize=(max(11, 1.4 * (len(datasets) + 1) * len(policies)), 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    groups = [*datasets, "average"]
    width = 0.8 / max(len(policies), 1)
    x = np.arange(len(groups))
    for i, policy in enumerate(policies):
        vals = [final.get((ds, policy), float("nan")) for ds in datasets]
        avg = float(np.nanmean(vals)) if any(not math.isnan(v) for v in vals) else float("nan")
        heights = [*vals, avg]
        ax_bar.bar(x + i * width, heights, width, label=policy, edgecolor="k", linewidth=0.4)
    ax_bar.axhline(0, color="#d62728", ls="--", lw=1.2, label="best fixed (0%)")
    ax_bar.set_xticks(x + width * (len(policies) - 1) / 2)
    ax_bar.set_xticklabels(groups, rotation=20, ha="right")
    ax_bar.set_ylabel("captured headroom (%)")
    ax_bar.set_title("Final captured headroom (higher = better personalization)")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.3)

    train_s: dict[str, list[float]] = defaultdict(list)
    for bp in results.values():
        for policy, recs in bp.items():
            train_s[policy].extend(r["train_seconds"] for r in recs)
    pol = sorted(train_s)
    ax_speed.bar(
        np.arange(len(pol)),
        [float(np.mean(train_s[p])) for p in pol],
        color="#7f7f7f", edgecolor="k", linewidth=0.4,
    )
    ax_speed.set_xticks(np.arange(len(pol)))
    ax_speed.set_xticklabels(pol, rotation=20, ha="right")
    ax_speed.set_ylabel("mean train seconds / iteration")
    ax_speed.set_title("Training speed")
    ax_speed.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--results", default=str(HERE / "solution" / "results"), help="results directory",
    )
    ap.add_argument("--out", default=None, help="output dir (default: <results>/plots)")
    ap.add_argument("--datasets", default=None, help="comma-separated subset (default: all)")
    ap.add_argument(
        "--policies", default=None,
        help="comma-separated policies to plot (default: all present, e.g. drop `random`)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results)
    datasets = args.datasets.split(",") if args.datasets else None
    policies = set(args.policies.split(",")) if args.policies else None
    results = _read_results(results_dir, datasets, policies)
    if not results:
        sel = f" matching policies={sorted(policies)}" if policies else ""
        raise SystemExit(
            f"no <policy>.jsonl found under {results_dir}/<dataset>/{sel} — run run_eval.py first",
        )

    out_dir = Path(args.out) if args.out else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = [_plot_over_time(ds, bp, out_dir) for ds, bp in sorted(results.items())]
    written.append(_plot_summary(results, out_dir))
    for p in written:
        sys.stdout.write(f"wrote {p}\n")

if __name__ == "__main__":
    main()

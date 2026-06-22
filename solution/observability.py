from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.datasets import list_datasets, load_dataset

OUT_DIR = Path(__file__).resolve().parent / "results" / "observability"
PRIMARY_FEATURES = [
    "country",
    "language",
    "platform",
    "device_type",
    "referrer_id",
    "color_scheme",
    "is_returning",
]


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _short_variant(v: str) -> str:
    return v.replace("variant_", "v")


def _time_bin(df: pd.DataFrame, ts_col: str, n_bins: int = 12) -> pd.Series:
    order = df[ts_col].rank(method="first")
    bins = pd.qcut(order, q=min(n_bins, len(df)), labels=False, duplicates="drop")
    return bins.astype(int)


def _propensity_weighted_stats(df: pd.DataFrame, variant_col: str, reward_col: str, prop_col: str):
    rows = []
    for variant, sub in df.groupby(variant_col):
        p = sub[prop_col].astype(float).clip(lower=1e-9)
        w = 1.0 / p
        reward = sub[reward_col].astype(float)
        weighted_mean = float((reward * w).sum() / w.sum())
        ess = float(w.sum() ** 2 / (w.pow(2).sum()))
        se = math.sqrt(max(weighted_mean * (1.0 - weighted_mean), 0.0) / ess)
        rows.append(
            {
                "variant": variant,
                "weighted_reward": weighted_mean,
                "weighted_ess": ess,
                "weighted_se": se,
                "max_weight": float(w.max()),
            },
        )
    return pd.DataFrame(rows)


def overview_plot() -> Path:
    rows = []
    for name in list_datasets():
        df, meta = load_dataset(name)
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "variants": len(meta.variant_ids),
                "reward": float(df[meta.reward_column].mean()),
                "propensity_logged": 1 if meta.propensity_column else 0,
            },
        )
    data = pd.DataFrame(rows)

    fig, axs = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    axs = axs.ravel()
    axs[0].bar(data["dataset"], data["rows"], color="#4c78a8")
    axs[0].set_title("Rows")
    axs[0].set_ylabel("logged impressions")
    axs[0].tick_params(axis="x", rotation=25)

    axs[1].bar(data["dataset"], data["variants"], color="#72b7b2")
    axs[1].set_title("Action Count")
    axs[1].set_ylabel("variants")
    axs[1].tick_params(axis="x", rotation=25)

    axs[2].bar(data["dataset"], data["reward"] * 100, color="#f58518")
    axs[2].set_title("Observed Reward Rate")
    axs[2].set_ylabel("conversion rate (%)")
    axs[2].tick_params(axis="x", rotation=25)

    colors = np.where(data["propensity_logged"] == 1, "#54a24b", "#bab0ac")
    axs[3].bar(data["dataset"], data["propensity_logged"], color=colors)
    axs[3].set_title("Logged Propensities")
    axs[3].set_yticks([0, 1])
    axs[3].set_yticklabels(["no", "yes"])
    axs[3].tick_params(axis="x", rotation=25)
    fig.suptitle("Dataset Overview", fontsize=16)
    return _save(fig, "overview.png")


def assignment_mix_plot() -> Path:
    datasets = list_datasets()
    shares = []
    all_variants: list[str] = []
    for name in datasets:
        df, meta = load_dataset(name)
        s = df[meta.variant_column].value_counts(normalize=True).sort_index()
        shares.append(s)
        all_variants.extend(s.index.tolist())
    all_variants = sorted(set(all_variants))
    table = pd.DataFrame(index=datasets, columns=all_variants, data=0.0)
    for name, s in zip(datasets, shares, strict=True):
        table.loc[name, s.index] = s.values

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    bottom = np.zeros(len(datasets))
    palette = plt.cm.tab20(np.linspace(0, 1, len(all_variants)))
    for color, variant in zip(palette, all_variants, strict=True):
        vals = table[variant].to_numpy(dtype=float) * 100
        if vals.max() == 0:
            continue
        ax.bar(datasets, vals, bottom=bottom, label=_short_variant(variant), color=color)
        bottom += vals
    ax.set_title("Logged Assignment Mix")
    ax.set_ylabel("share of logged rows (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, "assignment_mix.png")


def reward_support_plot() -> Path:
    datasets = list_datasets()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axs = axs.ravel()
    for ax, name in zip(axs, datasets, strict=True):
        df, meta = load_dataset(name)
        stats = (
            df.groupby(meta.variant_column)[meta.reward_column]
            .agg(["mean", "count"])
            .reindex(meta.variant_ids)
        )
        stats["se"] = np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["count"])
        x = np.arange(len(stats))
        ax.bar(x, stats["mean"] * 100, yerr=1.96 * stats["se"] * 100, color="#4c78a8")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_variant(v) for v in stats.index], rotation=40, ha="right")
        ax.set_ylabel("observed reward (%)")
        ax.grid(axis="y", alpha=0.25)
        if meta.propensity_column:
            weighted = _propensity_weighted_stats(
                df,
                meta.variant_column,
                meta.reward_column,
                meta.propensity_column,
            ).set_index("variant").reindex(meta.variant_ids)
            ax.scatter(
                x,
                weighted["weighted_reward"] * 100,
                marker="D",
                s=35,
                color="#e45756",
                label="IPW mean",
                zorder=3,
            )
            ax.legend(frameon=False, fontsize=8)
        for i, cnt in enumerate(stats["count"]):
            ax.text(i, 0, f"n={int(cnt):,}", rotation=90, va="bottom", ha="center", fontsize=7)
    fig.suptitle("Reward by Variant with 95% Binomial Error Bars", fontsize=16)
    return _save(fig, "reward_by_variant_support.png")


def propensity_plot() -> Path:
    prop_datasets = []
    for name in list_datasets():
        df, meta = load_dataset(name)
        if meta.propensity_column:
            prop_datasets.append((name, df, meta))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, df, meta in prop_datasets:
        axs[0].hist(
            df[meta.propensity_column].astype(float),
            bins=np.linspace(0, 1, 21),
            alpha=0.65,
            label=name,
        )
    axs[0].set_title("Propensity Distribution")
    axs[0].set_xlabel("logged assignment probability")
    axs[0].set_ylabel("rows")
    axs[0].legend(frameon=False)

    labels = []
    ess_ratios = []
    max_weights = []
    for name, df, meta in prop_datasets:
        p = df[meta.propensity_column].astype(float).clip(lower=1e-9)
        w = 1.0 / p
        labels.append(name)
        ess_ratios.append(float((w.sum() ** 2 / w.pow(2).sum()) / len(df)))
        max_weights.append(float(w.max()))
    x = np.arange(len(labels))
    axs[1].bar(x - 0.18, np.array(ess_ratios) * 100, width=0.36, label="ESS / rows")
    axs[1].bar(x + 0.18, max_weights, width=0.36, label="max IPW weight")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_title("Weight Variance Diagnostics")
    axs[1].set_ylabel("% or weight multiplier")
    axs[1].legend(frameon=False)
    return _save(fig, "propensity_diagnostics.png")


def action_availability_plot() -> Path:
    datasets = list_datasets()
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axs = axs.ravel()
    for ax, name in zip(axs, datasets, strict=True):
        df, meta = load_dataset(name)
        t0 = df[meta.timestamp_column].min()
        t1 = df[meta.timestamp_column].max()
        span = (t1 - t0).total_seconds() or 1.0
        for y, variant in enumerate(meta.variant_ids):
            sub = df[df[meta.variant_column] == variant]
            start = (sub[meta.timestamp_column].min() - t0).total_seconds() / span
            end = (sub[meta.timestamp_column].max() - t0).total_seconds() / span
            ax.plot([start, end], [y, y], lw=7, solid_capstyle="butt")
        ax.set_title(name)
        ax.set_yticks(range(len(meta.variant_ids)))
        ax.set_yticklabels([_short_variant(v) for v in meta.variant_ids], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of month")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Action Availability Over Time", fontsize=16)
    return _save(fig, "action_availability.png")


def temporal_assignment_plot() -> Path:
    datasets = list_datasets()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axs = axs.ravel()
    for ax, name in zip(axs, datasets, strict=True):
        df, meta = load_dataset(name)
        work = df.copy()
        work["_bin"] = _time_bin(work, meta.timestamp_column, n_bins=12)
        tab = pd.crosstab(work["_bin"], work[meta.variant_column], normalize="index").reindex(
            columns=meta.variant_ids,
            fill_value=0,
        )
        bottom = np.zeros(len(tab))
        x = np.arange(len(tab))
        palette = plt.cm.tab20(np.linspace(0, 1, len(meta.variant_ids)))
        for color, variant in zip(palette, meta.variant_ids, strict=True):
            vals = tab[variant].to_numpy() * 100
            ax.bar(x, vals, bottom=bottom, color=color, label=_short_variant(variant))
            bottom += vals
        ax.set_title(name)
        ax.set_ylim(0, 100)
        ax.set_xlabel("time bin")
        ax.set_ylabel("assignment share (%)")
        if len(meta.variant_ids) <= 4:
            ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Assignment Mix by Time Bin", fontsize=16)
    return _save(fig, "temporal_assignment_mix.png")


def temporal_reward_plot() -> Path:
    datasets = list_datasets()
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axs = axs.ravel()
    for ax, name in zip(axs, datasets, strict=True):
        df, meta = load_dataset(name)
        work = df.copy()
        work["_bin"] = _time_bin(work, meta.timestamp_column, n_bins=16)
        stats = work.groupby("_bin")[meta.reward_column].agg(["mean", "count"])
        se = np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["count"])
        ax.errorbar(
            stats.index,
            stats["mean"] * 100,
            yerr=1.96 * se * 100,
            marker="o",
            lw=1.5,
            color="#4c78a8",
        )
        ax.set_title(name)
        ax.set_xlabel("time bin")
        ax.set_ylabel("observed reward (%)")
        ax.grid(alpha=0.25)
    fig.suptitle("Observed Reward Over Time", fontsize=16)
    return _save(fig, "temporal_reward.png")


def segment_signal_plot() -> Path:
    rows = []
    for name in list_datasets():
        df, meta = load_dataset(name)
        global_rates = (
            df.groupby(meta.variant_column)[meta.reward_column].mean().reindex(meta.variant_ids)
        )
        best_global = float(global_rates.max())
        for feature in [f for f in PRIMARY_FEATURES if f in df.columns]:
            work = df[[feature, meta.variant_column, meta.reward_column]].copy()
            work[feature] = work[feature].astype(object).where(work[feature].notna(), "Unknown")
            work[feature] = work[feature].astype(str)
            grouped = (
                work.groupby([feature, meta.variant_column])[meta.reward_column]
                .agg(["sum", "count"])
                .reset_index()
            )
            segment_sizes = work[feature].value_counts()
            total_score = 0.0
            prior_strength = 50.0
            for segment, segment_n in segment_sizes.items():
                sub = grouped[grouped[feature] == segment].set_index(meta.variant_column)
                estimates = []
                for variant in meta.variant_ids:
                    if variant in sub.index:
                        reward_sum = float(sub.loc[variant, "sum"])
                        count = float(sub.loc[variant, "count"])
                    else:
                        reward_sum = 0.0
                        count = 0.0
                    prior = float(global_rates.loc[variant])
                    estimates.append(
                        (reward_sum + prior_strength * prior) / (count + prior_strength),
                    )
                total_score += int(segment_n) * max(estimates)
            proxy = total_score / len(df)
            rows.append(
                {
                    "dataset": name,
                    "feature": feature,
                    "lift_pp": (proxy - best_global) * 100,
                },
            )

    table = pd.DataFrame(rows).pivot(index="dataset", columns="feature", values="lift_pp")
    table = table.reindex(index=list_datasets(), columns=PRIMARY_FEATURES)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    im = ax.imshow(table.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("smoothed in-sample lift proxy (pp)")
    ax.set_title("Single-Feature Context Signal Diagnostic")
    return _save(fig, "segment_signal_heatmap.png")


def write_index(paths: list[Path]) -> Path:
    md = [
        "# Data Observability Pack",
        "",
        "These displays summarize the logged data used by the policy. They are",
        "generated from the observed logs, not from the hidden truth files.",
        "",
        "## Displays",
        "",
    ]
    descriptions = {
        "overview.png": "Dataset size, action count, observed reward rate, and whether propensities are logged.",
        "assignment_mix.png": "Overall logged action mix by dataset.",
        "reward_by_variant_support.png": "Observed reward by action with binomial uncertainty; red diamonds show IPW means where propensities exist.",
        "propensity_diagnostics.png": "Propensity distributions and effective sample size diagnostics.",
        "action_availability.png": "First-to-last appearance of each action over the month.",
        "temporal_assignment_mix.png": "How assignment mix changes across chronological bins.",
        "temporal_reward.png": "Observed conversion rate over chronological bins.",
        "segment_signal_heatmap.png": "Single-feature smoothed segment signal proxy used to guide policy design.",
    }
    for path in paths:
        md.extend(
            [
                f"### {path.name}",
                "",
                descriptions.get(path.name, ""),
                "",
                f"![{path.stem}]({path.name})",
                "",
            ],
        )
    out = OUT_DIR / "README.md"
    out.write_text("\n".join(md))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        overview_plot(),
        assignment_mix_plot(),
        reward_support_plot(),
        propensity_plot(),
        action_availability_plot(),
        temporal_assignment_plot(),
        temporal_reward_plot(),
        segment_signal_plot(),
    ]
    index = write_index(paths)
    print(f"wrote {index}")
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"
OUT_DIR = RESULTS / "eval_observability"

POLICY_ORDER = [
    "random",
    "my_policy",
    "best_fixed",
    "example",
    "feature_demo",
    "seg_country",
    "seg_eb",
    "seg_eb_recency",
    "gbm_tlearner",
    "hybrid_eb_gbm",
    "lgbm_tlearner",
    "hybrid_eb_lgbm",
]


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _policy_order(policies: list[str]) -> list[str]:
    known = [p for p in POLICY_ORDER if p in policies]
    other = sorted(p for p in policies if p not in known)
    return known + other


def _load_records() -> pd.DataFrame:
    rows = []
    for ds_dir in sorted(RESULTS.iterdir()):
        if not ds_dir.is_dir():
            continue
        if ds_dir.name in {"plots", "observability", "eval_observability"}:
            continue
        for path in sorted(ds_dir.glob("*.jsonl")):
            with path.open() as fh:
                for line in fh:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no eval jsonl records found under {RESULTS}")
    df = pd.DataFrame(rows)
    df["captured_display"] = df["cumulative_captured_pct"].astype(float)
    return df


def _last_records(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["dataset", "policy", "iteration"])
        .groupby(["dataset", "policy"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def score_heatmap(df: pd.DataFrame) -> Path:
    final = _last_records(df)
    datasets = sorted(final["dataset"].unique())
    policies = _policy_order(final["policy"].unique().tolist())
    table = final.pivot(index="dataset", columns="policy", values="captured_display")
    table = table.reindex(index=datasets, columns=policies)
    avg = table.mean(axis=0, skipna=True)
    table = pd.concat([table, pd.DataFrame([avg], index=["average"])])

    fig, ax = plt.subplots(figsize=(1.3 * len(policies) + 4, 0.55 * len(table) + 3), constrained_layout=True)
    vals = table.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmax = max(5.0, float(np.nanmax(np.abs(finite)))) if finite.size else 5.0
    im = ax.imshow(vals, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            label = "n/a" if pd.isna(val) else f"{val:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("final captured headroom (%)")
    ax.set_title("Policy Scoreboard: Final Captured Headroom")
    return _save(fig, "policy_scoreboard_heatmap.png")


def convergence_grid(df: pd.DataFrame) -> Path:
    datasets = sorted(df["dataset"].unique())
    policies = _policy_order(df["policy"].unique().tolist())
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axs = axs.ravel()
    colors = dict(zip(policies, plt.cm.tab10(np.linspace(0, 1, max(len(policies), 2))), strict=False))
    for ax, dataset in zip(axs, datasets, strict=True):
        sub_ds = df[df["dataset"] == dataset]
        any_line = False
        for policy in policies:
            sub = sub_ds[sub_ds["policy"] == policy].sort_values("iteration")
            if sub.empty:
                continue
            y = sub["captured_display"].to_numpy(dtype=float)
            if np.all(np.isnan(y)):
                continue
            ax.plot(
                sub["n_train"],
                y,
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=policy,
                color=colors[policy],
            )
            any_line = True
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(dataset)
        ax.set_xlabel("training rows seen")
        ax.set_ylabel("captured headroom (%)")
        ax.grid(alpha=0.25)
        if not any_line:
            ax.text(0.5, 0.5, "no headroom", transform=ax.transAxes, ha="center", va="center")
        if len(policies) <= 8 and any_line:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Walk-Forward Convergence by Dataset", fontsize=16)
    return _save(fig, "captured_convergence_grid.png")


def value_components(df: pd.DataFrame) -> Path:
    final = _last_records(df)
    datasets = sorted(final["dataset"].unique())
    policies = _policy_order(final["policy"].unique().tolist())
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axs = axs.ravel()
    for ax, dataset in zip(axs, datasets, strict=True):
        sub = final[final["dataset"] == dataset].set_index("policy").reindex(policies).dropna(how="all")
        x = np.arange(len(sub))
        ax.plot(
            x,
            sub["cumulative_best_fixed"] * 100,
            color="black",
            linestyle="--",
            label="best fixed",
        )
        ax.plot(
            x,
            sub["cumulative_oracle"] * 100,
            color="#54a24b",
            linestyle="--",
            label="oracle",
        )
        ax.scatter(x, sub["cumulative_cvr"] * 100, color="#4c78a8", s=35, label="policy")
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("true expected CVR (%)")
        ax.grid(axis="y", alpha=0.25)
        if dataset == datasets[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Final Policy Value vs Best-Fixed and Oracle", fontsize=16)
    return _save(fig, "final_value_components.png")


def speed_plot(df: pd.DataFrame) -> Path:
    grouped = (
        df.groupby("policy")[["train_seconds", "inference_seconds"]]
        .mean()
        .reindex(_policy_order(df["policy"].unique().tolist()))
    )
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(grouped))
    ax.bar(x - 0.18, grouped["train_seconds"], width=0.36, label="mean train seconds / window")
    ax.bar(x + 0.18, grouped["inference_seconds"], width=0.36, label="mean inference seconds / window")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=35, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Harness Runtime by Policy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, "policy_runtime.png")


def recommendation_mix_plot(df: pd.DataFrame) -> Path:
    final = _last_records(df)
    datasets = sorted(final["dataset"].unique())
    policies = _policy_order(final["policy"].unique().tolist())
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axs = axs.ravel()
    for ax, dataset in zip(axs, datasets, strict=True):
        sub = final[final["dataset"] == dataset]
        variants = sorted({v for mix in sub["recommendation_mix"] for v in mix})
        if not variants:
            ax.set_axis_off()
            continue
        plot_policies = [p for p in policies if p in set(sub["policy"])]
        matrix = np.zeros((len(plot_policies), len(variants)))
        for i, policy in enumerate(plot_policies):
            mix = sub[sub["policy"] == policy]["recommendation_mix"].iloc[0]
            for j, variant in enumerate(variants):
                matrix[i, j] = float(mix.get(variant, 0.0))
        bottom = np.zeros(len(plot_policies))
        x = np.arange(len(plot_policies))
        palette = plt.cm.tab20(np.linspace(0, 1, len(variants)))
        for color, variant, vals in zip(palette, variants, matrix.T, strict=True):
            ax.bar(x, vals * 100, bottom=bottom * 100, color=color, label=variant.replace("variant_", "v"))
            bottom += vals
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_policies, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("final-window recommendation share (%)")
        ax.set_ylim(0, 100)
        if len(variants) <= 6:
            ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Final-Window Recommendation Mix", fontsize=16)
    return _save(fig, "final_recommendation_mix.png")


def harness_windows_plot(df: pd.DataFrame) -> Path:
    baseline_policy = "random" if "random" in set(df["policy"]) else df["policy"].iloc[0]
    base = df[df["policy"] == baseline_policy]
    datasets = sorted(base["dataset"].unique())
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axs = axs.ravel()
    for ax, dataset in zip(axs, datasets, strict=True):
        sub = base[base["dataset"] == dataset].sort_values("iteration")
        ax.bar(sub["iteration"], sub["n_window"], color="#72b7b2", alpha=0.8, label="window rows")
        ax2 = ax.twinx()
        ax2.plot(sub["iteration"], sub["n_train"], color="#4c78a8", linewidth=1.8, label="train rows")
        ax.set_title(dataset)
        ax.set_xlabel("iteration")
        ax.set_ylabel("window rows")
        ax2.set_ylabel("training rows")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Harness Walk-Forward Schedule", fontsize=16)
    return _save(fig, "harness_walk_forward_windows.png")


def violations_plot(df: pd.DataFrame) -> Path:
    totals = (
        df.groupby(["dataset", "policy"])["choice_violations"]
        .sum()
        .reset_index()
        .pivot(index="dataset", columns="policy", values="choice_violations")
        .fillna(0)
    )
    policies = _policy_order(totals.columns.tolist())
    totals = totals.reindex(columns=policies)
    fig, ax = plt.subplots(figsize=(1.1 * len(policies) + 4, 5), constrained_layout=True)
    im = ax.imshow(totals.to_numpy(dtype=float), cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=35, ha="right")
    ax.set_yticks(range(len(totals.index)))
    ax.set_yticklabels(totals.index)
    for i in range(totals.shape[0]):
        for j in range(totals.shape[1]):
            ax.text(j, i, f"{int(totals.iloc[i, j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="total invalid choices coerced by harness")
    ax.set_title("Choice Violations")
    return _save(fig, "choice_violations.png")


def write_index(paths: list[Path]) -> Path:
    descriptions = {
        "policy_scoreboard_heatmap.png": "Final captured-headroom table by dataset and policy, including policy averages.",
        "captured_convergence_grid.png": "Walk-forward convergence curves showing captured headroom versus training rows.",
        "final_value_components.png": "Policy CVR against the best-fixed baseline and oracle ceiling.",
        "policy_runtime.png": "Mean train and inference time per policy/window.",
        "final_recommendation_mix.png": "Final-window recommendation distribution by dataset and policy.",
        "harness_walk_forward_windows.png": "Window sizes and cumulative training set sizes used by the harness.",
        "choice_violations.png": "Invalid action choices caught and replaced by the harness.",
    }
    lines = [
        "# Eval Observability Pack",
        "",
        "These charts are generated from `solution/results/<dataset>/<policy>.jsonl`.",
        "They explain the walk-forward harness, policy performance, latency, and",
        "serving behavior. Regenerate with:",
        "",
        "```bash",
        "uv run python solution/eval_observability.py",
        "```",
        "",
    ]
    for path in paths:
        lines.extend(
            [
                f"## {path.name}",
                "",
                descriptions.get(path.name, ""),
                "",
                f"![{path.stem}]({path.name})",
                "",
            ],
        )
    out = OUT_DIR / "README.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_records()
    paths = [
        score_heatmap(df),
        convergence_grid(df),
        value_components(df),
        speed_plot(df),
        recommendation_mix_plot(df),
        harness_windows_plot(df),
        violations_plot(df),
    ]
    index = write_index(paths)
    print(f"wrote {index}")
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Arm-choice sensitivity diagnostics.

These plots use the hidden truth files for analysis only. The submitted policies
must not train on truth. The point is to understand how much the recommendation
depends on choosing the right arm, and whether value is concentrated in a small
number of variants.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.datasets import list_datasets, load_dataset
from engine.scoring import load_truth


OUT = Path(__file__).resolve().parent / "results" / "arm_sensitivity"


def _availability_mask(df: pd.DataFrame, variants: list[str]) -> np.ndarray:
    """Approximate row-level availability from first/last logged appearance."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    spans = df.groupby("variant_id")["timestamp"].agg(["min", "max"])
    mask = np.zeros((len(df), len(variants)), dtype=bool)
    for j, variant in enumerate(variants):
        if variant not in spans.index:
            continue
        start, end = spans.loc[variant, "min"], spans.loc[variant, "max"]
        mask[:, j] = (ts >= start).to_numpy() & (ts <= end).to_numpy()
    return mask


def compute_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    winner_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []

    for name in list_datasets():
        df, _meta = load_dataset(name)
        truth = load_truth(name)
        variants = truth.variant_order
        prob = truth.prob.copy()

        avail = _availability_mask(df, variants)
        # If span inference ever misses a row, fall back to all variants for that row.
        empty = ~avail.any(axis=1)
        avail[empty, :] = True

        masked = np.where(avail, prob, -np.inf)
        order = np.argsort(masked, axis=1)
        best_idx = order[:, -1]
        second_idx = order[:, -2] if len(variants) > 1 else order[:, -1]
        best_val = masked[np.arange(len(masked)), best_idx]
        second_val = masked[np.arange(len(masked)), second_idx]
        gap = best_val - second_val

        counts = pd.Series(np.asarray(variants, dtype=object)[best_idx]).value_counts(normalize=True)
        for variant in variants:
            winner_rows.append(
                {
                    "dataset": name,
                    "variant": variant,
                    "oracle_winner_share": float(counts.get(variant, 0.0)),
                },
            )

        margin_rows.append(
            {
                "dataset": name,
                "mean_top_second_gap_pp": float(np.mean(gap) * 100.0),
                "p10_gap_pp": float(np.quantile(gap, 0.10) * 100.0),
                "p50_gap_pp": float(np.quantile(gap, 0.50) * 100.0),
                "p90_gap_pp": float(np.quantile(gap, 0.90) * 100.0),
            },
        )

        for j, variant in enumerate(variants):
            without = masked.copy()
            without[:, j] = -np.inf
            best_without = without.max(axis=1)
            # If no alternative is available, the arm is not a meaningful choice sensitivity case.
            valid = np.isfinite(best_without)
            loss_rows.append(
                {
                    "dataset": name,
                    "variant": variant,
                    "leave_one_arm_out_loss_pp": float(np.mean(best_val[valid] - best_without[valid]) * 100.0),
                    "share_rows_where_arm_is_best": float(np.mean(best_idx == j)),
                },
            )

    return pd.DataFrame(winner_rows), pd.DataFrame(margin_rows), pd.DataFrame(loss_rows)


def plot_winner_share(winners: pd.DataFrame) -> None:
    pivot = winners.pivot(index="dataset", columns="variant", values="oracle_winner_share").fillna(0.0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6), width=0.82, colormap="tab20")
    ax.set_title("Oracle best-arm share by dataset")
    ax.set_xlabel("")
    ax.set_ylabel("Share of rows where variant is best")
    ax.legend(title="variant", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "oracle_winner_share.png", dpi=180)
    plt.close()


def plot_margins(margins: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(margins))
    ax.bar(x, margins["mean_top_second_gap_pp"], color="#16756f")
    ax.errorbar(
        x,
        margins["p50_gap_pp"],
        yerr=[
            margins["p50_gap_pp"] - margins["p10_gap_pp"],
            margins["p90_gap_pp"] - margins["p50_gap_pp"],
        ],
        fmt="o",
        color="#17212b",
        capsize=4,
        label="p50 with p10-p90 interval",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(margins["dataset"], rotation=0)
    ax.set_ylabel("Probability-point gap")
    ax.set_title("True top-arm vs runner-up reward gap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "top_second_gap.png", dpi=180)
    plt.close()


def plot_leave_one_out(losses: pd.DataFrame) -> None:
    pivot = losses.pivot(index="dataset", columns="variant", values="leave_one_arm_out_loss_pp").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Oracle value lost if an arm is unavailable")
    ax.set_xlabel("removed arm")
    ax.set_ylabel("")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean CVR loss, percentage points")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="#102030")
    plt.tight_layout()
    plt.savefig(OUT / "leave_one_arm_out_loss.png", dpi=180)
    plt.close()


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.astype(object).where(pd.notna(df), "")
    headers = [str(c) for c in display.columns]
    rows = [[str(v) for v in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_readme(winners: pd.DataFrame, margins: pd.DataFrame, losses: pd.DataFrame) -> None:
    lines = [
        "# Arm Sensitivity Diagnostics",
        "",
        "These plots use the hidden truth files for analysis only. They are not used by the submitted policies.",
        "",
        "## Displays",
        "",
        "### oracle_winner_share.png",
        "",
        "How often each variant is the true best arm for a row.",
        "",
        "![oracle_winner_share](oracle_winner_share.png)",
        "",
        "### top_second_gap.png",
        "",
        "How much better the true best arm is than the runner-up. Small gaps mean arm choice is intrinsically noisy; large gaps mean getting the arm right matters.",
        "",
        "![top_second_gap](top_second_gap.png)",
        "",
        "### leave_one_arm_out_loss.png",
        "",
        "Oracle CVR lost if each arm is removed from consideration.",
        "",
        "![leave_one_arm_out_loss](leave_one_arm_out_loss.png)",
        "",
        "## Summary tables",
        "",
        "### Top-second gap",
        "",
        _markdown_table(margins.round(4)),
        "",
        "### Largest leave-one-arm-out losses",
        "",
        _markdown_table(
            losses.sort_values(["dataset", "leave_one_arm_out_loss_pp"], ascending=[True, False])
            .groupby("dataset")
            .head(3)
            .round(4),
        ),
        "",
        "### Oracle winner share",
        "",
        _markdown_table(winners.round(4)),
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    winners, margins, losses = compute_sensitivity()
    winners.to_csv(OUT / "oracle_winner_share.csv", index=False)
    margins.to_csv(OUT / "top_second_gap.csv", index=False)
    losses.to_csv(OUT / "leave_one_arm_out_loss.csv", index=False)
    plot_winner_share(winners)
    plot_margins(margins)
    plot_leave_one_out(losses)
    write_readme(winners, margins, losses)
    print(f"Wrote arm sensitivity diagnostics to {OUT}")


if __name__ == "__main__":
    main()

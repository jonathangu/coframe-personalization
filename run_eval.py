import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.datasets import list_datasets, load_dataset
from engine.evaluation import DEFAULT_ITERATIONS, DEFAULT_SCHEDULE, walk_forward
from engine.scoring import OracleScoring, load_truth
from policies import available, get_policy

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "solution" / "results"

Record = dict[str, Any]

def _fmt_pct(x: float, numw: int = 6) -> str:
    return f"{x:{numw}.1f}%" if not math.isnan(x) else "n/a".rjust(numw + 1)

def _echo(line: str = "", *, end: str = "\n") -> None:
    sys.stdout.write(f"{line}{end}")
    sys.stdout.flush()

def run_dataset(
    name: str,
    policy_labels: list[str],
    n_iterations: int,
    schedule: str,
    out_dir: Path,
    quiet: bool = False,
) -> list[Record]:
    df, meta = load_dataset(name)
    truth = load_truth(name)
    if not quiet:
        prop = " (propensity logged)" if meta.propensity_column else ""
        _echo(
            f"\n=== {name} ===  rows={len(df):,}  variants={len(meta.variant_ids)}  "
            f"features={meta.feature_columns}{prop}",
        )

    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[Record] = []
    for label in policy_labels:
        policy = get_policy(label)
        scorer = OracleScoring(truth, meta, dataset=name, policy=label)
        records = []
        t_wall = time.time()
        for step in walk_forward(policy, df, meta, n_iterations=n_iterations, schedule=schedule):
            rec = scorer.score_step(step)
            records.append(rec)
            if not quiet:
                _echo(
                    f"  {label:14s} iter {rec.iteration:>2d}/{n_iterations}  "
                    f"n_train={rec.n_train:>7,}  captured={_fmt_pct(rec.cumulative_captured_pct)}",
                    end="\r",
                )
        if not records:
            continue
        with (ds_dir / f"{label}.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(r.as_json()) + "\n")
        final = records[-1]
        tot_train = sum(r.train_seconds for r in records)
        tot_infer = sum(r.inference_seconds for r in records)
        summaries.append(
            {
                "dataset": name,
                "policy": label,
                "captured_pct": final.cumulative_captured_pct,
                "policy_cvr": final.cumulative_cvr,
                "oracle_cvr": final.cumulative_oracle,
                "best_fixed_cvr": final.cumulative_best_fixed,
                "n_eval": sum(r.n_window for r in records),
                "n_iterations": len(records),
                "total_train_seconds": round(tot_train, 3),
                "total_inference_seconds": round(tot_infer, 3),
                "mean_train_seconds": round(tot_train / len(records), 4),
                "mean_inference_seconds": round(tot_infer / len(records), 4),
            },
        )
        if not quiet:
            _echo(
                f"  {label:14s} done  captured={_fmt_pct(final.cumulative_captured_pct)}  "
                f"value={final.cumulative_cvr * 100:5.2f}%  "
                f"bestfix={final.cumulative_best_fixed * 100:5.2f}%  "
                f"oracle={final.cumulative_oracle * 100:5.2f}%  "
                f"train={tot_train:5.1f}s infer={tot_infer:4.1f}s "
                f"wall={time.time() - t_wall:5.1f}s",
            )

    return summaries

def _averages(summaries: list[Record], policy_labels: list[str]) -> list[Record]:
    rows = []
    for label in policy_labels:
        caps = [
            s["captured_pct"]
            for s in summaries
            if s["policy"] == label and not math.isnan(s["captured_pct"])
        ]
        mine = [s for s in summaries if s["policy"] == label]
        rows.append(
            {
                "policy": label,
                "mean_captured_pct": float(np.mean(caps)) if caps else float("nan"),
                "n_headroom_datasets": len(caps),
                "total_train_seconds": round(sum(s["total_train_seconds"] for s in mine), 2),
                "total_inference_seconds": round(
                    sum(s["total_inference_seconds"] for s in mine), 2,
                ),
            },
        )
    return rows

def _log_report(summaries: list[Record], averages: list[Record], datasets: list[str]) -> str:
    lines = ["", "=" * 78, "CAPTURED HEADROOM (true value vs per-impression oracle)", "=" * 78]
    header = (
        f"{'dataset':10s} {'policy':14s} {'captured':>9s} "
        f"{'value':>8s} {'bestfix':>8s} {'oracle':>8s}"
    )
    lines.append(header)
    lines.append("-" * 78)
    for ds in datasets:
        lines.extend(
            f"{ds:10s} {s['policy']:14s} {_fmt_pct(s['captured_pct'], 8)} "
            f"{s['policy_cvr'] * 100:7.2f}% {s['best_fixed_cvr'] * 100:7.2f}% "
            f"{s['oracle_cvr'] * 100:7.2f}%"
            for s in summaries
            if s["dataset"] == ds
        )
        lines.append("-" * 78)
    lines.append("")
    lines.append(f"{'AVERAGE across headroom datasets':40s}")
    lines.extend(
        f"  {a['policy']:14s} mean captured={_fmt_pct(a['mean_captured_pct'])}  "
        f"(n={a['n_headroom_datasets']} datasets)  "
        f"train={a['total_train_seconds']}s infer={a['total_inference_seconds']}s"
        for a in averages
    )
    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("names", nargs="*", help="datasets (positional; same as --datasets)")
    ap.add_argument("--datasets", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--policies", default="random", help="comma-separated registered policies")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERATIONS)
    ap.add_argument("--schedule", choices=["hybrid", "uniform"], default=DEFAULT_SCHEDULE)
    ap.add_argument("--quick", action="store_true", help="16 iters on vega+meadow")
    ap.add_argument("--workers", type=int, default=1, help="run datasets in parallel processes")
    ap.add_argument(
        "--threads", type=int, default=None,
        help="Threads per worker (default: ~cores/workers; all cores if serial)",
    )
    ap.add_argument("--out", default=str(RESULTS), help="results directory")
    args = ap.parse_args()

    names = args.names or (args.datasets.split(",") if args.datasets else None)
    iters = args.iters
    if args.quick:
        names = names or ["vega", "meadow"]
        iters = min(iters, 16)
    names = names or list_datasets()
    unknown_ds = [n for n in names if n not in list_datasets()]
    if unknown_ds:
        raise SystemExit(f"unknown datasets {unknown_ds}; available: {list_datasets()}")
    labels = [p for p in args.policies.split(",") if p]
    unknown_p = [p for p in labels if p not in available()]
    if unknown_p:
        raise SystemExit(f"unknown policies {unknown_p}; registered: {available()}")

    threads = args.threads
    if threads is None and args.workers > 1 and len(names) > 1:
        threads = max(1, round((os.cpu_count() or args.workers) / args.workers))
    if threads is not None:
        os.environ["EVAL_THREADS"] = str(threads)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, str(threads))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _echo("Truth-scored captured headroom (NOT off-policy estimates).")
    _echo(
        f"datasets={names}  policies={labels}  iters={iters}  schedule={args.schedule}  "
        f"workers={args.workers}  threads={threads or 'all'}",
    )

    started = datetime.now(UTC)
    summaries: list[Record] = []
    if args.workers > 1 and len(names) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(run_dataset, n, labels, iters, args.schedule, out_dir, True): n
                for n in names
            }
            for fut in as_completed(futs):
                rows = fut.result()
                summaries.extend(rows)
                done = futs[fut]
                caps = [f"{r['policy']}={_fmt_pct(r['captured_pct']).strip()}" for r in rows]
                _echo(f"  [done] {done:10s} {'  '.join(caps)}")
    else:
        for n in names:
            summaries.extend(run_dataset(n, labels, iters, args.schedule, out_dir, quiet=False))

    averages = _averages(summaries, labels)
    summary = {
        "generated_at": started.isoformat(),
        "n_iterations": iters,
        "schedule": args.schedule,
        "datasets": names,
        "policies": labels,
        "runs": summaries,
        "averages": averages,
        "wall_seconds": round((datetime.now(UTC) - started).total_seconds(), 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    report = _log_report(summaries, averages, names)
    _echo(report)
    (out_dir / "eval.log").write_text(report.lstrip("\n") + "\n")
    _echo(f"\nWrote per-dataset jsonl + summary.json + eval.log to {out_dir}/")

if __name__ == "__main__":
    main()

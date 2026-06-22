import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DATA_DIR, DatasetMeta
from engine.evaluation import IterationStep

HEADROOM_EPS = 1e-6

def captured_pct(policy_cvr: float, oracle_cvr: float, best_fixed: float) -> float:
    headroom = oracle_cvr - best_fixed
    if not headroom > HEADROOM_EPS:
        return float("nan")
    return (policy_cvr - best_fixed) / headroom * 100.0

class BestFixedAccumulator:

    def __init__(self, n_variants: int) -> None:
        self._sum = np.zeros(n_variants)
        self._n = np.zeros(n_variants)

    def update(self, prob_window: NDArray[np.float64], avail_idx: list[int]) -> None:
        for j in avail_idx:
            self._sum[j] += float(prob_window[:, j].sum())
            self._n[j] += len(prob_window)

    @property
    def value(self) -> float:
        with np.errstate(invalid="ignore"):
            variant_means = np.where(self._n > 0, self._sum / np.maximum(self._n, 1), -np.inf)
        return float(variant_means.max())

@dataclass(frozen=True)
class Truth:

    prob: NDArray[np.float64]
    oracle: NDArray[np.float64]
    ts: NDArray[np.int64]
    variant_order: list[str]

    @property
    def index_of(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.variant_order)}

class TruthAlignmentError(RuntimeError):
    pass

def load_truth(name: str) -> Truth:
    path = DATA_DIR / f"{name}_truth.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no truth file for {name!r} at {path}")
    t = pd.read_parquet(path)
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.sort_values("timestamp", kind="stable").reset_index(drop=True)
    variant_order = json.loads(t["variant_order"].iloc[0])
    return Truth(
        prob=t[[f"p_{v}" for v in variant_order]].to_numpy(dtype=float),
        oracle=t["oracle_value"].to_numpy(dtype=float),
        ts=t["timestamp"].astype("int64").to_numpy(),
        variant_order=variant_order,
    )

@dataclass(frozen=True)
class IterationRecord:

    dataset: str
    policy: str
    iteration: int
    window_start: str
    window_end: str
    n_train: int
    n_window: int
    train_seconds: float
    inference_seconds: float
    raw_cvr: float
    cumulative_cvr: float
    raw_oracle: float
    cumulative_oracle: float
    cumulative_best_fixed: float
    cumulative_captured_pct: float
    available_variants: list[str]
    new_variants: list[str]
    choice_violations: int
    recommendation_mix: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, object]:
        return asdict(self)

class OracleScoring:

    def __init__(self, truth: Truth, meta: DatasetMeta, dataset: str, policy: str) -> None:
        self._truth = truth
        self._meta = meta
        self._dataset = dataset
        self._policy = policy
        self._best_fixed = BestFixedAccumulator(len(truth.variant_order))
        self._cum_n = 0.0
        self._cum_policy = 0.0
        self._cum_oracle = 0.0

    def score_step(self, step: IterationStep) -> IterationRecord:
        truth = self._truth
        rows = step.window.index.to_numpy()
        window_ts = step.window[self._meta.timestamp_column].astype("int64").to_numpy()
        if not np.array_equal(window_ts, truth.ts[rows]):
            raise TruthAlignmentError(
                f"alignment mismatch on {self._dataset} iteration {step.iteration} — "
                "logged rows and truth rows are out of order",
            )

        index_of = truth.index_of
        avail_idx = [index_of[v] for v in step.available_variants]
        chosen_idx = np.array([index_of[c] for c in step.chosen])
        prob_win = truth.prob[rows]
        policy_vals = prob_win[np.arange(len(rows)), chosen_idx]

        self._cum_n += len(rows)
        self._cum_policy += float(policy_vals.sum())
        self._cum_oracle += float(truth.oracle[rows].sum())
        self._best_fixed.update(prob_win, avail_idx)

        best_fixed = self._best_fixed.value
        cum_policy_cvr = self._cum_policy / self._cum_n
        cum_oracle_cvr = self._cum_oracle / self._cum_n
        new_variants = [v for v in step.available_variants if v not in step.history_variants]
        mix = (
            pd.Series(step.chosen).value_counts(normalize=True).round(4).to_dict()
            if len(step.chosen)
            else {}
        )
        return IterationRecord(
            dataset=self._dataset,
            policy=self._policy,
            iteration=step.iteration,
            window_start=step.window_start.isoformat(),
            window_end=step.window_end.isoformat(),
            n_train=step.n_train,
            n_window=len(rows),
            train_seconds=round(step.train_seconds, 4),
            inference_seconds=round(step.inference_seconds, 4),
            raw_cvr=float(policy_vals.mean()),
            cumulative_cvr=cum_policy_cvr,
            raw_oracle=float(truth.oracle[rows].mean()),
            cumulative_oracle=cum_oracle_cvr,
            cumulative_best_fixed=best_fixed,
            cumulative_captured_pct=captured_pct(cum_policy_cvr, cum_oracle_cvr, best_fixed),
            available_variants=list(step.available_variants),
            new_variants=new_variants,
            choice_violations=step.choice_violations,
            recommendation_mix={str(k): float(v) for k, v in mix.items()},
        )

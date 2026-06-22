import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DatasetMeta
from engine.policy import IterationContext, IterationSummary, Policy

DEFAULT_ITERATIONS = 50
DEFAULT_WARMUP_FRAC = 0.0
DEFAULT_SCHEDULE = "hybrid"
LOG_UNTIL = 0.10
LOG_SHARE = 0.4

@dataclass
class IterationStep:

    iteration: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_train: int
    window: pd.DataFrame
    chosen: NDArray[np.object_]
    train_seconds: float
    inference_seconds: float
    available_variants: list[str]
    history_variants: list[str]
    choice_violations: int

def _window_fractions(n_iterations: int, n_rows: int, schedule: str) -> NDArray[np.float64]:
    f_min = max(0.002, 30 / max(n_rows, 1))
    n_log = min(max(3, round(LOG_SHARE * n_iterations)), n_iterations - 2)
    if schedule == "uniform" or n_iterations < 8 or f_min >= LOG_UNTIL:
        return np.linspace(0.0, 1.0, n_iterations + 1)
    n_uni = n_iterations - n_log
    geo = f_min * (LOG_UNTIL / f_min) ** (np.arange(n_log) / (n_log - 1))
    uni = LOG_UNTIL + (1.0 - LOG_UNTIL) * np.arange(1, n_uni + 1) / n_uni
    return np.concatenate([[0.0], geo, uni])

def make_windows(
    df: pd.DataFrame,
    meta: DatasetMeta,
    n_iterations: int = DEFAULT_ITERATIONS,
    warmup_frac: float = DEFAULT_WARMUP_FRAC,
    schedule: str = DEFAULT_SCHEDULE,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ts = pd.to_datetime(df[meta.timestamp_column], utc=True)
    t0, t1 = ts.min().value, ts.max().value
    start = t0 + int((t1 - t0) * warmup_frac)
    fracs = _window_fractions(n_iterations, len(df), schedule)
    edges = (start + fracs * (t1 - start)).astype("int64")
    return [
        (pd.Timestamp(edges[i], tz="UTC"), pd.Timestamp(edges[i + 1], tz="UTC"))
        for i in range(n_iterations)
    ]

def _coerce_choices(
    chosen: NDArray[np.object_],
    available: list[str],
    rng: np.random.Generator,
) -> tuple[NDArray[np.object_], int]:
    allowed = set(available)
    bad = np.array([c not in allowed for c in chosen])
    n_bad = int(bad.sum())
    if n_bad:
        chosen = chosen.copy()
        chosen[bad] = rng.choice(np.asarray(available, dtype=object), size=n_bad)
    return chosen, n_bad

def walk_forward(
    policy: Policy,
    df: pd.DataFrame,
    meta: DatasetMeta,
    n_iterations: int = DEFAULT_ITERATIONS,
    warmup_frac: float = DEFAULT_WARMUP_FRAC,
    schedule: str = DEFAULT_SCHEDULE,
) -> Iterator[IterationStep]:
    df = df.sort_values(meta.timestamp_column, kind="stable").reset_index(drop=True)
    ts = df[meta.timestamp_column]
    windows = make_windows(df, meta, n_iterations, warmup_frac, schedule)
    context_cols = [*meta.feature_columns, meta.timestamp_column]
    rng = np.random.default_rng(0)
    past: list[IterationSummary] = []
    counter = 0
    for i, (start, end) in enumerate(windows):
        train = df[ts < start]
        last = i == len(windows) - 1
        in_window = (ts >= start) & (ts <= end if last else ts < end)
        window = df[in_window]
        if window.empty:
            continue
        counter += 1
        available = sorted(window[meta.variant_column].unique().tolist())
        history_variants = sorted(train[meta.variant_column].unique().tolist())
        ctx = IterationContext(iteration=counter, n_seen=len(train), past=list(past))

        t0 = time.perf_counter()
        policy.fit(train, meta, ctx)
        train_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        chosen = np.asarray(policy.recommend(window[context_cols], available_variants=available))
        inference_seconds = time.perf_counter() - t1
        if len(chosen) != len(window):
            raise ValueError(f"recommend() returned {len(chosen)} rows, expected {len(window)}")
        chosen, violations = _coerce_choices(chosen, available, rng)

        yield IterationStep(
            iteration=counter,
            window_start=start,
            window_end=end,
            n_train=len(train),
            window=window,
            chosen=chosen,
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
            available_variants=available,
            history_variants=history_variants,
            choice_violations=violations,
        )
        past.append(
            IterationSummary(
                iteration=counter,
                n_train=len(train),
                n_window=len(window),
                window_start=start.isoformat(),
                window_end=end.isoformat(),
                observed_reward=float(window[meta.reward_column].mean()),
            ),
        )

"""Hello-world example: author a `FeaturePipeline` and consume it in a policy.

This is a minimal, end-to-end demonstration of the feature/policy contract —
not a serious contender for the leaderboard. It shows three things:

1. how to subclass `FeaturePipeline` (here, a stateless time-of-day bucket),
2. how a `ScoredPolicy` calls the pipeline's `fit` (on the training log only)
   and `transform` (at serve time) to turn raw context into features, and
3. how those features drive a per-variant score — leak-free and cold-start safe.

To combine several pipelines, concatenate their `transform` outputs (e.g.
`pd.concat([p.transform(df, meta) for p in pipelines], axis=1)`).
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DatasetMeta
from engine.features import UNKNOWN, FeaturePipeline
from engine.policy import IterationContext, ScoredPolicy

_TOD_COLUMN = "_tod"
# Hour-of-day cut points -> human-readable bucket. Hours 0-5 night, 6-11 morning,
# 12-17 afternoon, 18-23 evening; covers every hour, so no row is left unbucketed.
_TOD_BINS = [-1, 5, 11, 17, 23]
_TOD_LABELS = ["night", "morning", "afternoon", "evening"]

class TimeOfDayPipeline(FeaturePipeline):
    """Derive a coarse time-of-day bucket from the timestamp.

    The transform is a pure, row-local function of the timestamp, so `fit` has
    nothing data-dependent to learn (same as the stateless reference pipelines).
    Anything data-dependent — vocabularies, scalers, frequencies, target stats —
    would be learned here from `train` only, never from the frame we are scored on.
    """

    def fit(self, train: pd.DataFrame, meta: DatasetMeta) -> None:
        pass

    def transform(self, df: pd.DataFrame, meta: DatasetMeta) -> pd.DataFrame:
        ts = pd.to_datetime(df[meta.timestamp_column], utc=True)
        bucket = pd.cut(ts.dt.hour, bins=_TOD_BINS, labels=_TOD_LABELS).astype(object)
        out = pd.DataFrame(index=df.index)
        out[_TOD_COLUMN] = bucket.where(pd.notna(bucket), UNKNOWN).astype(str)
        return out

    def categorical_outputs(self, _meta: DatasetMeta) -> list[str]:
        return [_TOD_COLUMN]

class TimeOfDayPolicy(ScoredPolicy):
    """Personalize by time of day using `TimeOfDayPipeline`.

    Learns a per-(time-of-day, variant) conversion rate on the training log,
    smoothed toward each variant's global rate so thin buckets degrade
    gracefully toward the best-fixed answer rather than chasing noise.
    """

    def __init__(self, seed: int = 0, prior_strength: float = 20.0) -> None:
        super().__init__(seed=seed)
        self.prior_strength = prior_strength
        self.pipeline = TimeOfDayPipeline()
        self._base_rate = 0.0
        self._variant_rate: dict[str, float] = {}
        self._table: pd.DataFrame | None = None

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        _context: IterationContext | None = None,
    ) -> None:
        self.meta = meta
        if len(train) == 0:
            self._table = None
            self._variant_rate = {}
            return

        # Pipeline state and all reward statistics come from the past only (leak-free).
        self.pipeline.fit(train, meta)
        tod = self.pipeline.transform(train, meta)[_TOD_COLUMN].to_numpy()

        self._base_rate = float(train[meta.reward_column].mean())
        self._variant_rate = (
            train.groupby(meta.variant_column)[meta.reward_column].mean().to_dict()
        )

        grp = (
            pd.DataFrame(
                {
                    "tod": tod,
                    "variant": train[meta.variant_column].to_numpy(),
                    "reward": train[meta.reward_column].to_numpy(dtype=float),
                },
            )
            .groupby(["tod", "variant"])["reward"]
            .agg(["sum", "count"])
        )
        variant_prior = grp.index.get_level_values("variant").map(self._variant_rate).to_numpy()
        k = self.prior_strength
        grp["rate"] = (grp["sum"].to_numpy() + k * variant_prior) / (grp["count"].to_numpy() + k)
        self._table = (
            grp.reset_index()
            .pivot_table(index="tod", columns="variant", values="rate", aggfunc="mean")
            .reindex(columns=meta.variant_ids)
        )

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        meta = self.meta
        if meta is None:
            raise RuntimeError("call fit() before score_variants()")
        if not self._variant_rate:
            # Cold start: no history yet -> abstain; ScoredPolicy.recommend picks at random.
            return None

        variant_ids = list(meta.variant_ids)
        row = np.array([self._variant_rate.get(v, self._base_rate) for v in variant_ids])
        scores = np.tile(row, (len(contexts), 1))
        if self._table is None:
            return scores

        # Identical transform at serve time keeps train/serve features consistent.
        tod = self.pipeline.transform(contexts, meta)[_TOD_COLUMN]
        known = tod.isin(self._table.index).to_numpy()
        if known.any():
            looked = self._table.reindex(tod[known]).to_numpy()
            block = scores[known]
            scores[known] = np.where(np.isnan(looked), block, looked)
        return scores

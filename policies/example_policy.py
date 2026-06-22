import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DatasetMeta
from engine.features import clean_categorical
from engine.policy import IterationContext, ScoredPolicy

class ExamplePolicy(ScoredPolicy):

    def __init__(
        self,
        seed: int = 0,
        segment_column: str | None = None,
        prior_strength: float = 20.0,
    ) -> None:
        super().__init__(seed=seed)
        self.segment_column = segment_column
        self.prior_strength = prior_strength
        self._seg_col: str | None = None
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
        self._seg_col = self.segment_column or (
            meta.categorical_columns[0] if meta.categorical_columns else None
        )
        if len(train) == 0:
            self._table = None
            self._variant_rate = {}
            return

        self._base_rate = float(train[meta.reward_column].mean())
        self._variant_rate = (
            train.groupby(meta.variant_column)[meta.reward_column].mean().to_dict()
        )
        if self._seg_col is None:
            self._table = None
            return

        seg = clean_categorical(train[self._seg_col])
        grp = (
            pd.DataFrame(
                {
                    "seg": seg,
                    "variant": train[meta.variant_column].to_numpy(),
                    "reward": train[meta.reward_column].to_numpy(dtype=float),
                },
            )
            .groupby(["seg", "variant"])["reward"]
            .agg(["sum", "count"])
        )
        variant_prior = grp.index.get_level_values("variant").map(self._variant_rate).to_numpy()
        k = self.prior_strength
        grp["rate"] = (grp["sum"].to_numpy() + k * variant_prior) / (grp["count"].to_numpy() + k)
        self._table = (
            grp.reset_index()
            .pivot_table(index="seg", columns="variant", values="rate", aggfunc="mean")
            .reindex(columns=meta.variant_ids)
        )

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        meta = self.meta
        if meta is None:
            raise RuntimeError("call fit() before score_variants()")
        if not self._variant_rate:
            return None
        variant_ids = list(meta.variant_ids)
        row = np.array([self._variant_rate.get(v, self._base_rate) for v in variant_ids])
        scores = np.tile(row, (len(contexts), 1))
        if self._table is None or self._seg_col not in contexts.columns:
            return scores
        seg = clean_categorical(contexts[self._seg_col])
        known = seg.isin(self._table.index).to_numpy()
        if known.any():
            looked = self._table.reindex(seg[known]).to_numpy()
            block = scores[known]
            scores[known] = np.where(np.isnan(looked), block, looked)
        return scores

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DatasetMeta
from engine.policy import IterationContext, ScoredPolicy

class MyPolicy(ScoredPolicy):

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._variant_rate: dict[str, float] = {}
        self._base_rate = 0.0

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        context: IterationContext | None = None,
    ) -> None:
        self.meta = meta
        if len(train) == 0:
            self._variant_rate = {}
            return
        self._base_rate = float(train[meta.reward_column].mean())
        self._variant_rate = (
            train.groupby(meta.variant_column)[meta.reward_column].mean().to_dict()
        )

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        meta = self.meta
        if meta is None:
            raise RuntimeError("call fit() before score_variants()")
        if not self._variant_rate:
            return None
        row = np.array([self._variant_rate.get(v, self._base_rate) for v in meta.variant_ids])
        return np.tile(row, (len(contexts), 1))

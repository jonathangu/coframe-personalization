from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DatasetMeta

@dataclass(frozen=True)
class IterationSummary:

    iteration: int
    n_train: int
    n_window: int
    window_start: str
    window_end: str
    observed_reward: float

@dataclass(frozen=True)
class IterationContext:

    iteration: int
    n_seen: int
    past: list[IterationSummary] = field(default_factory=list)

class Policy(ABC):

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        context: IterationContext | None = None,
    ) -> None:
        pass

    @abstractmethod
    def recommend(
        self,
        contexts: pd.DataFrame,
        available_variants: list[str] | None = None,
    ) -> NDArray[np.object_]:
        pass

    def score_variants(self, _contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        return None

class ScoredPolicy(Policy):

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.meta: DatasetMeta | None = None
        self._rng = np.random.default_rng(seed)

    def recommend(
        self,
        contexts: pd.DataFrame,
        available_variants: list[str] | None = None,
    ) -> NDArray[np.object_]:
        meta = self.meta
        if meta is None:
            raise RuntimeError("call fit() before recommend()")
        choices = available_variants or list(meta.variant_ids)
        scores = self.score_variants(contexts)
        if scores is None:
            idx = self._rng.integers(0, len(choices), size=len(contexts))
            return np.asarray(choices, dtype=object)[idx]
        scores = np.asarray(scores, dtype=float)
        allowed = set(choices)
        mask = np.array([v not in allowed for v in meta.variant_ids])
        scores[:, mask] = -np.inf
        return np.asarray(meta.variant_ids)[scores.argmax(axis=1)]

class RandomPolicy(Policy):

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._variant_ids: list[str] = []

    def fit(
        self,
        _train: pd.DataFrame,
        meta: DatasetMeta,
        _context: IterationContext | None = None,
    ) -> None:
        self._variant_ids = list(meta.variant_ids)

    def recommend(
        self,
        contexts: pd.DataFrame,
        available_variants: list[str] | None = None,
    ) -> NDArray[np.object_]:
        choices = available_variants or self._variant_ids
        idx = self._rng.integers(0, len(choices), size=len(contexts))
        return np.asarray(choices, dtype=object)[idx]

class BestFixedVariantPolicy(Policy):

    def __init__(self, seed: int = 0) -> None:
        self._rates: pd.Series | None = None
        self._variant_ids: list[str] = []
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        _context: IterationContext | None = None,
    ) -> None:
        self._variant_ids = list(meta.variant_ids)
        if len(train) == 0:
            self._rates = None
        else:
            self._rates = train.groupby(meta.variant_column)[meta.reward_column].mean()

    def recommend(
        self,
        contexts: pd.DataFrame,
        available_variants: list[str] | None = None,
    ) -> NDArray[np.object_]:
        choices = available_variants or self._variant_ids
        ranked = None if self._rates is None else self._rates.reindex(choices).dropna()
        if ranked is None or ranked.empty:
            idx = self._rng.integers(0, len(choices), size=len(contexts))
            return np.asarray(choices, dtype=object)[idx]
        return np.full(len(contexts), str(ranked.idxmax()), dtype=object)

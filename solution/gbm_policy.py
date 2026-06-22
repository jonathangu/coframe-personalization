"""Gradient-boosted T-learner ablation for the personalization harness.

This policy is intentionally registered as an experiment, not as the default
workhorse. It checks whether the additive EB policy is leaving nonlinear
context/action signal on the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from engine.datasets import DatasetMeta
from engine.features import clean_categorical
from engine.policy import IterationContext, Policy, ScoredPolicy
from solution.eb_policy import AdditiveEBPolicy


_UNKNOWN = "__UNKNOWN__"
_OTHER = "__OTHER__"


@dataclass
class _ArmModel:
    prior: float
    n_rows: int
    model: HistGradientBoostingClassifier | None = None


class GBMTLearnerPolicy(ScoredPolicy):
    """Per-arm histogram gradient boosting outcome model.

    For each variant, fit a binary classifier on rows where that variant was
    shown. At serve time, score every row under every arm and choose the highest
    predicted conversion probability after the base ScoredPolicy masks to the
    live action set.
    """

    def __init__(
        self,
        seed: int = 0,
        max_train_rows: int = 90_000,
        max_category_levels: int = 120,
        min_arm_rows: int = 250,
        prior_strength: float = 100.0,
        prediction_shrinkage: float = 600.0,
        max_iter: int = 45,
        max_leaf_nodes: int = 15,
        learning_rate: float = 0.07,
        l2_regularization: float = 0.25,
        include_time_features: bool = True,
        drop_features: tuple[str, ...] = ("color_scheme",),
    ) -> None:
        super().__init__(seed=seed)
        self.max_train_rows = max_train_rows
        self.max_category_levels = max_category_levels
        self.min_arm_rows = min_arm_rows
        self.prior_strength = prior_strength
        self.prediction_shrinkage = prediction_shrinkage
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.include_time_features = include_time_features
        self.drop_features = set(drop_features)

        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._cat_levels: dict[str, list[str]] = {}
        self._cat_maps: dict[str, dict[str, int]] = {}
        self._num_medians: dict[str, float] = {}
        self._time_origin: pd.Timestamp | None = None
        self._base_rate = 0.0
        self._models: dict[str, _ArmModel] = {}

    def _resolve_features(self, meta: DatasetMeta) -> None:
        self._cat_cols = [c for c in meta.categorical_columns if c not in self.drop_features]
        self._num_cols = [c for c in meta.numeric_columns if c not in self.drop_features]

    def _fit_encoder(self, train: pd.DataFrame, meta: DatasetMeta) -> None:
        self._cat_levels = {}
        self._cat_maps = {}
        self._num_medians = {}

        for col in self._cat_cols:
            values = clean_categorical(train[col])
            levels = values.value_counts(dropna=False).head(self.max_category_levels).index.astype(str).tolist()
            if _UNKNOWN not in levels:
                levels.append(_UNKNOWN)
            if _OTHER not in levels:
                levels.append(_OTHER)
            self._cat_levels[col] = levels
            self._cat_maps[col] = {v: i for i, v in enumerate(levels)}

        for col in self._num_cols:
            vals = pd.to_numeric(train[col], errors="coerce")
            median = float(vals.median()) if vals.notna().any() else 0.0
            self._num_medians[col] = median

        if self.include_time_features and len(train):
            self._time_origin = pd.to_datetime(train[meta.timestamp_column], utc=True).min()
        else:
            self._time_origin = None

    def _transform(self, df: pd.DataFrame, meta: DatasetMeta) -> NDArray[np.float64]:
        blocks: list[NDArray[np.float64]] = []

        for col in self._cat_cols:
            values = clean_categorical(df[col]).astype(str)
            mapping = self._cat_maps[col]
            other_idx = mapping[_OTHER]
            encoded = values.map(mapping).fillna(other_idx).to_numpy(dtype=float)
            blocks.append(encoded.reshape(-1, 1))

        for col in self._num_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(self._num_medians[col])
            blocks.append(vals.to_numpy(dtype=float).reshape(-1, 1))

        if self.include_time_features:
            ts = pd.to_datetime(df[meta.timestamp_column], utc=True)
            hour = ts.dt.hour.to_numpy(dtype=float).reshape(-1, 1)
            dow = ts.dt.dayofweek.to_numpy(dtype=float).reshape(-1, 1)
            if self._time_origin is None:
                age = np.zeros((len(df), 1), dtype=float)
            else:
                age = ((ts - self._time_origin).dt.total_seconds() / 86_400.0).to_numpy(dtype=float)
                age = age.reshape(-1, 1)
            blocks.extend([hour, dow, age])

        if not blocks:
            return np.zeros((len(df), 1), dtype=float)
        return np.hstack(blocks)

    def _categorical_mask(self) -> list[bool]:
        n_time = 3 if self.include_time_features else 0
        return [True] * len(self._cat_cols) + [False] * len(self._num_cols) + [False] * n_time

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        _context: IterationContext | None = None,
    ) -> None:
        self.meta = meta
        self._resolve_features(meta)
        self._models = {}

        if len(train) == 0:
            return

        if self.max_train_rows and len(train) > self.max_train_rows:
            train = train.tail(self.max_train_rows)

        reward = train[meta.reward_column].to_numpy(dtype=float)
        self._base_rate = float(reward.mean()) if len(reward) else 0.0
        self._fit_encoder(train, meta)
        x_all = self._transform(train, meta)
        categorical_features = self._categorical_mask()

        variants = train[meta.variant_column].astype(str).to_numpy()
        for variant in meta.variant_ids:
            mask = variants == str(variant)
            y = reward[mask]
            n_rows = int(mask.sum())
            reward_sum = float(y.sum())
            prior = (reward_sum + self.prior_strength * self._base_rate) / (
                n_rows + self.prior_strength
            )

            if n_rows < self.min_arm_rows or len(np.unique(y)) < 2:
                self._models[variant] = _ArmModel(prior=prior, n_rows=n_rows, model=None)
                continue

            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=30,
                l2_regularization=self.l2_regularization,
                categorical_features=categorical_features,
                early_stopping=True,
                random_state=self.seed,
            )
            model.fit(x_all[mask], y.astype(int))
            self._models[variant] = _ArmModel(prior=prior, n_rows=n_rows, model=model)

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        meta = self.meta
        if meta is None:
            raise RuntimeError("call fit() before score_variants()")
        if not self._models:
            return None

        x = self._transform(contexts, meta)
        scores = np.zeros((len(contexts), len(meta.variant_ids)), dtype=float)
        for j, variant in enumerate(meta.variant_ids):
            arm = self._models.get(variant)
            if arm is None:
                scores[:, j] = self._base_rate
                continue
            if arm.model is None:
                pred = np.full(len(contexts), arm.prior, dtype=float)
            else:
                pred = arm.model.predict_proba(x)[:, 1]
                reliability = arm.n_rows / (arm.n_rows + self.prediction_shrinkage)
                pred = reliability * pred + (1.0 - reliability) * arm.prior
            scores[:, j] = pred

        return scores


class HybridEBGBMPolicy(Policy):
    """Support-adaptive policy: EB early/sparse, GBM when enough data exists."""

    def __init__(
        self,
        seed: int = 0,
        gbm_min_train_rows: int = 50_000,
        min_rows_per_arm: int = 2_000,
    ) -> None:
        self.seed = seed
        self.gbm_min_train_rows = gbm_min_train_rows
        self.min_rows_per_arm = min_rows_per_arm
        self._eb = AdditiveEBPolicy(seed=seed)
        self._gbm = GBMTLearnerPolicy(seed=seed)
        self._active: Policy = self._eb

    def _has_gbm_support(self, train: pd.DataFrame, meta: DatasetMeta) -> bool:
        if len(train) < self.gbm_min_train_rows:
            return False
        counts = train[meta.variant_column].value_counts()
        return bool(counts.reindex(meta.variant_ids).fillna(0).min() >= self.min_rows_per_arm)

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        context: IterationContext | None = None,
    ) -> None:
        if self._has_gbm_support(train, meta):
            self._active = self._gbm
        else:
            self._active = self._eb
        self._active.fit(train, meta, context)

    def recommend(
        self,
        contexts: pd.DataFrame,
        available_variants: list[str] | None = None,
    ) -> NDArray[np.object_]:
        return self._active.recommend(contexts, available_variants=available_variants)

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        return self._active.score_variants(contexts)

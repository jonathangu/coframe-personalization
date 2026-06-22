"""LightGBM T-learner policies for the personalization harness.

LightGBM is the production-shaped boosted-tree backend here: faster and more
configurable than sklearn's histogram GBM, while preserving the same direct
reward-modeling contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray

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
    model: LGBMClassifier | None = None


class LightGBMTLearnerPolicy(ScoredPolicy):
    """Per-arm LightGBM outcome model.

    It directly predicts reward for every `(context, arm)` pair:
    `P(reward=1 | context=x, arm=a)`. The base `ScoredPolicy` then masks to the
    currently available variants and returns the argmax.
    """

    def __init__(
        self,
        seed: int = 0,
        max_train_rows: int = 120_000,
        max_category_levels: int = 160,
        min_arm_rows: int = 250,
        prior_strength: float = 100.0,
        prediction_shrinkage: float = 500.0,
        n_estimators: int = 90,
        learning_rate: float = 0.055,
        num_leaves: int = 31,
        min_child_samples: int = 35,
        reg_alpha: float = 0.02,
        reg_lambda: float = 0.45,
        include_time_features: bool = True,
        drop_features: tuple[str, ...] = ("color_scheme",),
    ) -> None:
        super().__init__(seed=seed)
        self.max_train_rows = max_train_rows
        self.max_category_levels = max_category_levels
        self.min_arm_rows = min_arm_rows
        self.prior_strength = prior_strength
        self.prediction_shrinkage = prediction_shrinkage
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.include_time_features = include_time_features
        self.drop_features = set(drop_features)

        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._cat_maps: dict[str, dict[str, int]] = {}
        self._num_medians: dict[str, float] = {}
        self._time_origin: pd.Timestamp | None = None
        self._base_rate = 0.0
        self._models: dict[str, _ArmModel] = {}

    def _resolve_features(self, meta: DatasetMeta) -> None:
        self._cat_cols = [c for c in meta.categorical_columns if c not in self.drop_features]
        self._num_cols = [c for c in meta.numeric_columns if c not in self.drop_features]

    def _fit_encoder(self, train: pd.DataFrame, meta: DatasetMeta) -> None:
        self._cat_maps = {}
        self._num_medians = {}

        for col in self._cat_cols:
            values = clean_categorical(train[col])
            levels = values.value_counts(dropna=False).head(self.max_category_levels).index.astype(str).tolist()
            if _UNKNOWN not in levels:
                levels.append(_UNKNOWN)
            if _OTHER not in levels:
                levels.append(_OTHER)
            self._cat_maps[col] = {v: i for i, v in enumerate(levels)}

        for col in self._num_cols:
            vals = pd.to_numeric(train[col], errors="coerce")
            self._num_medians[col] = float(vals.median()) if vals.notna().any() else 0.0

        self._time_origin = (
            pd.to_datetime(train[meta.timestamp_column], utc=True).min()
            if self.include_time_features and len(train)
            else None
        )

    def _transform(self, df: pd.DataFrame, meta: DatasetMeta) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self._cat_cols:
            values = clean_categorical(df[col]).astype(str)
            mapping = self._cat_maps[col]
            other_idx = mapping[_OTHER]
            out[col] = values.map(mapping).fillna(other_idx).astype("int32")

        for col in self._num_cols:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(self._num_medians[col]).astype(float)

        if self.include_time_features:
            ts = pd.to_datetime(df[meta.timestamp_column], utc=True)
            out["_hour"] = ts.dt.hour.astype("int16")
            out["_dow"] = ts.dt.dayofweek.astype("int16")
            if self._time_origin is None:
                out["_age_days"] = 0.0
            else:
                out["_age_days"] = (ts - self._time_origin).dt.total_seconds() / 86_400.0

        return out

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
        categorical_features = [c for c in self._cat_cols if c in x_all.columns]

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

            model = LGBMClassifier(
                objective="binary",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=0.92,
                subsample_freq=1,
                colsample_bytree=0.92,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.seed,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(
                x_all.loc[mask],
                y.astype(int),
                categorical_feature=categorical_features,
            )
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


class HybridEBLightGBMPolicy(Policy):
    """Support-adaptive EB -> LightGBM policy."""

    def __init__(
        self,
        seed: int = 0,
        lgbm_min_train_rows: int = 35_000,
        min_rows_per_arm: int = 1_500,
    ) -> None:
        self.seed = seed
        self.lgbm_min_train_rows = lgbm_min_train_rows
        self.min_rows_per_arm = min_rows_per_arm
        self._eb = AdditiveEBPolicy(seed=seed)
        self._lgbm = LightGBMTLearnerPolicy(seed=seed)
        self._active: Policy = self._eb

    def _has_lgbm_support(self, train: pd.DataFrame, meta: DatasetMeta) -> bool:
        if len(train) < self.lgbm_min_train_rows:
            return False
        counts = train[meta.variant_column].value_counts()
        return bool(counts.reindex(meta.variant_ids).fillna(0).min() >= self.min_rows_per_arm)

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        context: IterationContext | None = None,
    ) -> None:
        self._active = self._lgbm if self._has_lgbm_support(train, meta) else self._eb
        self._active.fit(train, meta, context)

    def recommend(
        self,
        contexts: pd.DataFrame,
        available_variants: list[str] | None = None,
    ) -> NDArray[np.object_]:
        return self._active.recommend(contexts, available_variants=available_variants)

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        return self._active.score_variants(contexts)

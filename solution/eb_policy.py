"""Empirical-Bayes additive T-learner for the personalization harness.

This policy treats the task as a one-shot contextual bandit: estimate
P(convert | context, variant) for every available arm, then choose the best arm.

The model is deliberately simple and production-shaped:
- per-arm response surfaces, so arm-specific flips are explicit;
- additive segment effects in centered log-odds space;
- empirical-Bayes shrinkage, so thin or noisy cells fall back to stable priors;
- optional recency weighting for drift experiments;
- no truth-file access and no train-window leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from engine.datasets import DatasetMeta
from engine.features import clean_categorical
from engine.policy import IterationContext, ScoredPolicy

_EPS = 1e-6
_CROSS_SEP = "__x__"


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


@dataclass
class _DevTable:
    """Lookup table: feature value -> per-arm centered-logit deviation."""

    index: dict[str, int]
    dev: NDArray[np.float64]


class AdditiveEBPolicy(ScoredPolicy):
    """Additive empirical-Bayes segment policy.

    `score_variants()` returns columns in canonical `meta.variant_ids` order.
    This is important: `ScoredPolicy.recommend()` masks unavailable variants by
    indexing those canonical columns, not by `available_variants` order.
    """

    def __init__(
        self,
        seed: int = 0,
        features: list[str] | None = None,
        crosses: list[tuple[str, str]] | None = None,
        prior_strength: float = 75.0,
        cross_prior_strength: float = 200.0,
        arm_prior_strength: float = 20.0,
        n_num_bins: int = 6,
        recency_halflife_days: float | None = None,
        drop_features: tuple[str, ...] = ("color_scheme",),
    ) -> None:
        super().__init__(seed=seed)
        self._cfg_features = features
        self._cfg_crosses = crosses
        self.prior_strength = prior_strength
        self.cross_prior_strength = cross_prior_strength
        self.arm_prior_strength = arm_prior_strength
        self.n_num_bins = n_num_bins
        self.recency_halflife_days = recency_halflife_days
        self.drop_features = set(drop_features)

        self._cold = True
        self._global_logit: NDArray[np.float64] | None = None
        self._tables: dict[str, _DevTable] = {}
        self._mains: list[str] = []
        self._crosses: list[tuple[str, str]] = []
        self._bin_edges: dict[str, NDArray[np.float64]] = {}

    def _resolve_features(self, meta: DatasetMeta) -> None:
        if self._cfg_features is not None:
            self._mains = [c for c in self._cfg_features if c in meta.feature_columns]
        else:
            self._mains = [
                c
                for c in (*meta.categorical_columns, *meta.numeric_columns)
                if c not in self.drop_features
            ]

        if self._cfg_crosses is not None:
            self._crosses = [
                (a, b)
                for (a, b) in self._cfg_crosses
                if a in meta.feature_columns and b in meta.feature_columns
            ]
            return

        base = "country" if "country" in self._mains else (self._mains[0] if self._mains else None)
        if base is None:
            self._crosses = []
        else:
            self._crosses = [
                (base, c)
                for c in self._mains
                if c != base and c in {"language", "device_type", "platform"}
            ]

    def _col_values(self, df: pd.DataFrame, col: str, meta: DatasetMeta) -> pd.Series:
        """Return a categorical string view, with numeric columns train-binned."""
        if col in meta.numeric_columns and col in self._bin_edges:
            edges = self._bin_edges[col]
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            idx = np.digitize(x, edges)
            out = pd.Series([f"b{i}" for i in idx], index=df.index)
            out[np.isnan(x)] = "Unknown"
            return out
        return clean_categorical(df[col])

    def _featurize(self, df: pd.DataFrame, meta: DatasetMeta) -> dict[str, pd.Series]:
        cols = {c: self._col_values(df, c, meta) for c in self._mains}
        feats: dict[str, pd.Series] = dict(cols)
        for a, b in self._crosses:
            feats[f"{a}{_CROSS_SEP}{b}"] = cols[a].str.cat(cols[b], sep="||")
        return feats

    def fit(
        self,
        train: pd.DataFrame,
        meta: DatasetMeta,
        _context: IterationContext | None = None,
    ) -> None:
        self.meta = meta
        self._resolve_features(meta)
        self._tables = {}
        self._bin_edges = {}

        if len(train) == 0:
            self._cold = True
            self._global_logit = None
            return
        self._cold = False

        variant_ids = list(meta.variant_ids)
        arm_pos = {v: i for i, v in enumerate(variant_ids)}
        n_arms = len(variant_ids)

        for col in self._mains:
            if col not in meta.numeric_columns:
                continue
            x = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
            x = x[~np.isnan(x)]
            if x.size and np.unique(x).size > self.n_num_bins:
                qs = np.linspace(0, 1, self.n_num_bins + 1)[1:-1]
                self._bin_edges[col] = np.unique(np.quantile(x, qs))

        arm = train[meta.variant_column].to_numpy()
        reward = train[meta.reward_column].to_numpy(dtype=float)

        if self.recency_halflife_days is not None:
            ts = pd.to_datetime(train[meta.timestamp_column], utc=True).astype("int64").to_numpy()
            age_days = (ts.max() - ts) / 8.64e13
            weights = 0.5 ** (age_days / self.recency_halflife_days)
        else:
            weights = np.ones(len(train))

        base = float(np.sum(weights * reward) / max(np.sum(weights), _EPS))
        prior = np.full(n_arms, base)
        for variant, arm_index in arm_pos.items():
            mask = arm == variant
            weight_n = float(weights[mask].sum())
            reward_sum = float((weights[mask] * reward[mask]).sum())
            prior[arm_index] = (
                reward_sum + self.arm_prior_strength * base
            ) / (weight_n + self.arm_prior_strength)
        self._global_logit = _logit(prior)

        feats = self._featurize(train, meta)
        arm_idx = np.array([arm_pos[a] for a in arm])

        for feat_name, feat_values in feats.items():
            k = self.cross_prior_strength if _CROSS_SEP in feat_name else self.prior_strength
            frame = pd.DataFrame(
                {
                    "value": feat_values.to_numpy(),
                    "arm_idx": arm_idx,
                    "weight": weights,
                    "weighted_reward": weights * reward,
                },
            )
            grouped = frame.groupby(["value", "arm_idx"], sort=False)[
                ["weight", "weighted_reward"]
            ].sum()
            arm_priors = prior[grouped.index.get_level_values("arm_idx").to_numpy()]
            shrunk = (
                grouped["weighted_reward"].to_numpy() + k * arm_priors
            ) / (grouped["weight"].to_numpy() + k)
            dev = _logit(shrunk) - _logit(arm_priors)
            table = (
                pd.Series(dev, index=grouped.index)
                .unstack("arm_idx")
                .reindex(columns=range(n_arms))
                .fillna(0.0)
            )
            self._tables[feat_name] = _DevTable(
                index={v: i for i, v in enumerate(table.index.tolist())},
                dev=table.to_numpy(dtype=float),
            )

    def score_variants(self, contexts: pd.DataFrame) -> NDArray[np.float64] | None:
        meta = self.meta
        if meta is None:
            raise RuntimeError("call fit() before score_variants()")
        if self._cold or self._global_logit is None:
            return None

        n_rows = len(contexts)
        n_arms = len(meta.variant_ids)
        scores = np.tile(self._global_logit, (n_rows, 1))

        feats = self._featurize(contexts, meta)
        for feat_name, table in self._tables.items():
            positions = feats[feat_name].map(table.index).to_numpy()
            known = ~pd.isna(positions)
            if known.any():
                lookup_idx = positions[known].astype(int)
                scores[known] += table.dev[lookup_idx]

        if scores.shape != (n_rows, n_arms):
            raise RuntimeError("score matrix must be in meta.variant_ids order")
        return scores

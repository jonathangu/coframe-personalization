from abc import ABC, abstractmethod

import pandas as pd

from engine.datasets import DatasetMeta

UNKNOWN = "Unknown"

def clean_categorical(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), UNKNOWN).astype(str)

class FeaturePipeline(ABC):

    @abstractmethod
    def fit(self, train: pd.DataFrame, meta: DatasetMeta) -> None:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, meta: DatasetMeta) -> pd.DataFrame:
        pass

    def categorical_outputs(self, _meta: DatasetMeta) -> list[str]:
        return []

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VARIANT_COLUMN = "variant_id"
REWARD_COLUMN = "reward"
TIMESTAMP_COLUMN = "timestamp"
PROPENSITY_COLUMN = "propensity"

_RESERVED = {VARIANT_COLUMN, REWARD_COLUMN, TIMESTAMP_COLUMN, PROPENSITY_COLUMN, "row_id"}

@dataclass(frozen=True)
class DatasetMeta:

    name: str
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    datetime_columns: list[str]
    variant_ids: list[str]
    variant_column: str = VARIANT_COLUMN
    reward_column: str = REWARD_COLUMN
    timestamp_column: str = TIMESTAMP_COLUMN
    propensity_column: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

def infer_meta(name: str, df: pd.DataFrame) -> DatasetMeta:
    feature_columns = [c for c in df.columns if c not in _RESERVED]
    categorical: list[str] = []
    numeric: list[str] = []
    datetime_cols: list[str] = []
    for c in feature_columns:
        dt = df[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            datetime_cols.append(c)
        elif pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
            numeric.append(c)
        else:
            categorical.append(c)
    return DatasetMeta(
        name=name,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        numeric_columns=numeric,
        datetime_columns=datetime_cols,
        variant_ids=sorted(df[VARIANT_COLUMN].unique().tolist()),
        propensity_column=PROPENSITY_COLUMN if PROPENSITY_COLUMN in df.columns else None,
    )

def list_datasets() -> list[str]:
    truths = {p.name[: -len("_truth.parquet")] for p in DATA_DIR.glob("*_truth.parquet")}
    logs = {p.stem for p in DATA_DIR.glob("*.parquet") if not p.stem.endswith("_truth")}
    return sorted(truths & logs)

def load_dataset(name: str) -> tuple[pd.DataFrame, DatasetMeta]:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"unknown dataset {name!r}; available: {list_datasets()}")
    df = pd.read_parquet(path)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True)
    df = df.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    return df, infer_meta(name, df)

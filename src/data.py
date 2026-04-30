"""Student-owned dataset loading contract.

Implements ``load_dataset_split`` for the Craigslist Used Cars regression
project. The processed dataset is produced by ``notebooks/02_cleaning.ipynb``
and stored at ``data/vehicles_processed.parquet`` as a single parquet file
containing all rows + a ``_split`` column ({"train", "test"}) + the target
column ``price``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "vehicles_processed.parquet"
TARGET = "price"
SPLIT_COL = "_split"


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return ``(X_train, X_test, y_train, y_test)`` for model evaluation.

    The processed parquet is fully numerical (all categoricals encoded,
    numerical features scaled), so it can be fed directly to any model
    exposing a ``.predict(X)`` method.
    """
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}. "
            "Run notebooks/02_cleaning.ipynb first to generate it."
        )

    df = pd.read_parquet(PROCESSED_PATH)

    train = df[df[SPLIT_COL] == "train"].drop(columns=[SPLIT_COL])
    test = df[df[SPLIT_COL] == "test"].drop(columns=[SPLIT_COL])

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    return X_train, X_test, y_train, y_test

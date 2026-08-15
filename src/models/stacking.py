"""Out-of-fold calibrated stacking across independent model families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibratedStack:
    model_names: tuple[str, ...]
    stacker: LogisticRegression

    @classmethod
    def fit(cls, oof_predictions: pd.DataFrame, target: pd.Series) -> CalibratedStack:
        if oof_predictions.isna().any().any():
            raise ValueError("Stacking requires complete out-of-fold predictions")
        if not oof_predictions.index.equals(target.index):
            raise ValueError("OOF predictions and targets must share an index")
        transformed = np.log(
            np.clip(oof_predictions, 1e-6, 1 - 1e-6) / np.clip(1 - oof_predictions, 1e-6, 1)
        )
        model = LogisticRegression(C=0.1, max_iter=2_000).fit(transformed, target)
        return cls(tuple(oof_predictions.columns), model)

    def predict(self, predictions: pd.DataFrame) -> np.ndarray:
        if tuple(predictions.columns) != self.model_names:
            raise ValueError("Stack input models or order do not match training")
        transformed = np.log(
            np.clip(predictions, 1e-6, 1 - 1e-6) / np.clip(1 - predictions, 1e-6, 1)
        )
        return self.stacker.predict_proba(transformed)[:, 1]

"""The Synchronous Sieve: SMOTE + XGBoost + isotonic calibration.

Clears the vast majority of transactions inside the payment gateway's
100-300 ms authorization budget (single-row inference is ~1 ms).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import AegisConfig


class PlattCalibrator:
    """Logistic calibration on the log-odds of the raw score.

    Smooth and monotone — unlike isotonic regression it does not collapse to a
    coarse step function when the calibration split holds only a few dozen
    positives (0.172% prevalence).
    """

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        logit = np.log(np.clip(raw, 1e-9, 1 - 1e-9) / np.clip(1 - raw, 1e-9, 1))
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(logit.reshape(-1, 1), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        logit = np.log(np.clip(raw, 1e-9, 1 - 1e-9) / np.clip(1 - raw, 1e-9, 1))
        return self.lr.predict_proba(logit.reshape(-1, 1))[:, 1]


class CalibratedSieve:
    """XGBoost wrapped with Platt calibration fitted on a clean hold-out.

    SMOTE distorts the base rate, so raw XGBoost scores are not probabilities.
    Calibration matters here: the ambiguity band (0.5 +/- delta) is defined on
    P(fraud), so miscalibrated scores would route the wrong transactions.
    """

    def __init__(self, model: XGBClassifier, calib: PlattCalibrator, features: list[str]):
        self.model = model
        self.calib = calib
        self.features = features

    def _as_matrix(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.features].to_numpy()
        return np.asarray(X)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.model.predict_proba(self._as_matrix(X))[:, 1]
        return np.clip(self.calib.predict(raw), 1e-6, 1 - 1e-6)


def train_sieve(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_calib: pd.DataFrame,
    y_calib: np.ndarray,
    cfg: AegisConfig,
) -> tuple[CalibratedSieve, pd.Series]:
    features = list(X_train.columns)

    print(f"[sieve] SMOTE oversampling (ratio={cfg.smote_ratio}) ...")
    smote = SMOTE(sampling_strategy=cfg.smote_ratio, random_state=cfg.seed)
    X_res, y_res = smote.fit_resample(X_train.to_numpy(), y_train)
    print(f"[sieve] resampled: {len(y_res):,} rows, {int(y_res.sum()):,} fraud")

    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=cfg.seed,
    )
    print("[sieve] training XGBoost ...")
    model.fit(X_res, y_res)

    raw_calib = model.predict_proba(X_calib.to_numpy())[:, 1]
    calib = PlattCalibrator().fit(raw_calib, y_calib)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )
    return CalibratedSieve(model, calib, features), importances


def benchmark_latency(sieve: CalibratedSieve, X: pd.DataFrame, n: int = 400) -> dict:
    """Single-transaction inference latency — the number that has to sit
    inside the ~130 ms Mastercard authorization window."""
    rows = X.sample(n=min(n, len(X)), random_state=0)
    # warm-up
    sieve.predict_proba(rows.iloc[[0]])
    times = []
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        t0 = time.perf_counter()
        sieve.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "max_ms": float(times.max()),
        "n": int(len(times)),
    }

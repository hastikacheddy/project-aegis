"""Dataset loading for Aegis.

Primary source: the European Cardholder dataset (Dal Pozzolo et al., 2015) —
284,807 transactions, 0.172% fraud — fetched from OpenML and cached locally.
If the download is unavailable (offline demo), a statistically faithful
synthetic twin is generated so the full pipeline always runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, ensure_dirs

CACHE = DATA_DIR / "creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def load_dataset(prefer_real: bool = True, seed: int = 42) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source). Source is 'openml', 'cache' or 'synthetic'."""
    ensure_dirs()
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        return df, "cache"
    if prefer_real:
        try:
            df = _fetch_openml()
            df.to_csv(CACHE, index=False)
            return df, "openml"
        except Exception as exc:  # offline / registry hiccup -> keep the demo alive
            print(f"[data] OpenML fetch failed ({exc!r}); falling back to synthetic twin.")
    return make_synthetic(seed=seed), "synthetic"


def _fetch_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    print("[data] downloading European Cardholder dataset from OpenML (~150 MB)...")
    bunch = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    df = bunch.frame
    df[TARGET] = df[TARGET].astype(str).astype(int)
    return df[FEATURES + [TARGET]]


def make_synthetic(n: int = 120_000, fraud_rate: float = 0.00172, seed: int = 42) -> pd.DataFrame:
    """Synthetic twin of the European Cardholder dataset.

    Legitimate traffic is standard-normal in the PCA space (V1..V28). Fraud is a
    mixture of 'blatant' cases (large shifts on the known signal features
    V14/V17/V12/V10) and 'subtle' cases carrying a non-linear V14/V17
    anti-correlation signature — this is what creates a genuine ambiguity band
    for the Quantum Adjudicator to resolve.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n * fraud_rate)), 40)
    n_legit = n - n_fraud

    legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    legit_amount = rng.lognormal(3.0, 1.0, size=n_legit)

    fraud = rng.normal(0.0, 1.0, size=(n_fraud, 28))
    n_blatant = int(n_fraud * 0.55)
    idx = {f"V{i}": i - 1 for i in range(1, 29)}

    b = slice(0, n_blatant)
    fraud[b, idx["V14"]] -= rng.uniform(4.0, 7.0, n_blatant)
    fraud[b, idx["V17"]] -= rng.uniform(3.0, 6.0, n_blatant)
    fraud[b, idx["V12"]] -= rng.uniform(2.0, 4.0, n_blatant)
    fraud[b, idx["V10"]] -= rng.uniform(2.0, 4.0, n_blatant)
    fraud[b, idx["V4"]] += rng.uniform(2.0, 4.0, n_blatant)

    s = slice(n_blatant, n_fraud)
    n_subtle = n_fraud - n_blatant
    sign = rng.choice([-1.0, 1.0], size=n_subtle)
    fraud[s, idx["V14"]] = sign * 1.8 + rng.normal(0, 0.6, n_subtle)
    fraud[s, idx["V17"]] = -sign * 1.6 + rng.normal(0, 0.6, n_subtle)
    fraud[s, idx["V10"]] -= rng.uniform(0.5, 1.5, n_subtle)
    fraud[s, idx["V12"]] -= rng.uniform(0.0, 1.0, n_subtle)
    fraud_amount = rng.lognormal(4.0, 1.2, size=n_fraud)

    X = np.vstack([legit, fraud])
    amount = np.concatenate([legit_amount, fraud_amount])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    time = np.sort(rng.uniform(0, 172_800, size=n))

    order = rng.permutation(n)
    df = pd.DataFrame(X[order], columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount[order]
    df[TARGET] = y[order]
    return df


def stratified_subsample(X: pd.DataFrame, y: np.ndarray, n: int, seed: int = 42):
    """Sub-sample preserving the fraud/non-fraud ratio (challenge rule)."""
    if len(X) <= n:
        return X, y
    from sklearn.model_selection import train_test_split

    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_sub, y_sub

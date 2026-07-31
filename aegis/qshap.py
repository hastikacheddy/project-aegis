"""Quantum SHAP: exact Shapley attribution over the quantum feature space.

    phi_i = sum_{S subset F\\{i}}  |S|!(|F|-|S|-1)!/|F|!  [ f_Q(S u {i}) - f_Q(S) ]

With k = 4 adjudication features the 2^k coalition values are computed
exactly (no sampling approximation), translating the quantum latent space
into a regulator-readable attribution — model-governance compliant.
"""

from __future__ import annotations

from math import factorial

import numpy as np


def quantum_shap(
    score_fn,
    x: np.ndarray,
    background: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Exact interventional Shapley values.

    score_fn : callable mapping (n, k) raw-feature matrix -> (n,) scores
    x        : the transaction's quantum features, shape (k,)
    background : reference vector (median legitimate transaction), shape (k,)
    """
    k = len(x)
    n_coalitions = 1 << k

    # evaluate all coalitions in a single batched call
    coalition_matrix = np.empty((n_coalitions, k))
    for mask in range(n_coalitions):
        z = background.copy()
        for i in range(k):
            if mask & (1 << i):
                z[i] = x[i]
        coalition_matrix[mask] = z
    values = np.asarray(score_fn(coalition_matrix), dtype=float)

    phis = np.zeros(k)
    for i in range(k):
        bit = 1 << i
        for mask in range(n_coalitions):
            if mask & bit:
                continue
            s = bin(mask).count("1")
            weight = factorial(s) * factorial(k - s - 1) / factorial(k)
            phis[i] += weight * (values[mask | bit] - values[mask])

    return {
        "phi": {name: float(p) for name, p in zip(feature_names, phis)},
        "base_value": float(values[0]),
        "prediction": float(values[n_coalitions - 1]),
        "efficiency_gap": float(values[n_coalitions - 1] - values[0] - phis.sum()),
    }

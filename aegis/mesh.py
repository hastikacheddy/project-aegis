"""The Asynchronous Quantum-Classical Mesh.

Synchronous path : CalibratedSieve -> APPROVE / DECLINE inside the latency
                   budget for the confident majority of traffic.
Asynchronous path: transactions inside the ambiguity band are provisionally
                   approved-with-hold and queued for the Quantum Adjudicator,
                   whose verdict lands out-of-band (post-auth review, step-up
                   authentication, settlement hold) — quantum advantage without
                   quantum latency on the authorization path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .classical import CalibratedSieve
from .qshap import quantum_shap
from .quantum import QuantumAdjudicator


@dataclass
class SieveVerdict:
    p_fraud: float
    decision: str          # APPROVE | DECLINE | AMBIGUOUS
    latency_ms: float


class AegisMesh:
    def __init__(
        self,
        sieve: CalibratedSieve,
        adjudicator: QuantumAdjudicator,
        band_lo: float,
        band_hi: float,
        blend_w: float,
        background: np.ndarray,
        verdict_threshold: float = 0.5,
    ):
        self.sieve = sieve
        self.adjudicator = adjudicator
        self.band_lo = band_lo
        self.band_hi = band_hi
        self.blend_w = blend_w          # weight on the quantum score inside the band
        self.background = background    # median legitimate quantum-feature vector
        self.verdict_threshold = verdict_threshold  # picked on validation F1

    # ---------------------------------------------------------------- sync path

    def sieve_decision(self, row: pd.DataFrame) -> SieveVerdict:
        t0 = time.perf_counter()
        p = float(self.sieve.predict_proba(row)[0])
        latency = (time.perf_counter() - t0) * 1000.0
        if p < self.band_lo:
            decision = "APPROVE"
        elif p > self.band_hi:
            decision = "DECLINE"
        else:
            decision = "AMBIGUOUS"
        return SieveVerdict(p_fraud=p, decision=decision, latency_ms=latency)

    # --------------------------------------------------------------- async path

    def adjudicate(self, row: pd.DataFrame, p_sieve: float, with_shap: bool = True) -> dict:
        t0 = time.perf_counter()
        xq = row[self.adjudicator.feature_names].to_numpy()[0]
        comp = self.adjudicator.score_components(xq.reshape(1, -1))
        q = float(comp["q_score"][0])
        final = self.blend_w * q + (1.0 - self.blend_w) * p_sieve
        result = {
            "p_vqc": float(comp["p_vqc"][0]),
            "qae_error": float(comp["qae_error"][0]),
            "anomaly": float(comp["anomaly"][0]),
            "q_score": q,
            "p_sieve": p_sieve,
            "final_score": final,
            "verdict": "DECLINE" if final >= self.verdict_threshold else "APPROVE",
        }
        if with_shap:
            result["shap"] = quantum_shap(
                self.adjudicator.q_scores,
                xq,
                self.background.copy(),
                self.adjudicator.feature_names,
            )
        result["adjudication_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------- batch scoring

    def batch_final_scores(self, X: pd.DataFrame, p_sieve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised mesh scores for offline evaluation.

        Inside the band, transactions are re-ranked by the blended
        quantum+sieve score, then mapped back onto the [band_lo, band_hi]
        interval by rank. The quantum verdict reorders the ambiguous tier
        without ever disturbing the calibrated ordering outside it.

        Returns (final_scores, ambiguous_mask).
        """
        final = p_sieve.copy()
        mask = (p_sieve >= self.band_lo) & (p_sieve <= self.band_hi)
        m = int(mask.sum())
        if m > 0:
            Xq = X.loc[mask, self.adjudicator.feature_names].to_numpy()
            q = self.adjudicator.q_scores(Xq)
            blended = self.blend_w * q + (1.0 - self.blend_w) * p_sieve[mask]
            ranks = blended.argsort().argsort()
            final[mask] = self.band_lo + (self.band_hi - self.band_lo) * (ranks + 0.5) / m
        return final, mask

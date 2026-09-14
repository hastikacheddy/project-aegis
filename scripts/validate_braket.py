"""Validate the locally-trained Quantum Adjudicator on Amazon Braket.

Training stays local (hardware training would be slow and expensive); this
script re-runs the *trained* circuits on a Braket backend and measures
agreement with the local statevector — the "noise-aware validation" step of
the hardware strategy.

    python scripts/validate_braket.py                        # DM1 simulator (cheap)
    python scripts/validate_braket.py --backend ionq-forte --confirm --n 2

Every run prints a cost estimate first, and QPU backends refuse to submit
without --confirm. Check the estimate before confirming: at 1000 shots a
single IonQ Forte circuit is ~$80.

Prerequisites: AWS credentials configured (aws configure), and:
  pip install amazon-braket-sdk amazon-braket-pennylane-plugin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.config import ARTIFACTS
from aegis.costs import (PRICING_DATE, QPU_SHOT_PRICE, MeshWorkload, qpu_cost,
                         simulator_cost)
from aegis.quantum import QuantumAdjudicator

QPU_BACKENDS = ("ionq-forte", "iqm-garnet", "rigetti-cepheus")
BACKENDS = ("braket-sv1", "braket-dm1") + QPU_BACKENDS


def estimate_cost(backend: str, n: int, shots: int) -> float:
    """Dollar estimate for this run, using the shared cost model."""
    wl = MeshWorkload(transactions=n, routed_fraction=1.0, shots=shots,
                      with_shap=False)
    if backend in QPU_SHOT_PRICE:
        return qpu_cost(wl, backend)["total"]
    return simulator_cost(wl, "DM1")["total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="braket-dm1", choices=BACKENDS)
    ap.add_argument("--n", type=int, default=10, help="transactions to validate")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--confirm", action="store_true",
                    help="required for QPU backends, which bill real money per shot")
    args = ap.parse_args()

    estimate = estimate_cost(args.backend, args.n, args.shots)
    print(f"[braket] backend={args.backend}  n={args.n}  shots={args.shots}")
    print(f"[braket] estimated cost for this run: ${estimate:,.2f} "
          f"({args.n} transactions x 2 circuits, list prices as of {PRICING_DATE})")
    if args.backend in QPU_BACKENDS and not args.confirm:
        sys.exit(f"[braket] refusing to submit to {args.backend} without --confirm "
                 f"(estimated ${estimate:,.2f})")

    run_config = json.loads((ARTIFACTS / "run_config.json").read_text())
    q_features = run_config["quantum_features"]

    pool = pd.read_csv(ARTIFACTS / "demo_pool.csv")
    rows = pool.sample(n=args.n, random_state=7)
    X = rows[q_features].to_numpy()
    y = rows["Class"].to_numpy()

    local = QuantumAdjudicator.load(ARTIFACTS, backend="local")
    remote = QuantumAdjudicator.load(ARTIFACTS, backend=args.backend)

    print("[braket] scoring locally ...")
    s_local = local.q_scores(X)
    print("[braket] submitting to Braket (each row = 2 circuits) ...")
    s_remote = np.empty(args.n)
    for i in range(args.n):
        s_remote[i] = remote.q_scores(X[i:i + 1])[0]
        print(f"  txn {i + 1}/{args.n}: local={s_local[i]:.4f}  "
              f"remote={s_remote[i]:.4f}  (Class={y[i]})")

    diff = np.abs(s_local - s_remote)
    tau = run_config.get("verdict_threshold", 0.5)
    agree = float(((s_local >= tau) == (s_remote >= tau)).mean())
    report = {
        "backend": args.backend,
        "n": args.n,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "verdict_agreement": agree,
        "scores_local": s_local.tolist(),
        "scores_remote": s_remote.tolist(),
    }
    out = ARTIFACTS / f"braket_validation_{args.backend}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[braket] mean |local - remote| = {diff.mean():.4f}  "
          f"max = {diff.max():.4f}  verdict agreement = {agree:.0%}")
    print(f"[braket] report written to {out}")


if __name__ == "__main__":
    main()

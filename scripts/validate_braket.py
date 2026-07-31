"""Validate the locally-trained Quantum Adjudicator on Amazon Braket.

Training stays local (hardware training would be slow and expensive); this
script re-runs the *trained* circuits on a Braket backend and measures
agreement with the local statevector — the "noise-aware validation" step of
the hardware strategy.

    python scripts/validate_braket.py                       # DM1 simulator (cheap)
    python scripts/validate_braket.py --backend ionq-aria --confirm --n 2

Prerequisites: AWS credentials configured (aws configure), region us-east-1,
and:  pip install amazon-braket-sdk amazon-braket-pennylane-plugin
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
from aegis.quantum import QuantumAdjudicator

# rough public list prices (verify in the Braket console before large runs)
COST_NOTE = {
    "braket-dm1": "DM1 bills per simulation-minute (~$0.075/min, seconds per task); "
                  "this run should cost well under $1.",
    "ionq-aria": "IonQ Aria bills per task + per shot (~$0.30 + $0.03/shot). At 1000 "
                 "shots, EACH circuit is ~$30 and each adjudication runs 2 circuits.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="braket-dm1", choices=["braket-dm1", "ionq-aria"])
    ap.add_argument("--n", type=int, default=10, help="transactions to validate")
    ap.add_argument("--confirm", action="store_true",
                    help="required for ionq-aria (real money per shot)")
    args = ap.parse_args()

    print(f"[braket] backend={args.backend}  n={args.n}")
    print(f"[braket] cost note: {COST_NOTE[args.backend]}")
    if args.backend == "ionq-aria" and not args.confirm:
        sys.exit("[braket] refusing to submit to IonQ Aria without --confirm "
                 f"(estimated ~${args.n * 2 * 30:.0f} at 1000 shots)")

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

"""Precompute a demo stream for the static (serverless) build of the dashboard.

Every value written here is produced by the trained models exactly as the live
server produces it: real calibrated sieve scores, real VQC and QAE circuit
evaluations, real exact Quantum SHAP. The static page replays these records
instead of calling a backend, which is what lets the demo run on hosting that
cannot execute Python.

    python scripts/export_static_demo.py --out build/static_demo
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from aegis.config import ARTIFACTS
from aegis.mesh import AegisMesh
from aegis.quantum import QuantumAdjudicator

FEED_MIX = {"legit": 0.70, "fraud": 0.08, "band": 0.22}


def build_mesh():
    sieve = joblib.load(ARTIFACTS / "sieve.joblib")
    adjudicator = QuantumAdjudicator.load(ARTIFACTS)
    cfg = json.loads((ARTIFACTS / "run_config.json").read_text())
    mesh = AegisMesh(
        sieve, adjudicator, cfg["band_lo"], cfg["band_hi"], cfg["blend_w"],
        np.array(cfg["background"]), cfg.get("verdict_threshold", 0.5),
    )
    return mesh, cfg


def split_pools(mesh, cfg):
    pool = pd.read_csv(ARTIFACTS / "demo_pool.csv")
    p = mesh.sieve.predict_proba(pool.drop(columns=["Class"]))
    in_band = (p >= cfg["band_lo"]) & (p <= cfg["band_hi"])
    if in_band.sum() < 20:
        in_band = np.abs(p - 0.5) <= np.sort(np.abs(p - 0.5))[min(19, len(p) - 1)]
    return {
        "fraud": pool[pool["Class"] == 1],
        "legit": pool[(pool["Class"] == 0) & ~in_band],
        "band": pool[in_band],
    }


def score_one(mesh, row_full, txn_id: int) -> dict:
    """Score one transaction through the mesh, recording every real output."""
    truth = int(row_full["Class"].iloc[0])
    row = row_full.drop(columns=["Class"])
    verdict = mesh.sieve_decision(row)

    rec = {
        "id": f"TXN-{txn_id:06d}",
        "amount": round(float(row["Amount"].iloc[0]), 2),
        "p_sieve": round(verdict.p_fraud, 4),
        "latency_ms": round(verdict.latency_ms, 2),
        "truth": truth,
        "ambiguous": verdict.decision == "AMBIGUOUS",
        "sieve_decision": verdict.decision,
    }
    if verdict.decision == "AMBIGUOUS":
        a = mesh.adjudicate(row, verdict.p_fraud, with_shap=True)
        rec["adjudication"] = {
            "p_vqc": round(a["p_vqc"], 4),
            "qae_error": round(a["qae_error"], 4),
            "anomaly": round(a["anomaly"], 4),
            "q_score": round(a["q_score"], 4),
            "final_score": round(a["final_score"], 4),
            "verdict": a["verdict"],
            "adjudication_ms": round(a["adjudication_ms"], 1),
            "shap": {
                "phi": {k: round(v, 4) for k, v in a["shap"]["phi"].items()},
                "base_value": round(a["shap"]["base_value"], 4),
            },
        }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/static_demo")
    ap.add_argument("--stream", type=int, default=400, help="replay sequence length")
    ap.add_argument("--pool", type=int, default=40, help="per-kind inject pool size")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mesh, cfg = build_mesh()
    pools = split_pools(mesh, cfg)
    print(f"[static] pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))

    def sample(kind: str) -> pd.DataFrame:
        pool = pools[kind]
        return pool.iloc[[rng.randrange(len(pool))]]

    kinds, weights = list(FEED_MIX), list(FEED_MIX.values())
    counter = 0
    stream = []
    for _ in range(args.stream):
        counter += 1
        kind = rng.choices(kinds, weights=weights)[0]
        stream.append(score_one(mesh, sample(kind), counter))
    print(f"[static] stream: {len(stream)} txns, "
          f"{sum(r['ambiguous'] for r in stream)} adjudicated, "
          f"{sum(r['truth'] for r in stream)} fraud")

    inject = {}
    for kind in ("legit", "band", "fraud"):
        recs = []
        for _ in range(args.pool):
            counter += 1
            recs.append(score_one(mesh, sample(kind), counter))
        inject[kind] = recs
    print("[static] inject pools: " + ", ".join(f"{k}={len(v)}" for k, v in inject.items()))

    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    payload = {
        "generated_from": "trained models, real circuit evaluations",
        "band": [cfg["band_lo"], cfg["band_hi"]],
        "blend_w": cfg["blend_w"],
        "quantum_features": cfg["quantum_features"],
        "metrics": metrics,
        "stream": stream,
        "inject": inject,
        "next_id": counter + 1,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "demo_data.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[static] written {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

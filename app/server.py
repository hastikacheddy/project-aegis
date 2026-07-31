"""Project Aegis — live demo server.

Simulates a payment-gateway feed against the trained mesh:
  - every transaction hits the Synchronous Sieve (real model, real latency)
  - ambiguous transactions are queued to the asynchronous Quantum Adjudicator
    (VQC + QAE + Quantum SHAP) running out-of-band in a worker
  - the dashboard polls /api/state for a live snapshot

Run from the repo root:  uvicorn app.server:app --port 8021
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from aegis.config import ARTIFACTS
from aegis.mesh import AegisMesh
from aegis.quantum import QuantumAdjudicator

# demo feed is enriched for visibility (declared in the UI): real prevalence is
# 0.172% fraud and <1% ambiguity, which would make for a very boring 3 minutes
FEED_MIX = {"legit": 0.70, "fraud": 0.08, "band": 0.22}

state: dict = {
    "txns": deque(maxlen=48),
    "adjudications": deque(maxlen=6),
    "stats": {
        "total": 0, "approved": 0, "declined": 0,
        "queued": 0, "adjudicated": 0,
        "frauds_blocked": 0, "false_declines": 0,
        "sync_latencies": deque(maxlen=200),
    },
    "queue_depth": 0,
}
mesh: AegisMesh | None = None
pools: dict = {}
metrics: dict = {}
run_config: dict = {}
adjudication_queue: asyncio.Queue | None = None
_txn_counter = 0


def load_artifacts():
    global mesh, metrics, run_config
    sieve = joblib.load(ARTIFACTS / "sieve.joblib")
    adjudicator = QuantumAdjudicator.load(ARTIFACTS)
    run_config = json.loads((ARTIFACTS / "run_config.json").read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    mesh = AegisMesh(
        sieve, adjudicator,
        run_config["band_lo"], run_config["band_hi"], run_config["blend_w"],
        np.array(run_config["background"]),
        run_config.get("verdict_threshold", 0.5),
    )
    pool = pd.read_csv(ARTIFACTS / "demo_pool.csv")
    # score the pool once so the feed can be enriched with genuinely ambiguous
    # transactions (otherwise a sharp sieve routes <1% and judges see nothing)
    p = sieve.predict_proba(pool.drop(columns=["Class"]))
    in_band = (p >= run_config["band_lo"]) & (p <= run_config["band_hi"])
    if in_band.sum() < 20:
        in_band = np.abs(p - 0.5) <= np.sort(np.abs(p - 0.5))[min(19, len(p) - 1)]
    pools["fraud"] = pool[pool["Class"] == 1]
    pools["legit"] = pool[(pool["Class"] == 0) & ~in_band]
    pools["band"] = pool[in_band]
    print(f"[server] loaded mesh: band=({run_config['band_lo']:.2f},{run_config['band_hi']:.2f}) "
          f"blend_w={run_config['blend_w']}  pool: {len(pools['fraud'])} fraud / "
          f"{len(pools['legit'])} legit / {len(pools['band'])} ambiguous")


def sample_row(kind: str) -> pd.DataFrame:
    pool = pools[kind]
    return pool.iloc[[random.randrange(len(pool))]]


async def process_txn(kind: str):
    global _txn_counter
    row_full = sample_row(kind)
    truth = int(row_full["Class"].iloc[0])
    row = row_full.drop(columns=["Class"])
    _txn_counter += 1
    txn_id = f"TXN-{_txn_counter:06d}"

    verdict = mesh.sieve_decision(row)
    s = state["stats"]
    s["total"] += 1
    s["sync_latencies"].append(verdict.latency_ms)

    txn = {
        "id": txn_id,
        "ts": time.strftime("%H:%M:%S"),
        "amount": round(float(row["Amount"].iloc[0]), 2),
        "p_sieve": round(verdict.p_fraud, 4),
        "latency_ms": round(verdict.latency_ms, 2),
        "truth": truth,
        "status": verdict.decision,   # APPROVE | DECLINE | AMBIGUOUS
        "final_score": None,
        "verdict": None,
    }

    if verdict.decision == "APPROVE":
        s["approved"] += 1
    elif verdict.decision == "DECLINE":
        s["declined"] += 1
        if truth == 1:
            s["frauds_blocked"] += 1
        else:
            s["false_declines"] += 1
    else:
        txn["status"] = "Q_PENDING"
        s["queued"] += 1
        state["queue_depth"] += 1
        await adjudication_queue.put((txn, row, verdict.p_fraud))

    state["txns"].appendleft(txn)


async def quantum_worker():
    loop = asyncio.get_running_loop()
    while True:
        txn, row, p_sieve = await adjudication_queue.get()
        try:
            result = await loop.run_in_executor(
                None, lambda: mesh.adjudicate(row, p_sieve, with_shap=True)
            )
            s = state["stats"]
            s["adjudicated"] += 1
            state["queue_depth"] = max(0, state["queue_depth"] - 1)
            txn["final_score"] = round(result["final_score"], 4)
            txn["verdict"] = result["verdict"]
            txn["status"] = "Q_DECLINE" if result["verdict"] == "DECLINE" else "Q_APPROVE"
            if result["verdict"] == "DECLINE":
                s["declined"] += 1
                if txn["truth"] == 1:
                    s["frauds_blocked"] += 1
                else:
                    s["false_declines"] += 1
            else:
                s["approved"] += 1
            state["adjudications"].appendleft({
                "id": txn["id"],
                "ts": time.strftime("%H:%M:%S"),
                "amount": txn["amount"],
                "truth": txn["truth"],
                "p_sieve": round(p_sieve, 4),
                "p_vqc": round(result["p_vqc"], 4),
                "anomaly": round(result["anomaly"], 4),
                "qae_error": round(result["qae_error"], 4),
                "q_score": round(result["q_score"], 4),
                "final_score": round(result["final_score"], 4),
                "verdict": result["verdict"],
                "adjudication_ms": round(result["adjudication_ms"], 1),
                "shap": result["shap"],
            })
        except Exception as exc:
            txn["status"] = "Q_ERROR"
            print(f"[worker] adjudication failed: {exc!r}")
        finally:
            adjudication_queue.task_done()


async def feed():
    await asyncio.sleep(1.0)
    kinds = list(FEED_MIX.keys())
    weights = list(FEED_MIX.values())
    while True:
        kind = random.choices(kinds, weights=weights)[0]
        try:
            await process_txn(kind)
        except Exception as exc:
            print(f"[feed] error: {exc!r}")
        await asyncio.sleep(random.uniform(0.5, 1.4))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global adjudication_queue
    load_artifacts()
    adjudication_queue = asyncio.Queue()
    tasks = [asyncio.create_task(feed()), asyncio.create_task(quantum_worker())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Project Aegis", lifespan=lifespan)


@app.get("/api/state")
async def get_state():
    s = state["stats"]
    lat = list(s["sync_latencies"])
    return JSONResponse({
        "stats": {
            "total": s["total"],
            "approved": s["approved"],
            "declined": s["declined"],
            "queued": s["queued"],
            "adjudicated": s["adjudicated"],
            "frauds_blocked": s["frauds_blocked"],
            "false_declines": s["false_declines"],
            "sync_cleared_pct": (1 - s["queued"] / s["total"]) if s["total"] else 1.0,
            "latency_p50": float(np.percentile(lat, 50)) if lat else 0.0,
            "latency_p99": float(np.percentile(lat, 99)) if lat else 0.0,
        },
        "queue_depth": state["queue_depth"],
        "txns": list(state["txns"]),
        "adjudications": list(state["adjudications"]),
        "band": [run_config.get("band_lo"), run_config.get("band_hi")],
        "blend_w": run_config.get("blend_w"),
        "quantum_features": run_config.get("quantum_features", []),
        "metrics": metrics,
    })


@app.post("/api/inject/{kind}")
async def inject(kind: str):
    if kind not in ("fraud", "legit", "band"):
        return JSONResponse({"error": "kind must be fraud|legit|band"}, status_code=400)
    await process_txn(kind)
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

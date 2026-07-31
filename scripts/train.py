"""End-to-end Aegis training pipeline.

    python scripts/train.py              # real data (OpenML, cached) if available
    python scripts/train.py --synthetic  # force the synthetic twin
    python scripts/train.py --quick      # fast smoke-test settings

Produces everything the live demo and the judges need in ./artifacts:
models, metrics.json, plots, and a demo transaction pool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split

from aegis.classical import benchmark_latency, train_sieve
from aegis.config import ARTIFACTS, PLOTS, AegisConfig, ensure_dirs
from aegis.data import TARGET, load_dataset, make_synthetic, stratified_subsample
from aegis.mesh import AegisMesh
from aegis.metrics import (auprc, plot_band_shift, plot_latency_hist,
                           plot_loss, plot_pr_curves, save_metrics)
from aegis.quantum import QuantumAdjudicator


def build_quantum_training_set(p_train, y_train, lo, hi, cfg):
    """Positional indices of the Quantum Adjudicator's training rows.

    Core set: everything inside the ambiguity band. If the sieve is so sharp
    that the band holds too few frauds (or too few legit rows for the QAE
    manifold), supplement with the hardest out-of-band rows — those nearest
    P=0.5, i.e. the next-most-ambiguous cases. All frauds are retained; legit
    rows are capped by ambiguity so the set stays within max_quantum_train.
    """
    dist = np.abs(p_train - (lo + hi) / 2.0)
    in_band = (p_train >= lo) & (p_train <= hi)
    idx = set(np.where(in_band)[0].tolist())

    for cls, floor in ((1, cfg.min_band_frauds), (0, cfg.min_band_legit)):
        have = sum(1 for i in idx if y_train[i] == cls)
        if have < floor:
            candidates = np.where((y_train == cls) & ~in_band)[0]
            extra = candidates[np.argsort(dist[candidates])][: floor - have]
            idx.update(extra.tolist())

    idx = np.array(sorted(idx))
    fraud_idx = idx[y_train[idx] == 1]
    legit_idx = idx[y_train[idx] == 0]
    max_legit = max(cfg.max_quantum_train - len(fraud_idx), cfg.min_band_legit)
    if len(legit_idx) > max_legit:
        legit_idx = legit_idx[np.argsort(dist[legit_idx])][:max_legit]
    return np.sort(np.concatenate([fraud_idx, legit_idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="force synthetic twin dataset")
    ap.add_argument("--quick", action="store_true", help="fast smoke-test settings")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--backend", default="local", help="local | braket-dm1 | ionq-aria")
    args = ap.parse_args()

    cfg = AegisConfig()
    if args.quick:
        cfg.n_estimators = 150
        cfg.epochs = 12
        cfg.max_quantum_train = 240
    if args.epochs:
        cfg.epochs = args.epochs

    ensure_dirs()
    t_start = time.time()

    # ------------------------------------------------------------------ data
    if args.synthetic:
        df, source = make_synthetic(n=40_000 if args.quick else 120_000, seed=cfg.seed), "synthetic"
    else:
        df, source = load_dataset(seed=cfg.seed)
        if args.quick and len(df) > 60_000:
            X_sub, y_sub = stratified_subsample(
                df.drop(columns=[TARGET]), df[TARGET].to_numpy(), 60_000, cfg.seed
            )
            df = X_sub.assign(**{TARGET: y_sub})
    y = df[TARGET].to_numpy()
    X = df.drop(columns=[TARGET])
    print(f"[data] source={source}  rows={len(df):,}  fraud={y.sum():,} ({y.mean():.4%})")

    # 70 / 10 / 20 stratified split: train / calibration / test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=cfg.seed
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=cfg.seed
    )
    print(f"[data] train={len(X_train):,}  calib={len(X_calib):,}  test={len(X_test):,}")

    # ------------------------------------------------- 1. synchronous sieve
    sieve, importances = train_sieve(X_train, y_train, X_calib, y_calib, cfg)
    latency = benchmark_latency(sieve, X_test)
    print(f"[sieve] latency p50={latency['p50_ms']:.2f} ms  p99={latency['p99_ms']:.2f} ms")

    p_train = sieve.predict_proba(X_train)
    p_test = sieve.predict_proba(X_test)
    sieve_auprc = auprc(y_test, p_test)
    print(f"[sieve] test AUPRC = {sieve_auprc:.4f}")

    # ------------------------------------------------- 2. ambiguity routing
    # thresholds from the *training* score distribution (volume budgets)
    hi = float(np.quantile(p_train, 1.0 - cfg.decline_traffic))
    lo = float(np.quantile(p_train, 1.0 - cfg.decline_traffic - cfg.review_traffic))
    band_mask_test = (p_test >= lo) & (p_test <= hi)
    routed_pct = float(band_mask_test.mean())
    n_band_fraud = int(y_test[band_mask_test].sum())
    print(f"[mesh] ambiguity band = ({lo:.4f}, {hi:.4f})  "
          f"routes {routed_pct:.2%} of test traffic "
          f"({int(band_mask_test.sum())} txns, {n_band_fraud} fraud = "
          f"{n_band_fraud / max(int(y_test.sum()), 1):.1%} of all test fraud)")

    # top-k fraud-salient features (Time excluded: not a behavioural signal)
    q_features = [f for f in importances.index if f != "Time"][: cfg.n_qubits]
    print(f"[quantum] adjudication features -> qubits: {q_features}")

    # quantum training set: band rows, supplemented/capped by ambiguity
    q_idx = build_quantum_training_set(p_train, y_train, lo, hi, cfg)
    Xq_all = X_train.iloc[q_idx][q_features]
    yq_all = y_train[q_idx]
    p_sieve_q = p_train[q_idx]

    Xq_tr, Xq_val, yq_tr, yq_val, pq_tr, pq_val = train_test_split(
        Xq_all.to_numpy(), yq_all, p_sieve_q, test_size=0.25,
        stratify=yq_all if yq_all.sum() >= 8 else None, random_state=cfg.seed,
    )
    print(f"[quantum] train={len(yq_tr)} (fraud {int(yq_tr.sum())})  "
          f"val={len(yq_val)} (fraud {int(yq_val.sum())})")

    # ------------------------------------------- 3. quantum adjudicator fit
    adjudicator = QuantumAdjudicator(
        feature_names=q_features,
        n_layers=cfg.n_layers,
        n_trash=cfg.n_trash,
        alpha=cfg.alpha,
        backend=args.backend,
        seed=cfg.seed,
    )
    adjudicator.fit(Xq_tr, yq_tr, epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)

    # ------------------------------------------- 4. mesh fusion (blend on val)
    q_val = adjudicator.q_scores(Xq_val)
    blend_w, best_ap = 0.5, -1.0
    if yq_val.sum() > 0 and yq_val.sum() < len(yq_val):
        for w in cfg.blend_grid:
            ap = auprc(yq_val, w * q_val + (1 - w) * pq_val)
            print(f"[mesh] blend w={w:.2f} -> val band AUPRC {ap:.4f}")
            # on ties (tiny val sets tie easily) prefer the most central w:
            # a hedged blend is more robust than either corner
            if ap > best_ap + 1e-9 or (abs(ap - best_ap) <= 1e-9
                                       and abs(w - 0.5) < abs(blend_w - 0.5)):
                best_ap, blend_w = ap, w
    print(f"[mesh] selected quantum blend weight = {blend_w}")

    # verdict threshold for the async DECLINE/APPROVE call, picked on val F1
    verdict_threshold = 0.5
    if yq_val.sum() > 0 and yq_val.sum() < len(yq_val):
        blended_val = blend_w * q_val + (1 - blend_w) * pq_val
        best_f1 = -1.0
        for t in np.unique(blended_val):
            pred = blended_val >= t
            tp = int((pred & (yq_val == 1)).sum())
            fp = int((pred & (yq_val == 0)).sum())
            fn = int((~pred & (yq_val == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, verdict_threshold = f1, float(t)
        print(f"[mesh] verdict threshold = {verdict_threshold:.4f} (val F1 {best_f1:.3f})")

    background = np.median(Xq_tr[yq_tr == 0], axis=0)
    mesh = AegisMesh(sieve, adjudicator, lo, hi, blend_w, background, verdict_threshold)

    # ------------------------------------------------------- 5. evaluation
    final_test, mask = mesh.batch_final_scores(X_test, p_test)
    mesh_auprc = auprc(y_test, final_test)

    band_metrics = {}
    yb = y_test[band_mask_test]
    if band_mask_test.any() and 0 < yb.sum() < len(yb):
        q_band = adjudicator.q_scores(X_test.loc[band_mask_test, q_features].to_numpy())
        band_metrics = {
            "sieve_auprc": auprc(yb, p_test[band_mask_test]),
            "quantum_auprc": auprc(yb, q_band),
            "mesh_auprc": auprc(yb, final_test[band_mask_test]),
            "n": int(band_mask_test.sum()),
            "n_fraud": int(yb.sum()),
        }

    # ------------------------------------------------------------ 6. plots
    plot_pr_curves(
        {"Classical sieve (XGBoost)": (y_test, p_test),
         "Aegis mesh (quantum-adjudicated)": (y_test, final_test)},
        PLOTS / "pr_overall.png", "Precision-Recall — full test set",
    )
    if band_metrics:
        plot_pr_curves(
            {"Sieve": (yb, p_test[band_mask_test]),
             "Quantum Adjudicator": (yb, q_band),
             "Mesh (blended)": (yb, final_test[band_mask_test])},
            PLOTS / "pr_band.png", "Precision-Recall — ambiguity band only",
        )
    lat_samples = np.array([mesh.sieve_decision(X_test.iloc[[i]]).latency_ms
                            for i in range(0, min(300, len(X_test)))])
    plot_latency_hist(lat_samples, 50.0, PLOTS / "latency.png")
    plot_band_shift(p_test, final_test, y_test, lo, hi, PLOTS / "band_shift.png")
    plot_loss(adjudicator.loss_history, PLOTS / "quantum_loss.png")

    # -------------------------------------------------------- 7. artifacts
    joblib.dump(sieve, ARTIFACTS / "sieve.joblib")
    adjudicator.save(ARTIFACTS)

    run_config = {
        "dataset_source": source,
        "band_lo": lo, "band_hi": hi,
        "blend_w": blend_w,
        "verdict_threshold": verdict_threshold,
        "quantum_features": q_features,
        "background": background.tolist(),
        "feature_order": list(X.columns),
        "config": cfg.to_dict(),
    }
    (ARTIFACTS / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # demo pool: every test fraud + a slice of legit traffic
    import pandas as pd

    fraud_rows = X_test[y_test == 1].copy()
    legit_rows = X_test[y_test == 0].sample(n=min(4000, int((y_test == 0).sum())),
                                            random_state=cfg.seed).copy()
    demo = pd.concat([fraud_rows.assign(Class=1), legit_rows.assign(Class=0)])
    demo.to_csv(ARTIFACTS / "demo_pool.csv", index=False)

    metrics = {
        "dataset": {"source": source, "rows": int(len(df)),
                    "fraud": int(y.sum()), "fraud_rate": float(y.mean())},
        "sieve": {"test_auprc": sieve_auprc, "latency": latency},
        "mesh": {"test_auprc": mesh_auprc,
                 "auprc_uplift": mesh_auprc - sieve_auprc,
                 "routed_pct": routed_pct,
                 "cleared_sync_pct": 1.0 - routed_pct,
                 "band": [lo, hi], "blend_w": blend_w},
        "ambiguity_band": band_metrics,
        "quantum": {"n_qubits": cfg.n_qubits, "n_layers": cfg.n_layers,
                    "alpha": cfg.alpha, "features": q_features,
                    "train_size": int(len(yq_tr)), "epochs": cfg.epochs,
                    "final_loss": adjudicator.loss_history[-1] if adjudicator.loss_history else None,
                    "backend": args.backend,
                    "hardware_target": "IonQ Aria (via Amazon Braket, DM1 for prototyping)"},
        "train_seconds": time.time() - t_start,
    }
    save_metrics(metrics, ARTIFACTS / "metrics.json")

    print("\n" + "=" * 64)
    print("PROJECT AEGIS - TRAINING COMPLETE")
    print("=" * 64)
    print(f"  dataset                 : {source} ({len(df):,} rows)")
    print(f"  sieve test AUPRC        : {sieve_auprc:.4f}")
    print(f"  mesh  test AUPRC        : {mesh_auprc:.4f}  "
          f"({'+' if mesh_auprc >= sieve_auprc else ''}{mesh_auprc - sieve_auprc:.4f})")
    if band_metrics:
        print(f"  band  AUPRC sieve/mesh  : {band_metrics['sieve_auprc']:.4f} "
              f"-> {band_metrics['mesh_auprc']:.4f}")
    print(f"  traffic cleared in sync : {(1 - routed_pct):.2%}  "
          f"(sieve p50 {latency['p50_ms']:.1f} ms)")
    print(f"  artifacts               : {ARTIFACTS}")
    print("=" * 64)


if __name__ == "__main__":
    main()

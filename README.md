# Project Aegis

Hybrid quantum-classical credit card fraud detection that keeps quantum inference off
the synchronous authorization path.

Team Beerantum — HSBC Enterprise Challenge, 2026 Global Quantum + AI Challenge.

## Problem

Payment authorization runs on a hard latency budget: 100–300 ms end to end, with the
Mastercard network averaging around 130 ms. Published quantum fraud models report
competitive offline accuracy, but a parameterized quantum circuit plus cloud queuing
does not fit inside that window. A design that places a QPU call in the authorization
path is not deployable regardless of how well it scores.

Aegis treats this as a routing problem rather than a circuit-depth problem. A calibrated
classical model decides the traffic it is confident about, synchronously. The small
fraction it cannot resolve is queued to a quantum adjudicator whose verdict returns
out-of-band, where it drives post-authorization review, step-up authentication, or a
settlement hold instead of the initial approve/decline.

```
                        ┌─────────────────────────────────────────────┐
 auth request ──────────►  SYNCHRONOUS SIEVE                          │
 (100–300 ms budget)    │  XGBoost + SMOTE + Platt calibration        ├──► APPROVE / DECLINE
                        │  2 ms inference · clears 99.4% of traffic   │    (in-window)
                        └───────────────────┬─────────────────────────┘
                                            │ ambiguity band: 0.6% of traffic, 65% of fraud
                                            ▼  (async queue — off the auth path)
                        ┌─────────────────────────────────────────────┐
                        │  QUANTUM ADJUDICATOR                        │
                        │  VQC + QAE dual loss:                       │
                        │    L(θ,φ) = α·L_VQC(θ) + (1−α)·R_QAE(φ)     ├──► out-of-band verdict:
                        │  + exact Quantum SHAP attribution           │    hold / step-up / release
                        │  PennyLane → Braket DM1 → IonQ Aria         │
                        └─────────────────────────────────────────────┘
```

![Aegis dashboard](docs/dashboard-overview.png)

The demo dashboard, running the trained models against held-out transactions from the
ULB dataset. Green rows cleared the sieve in single-digit milliseconds. Purple
`Q-DECLINED` rows fell in the ambiguity band and were resolved by the quantum
adjudicator asynchronously. The TRUTH column shows the ground-truth label, revealed
after the verdict. Note that the demo feed is deliberately enriched with fraud and
ambiguous cases so there is something to watch, so the "cleared synchronously" figure
shown is much lower than the 99.4% measured on the real traffic distribution.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# train (downloads the ULB dataset from OpenML; falls back to a synthetic
# statistical twin if the download is unavailable)
.venv\Scripts\python scripts\train.py

# run the demo dashboard
.venv\Scripts\python -m uvicorn app.server:app --port 8021
# http://localhost:8021
```

Flags: `--quick` (reduced settings for a smoke test), `--synthetic` (force the offline
dataset), `--epochs N`, `--backend braket-dm1|ionq-aria`.

## Architecture

| Stage | What | How |
|---|---|---|
| Sieve | Synchronous triage, ~2 ms | XGBoost (400 trees, hist) on SMOTE-balanced data, Platt-calibrated on a clean hold-out so routing thresholds are defined on real probabilities |
| Router | Ambiguity band, volume-budgeted | Thresholds are quantiles of the calibrated score: top 0.05% auto-decline, next 0.5% routed to quantum, rest auto-approve. A fixed `0.5 ± δ` margin is nearly empty once scores are calibrated at 0.17% prevalence; budgeting by review capacity also matches how issuers deploy score tiers |
| Adjudicator | VQC + QAE, 4 qubits | Top-4 fraud-salient features → angle embedding → strongly-entangling layers. Dual loss `L = α·L_VQC + (1−α)·R_QAE`: the classifier learns labelled fraud typologies, the autoencoder learns the legitimate manifold, which limits overfitting to a 0.172% minority class |
| Fusion | Mesh score | `w·quantum + (1−w)·sieve` inside the band, `w` selected on a validation split. Scores are rank-rescaled back into the band interval, so adjudication reorders within the band and cannot alter the ordering outside it |
| XAI | Quantum SHAP | Exact Shapley values over all 2⁴ coalitions of the quantum features, batched into a single circuit evaluation |
| Hardware | Braket path | Local statevector for development, Braket DM1 (noise-aware density matrix) for prototyping, IonQ Aria for execution — selected with `--backend` |

### An adjudicated transaction

![Quantum adjudication with SHAP attribution](docs/quantum-adjudication.png)

An $88.23 transaction the sieve scored inside the ambiguity band. The VQC put it at
0.856, the QAE scored it 0.957 anomalous against the legitimate manifold, and the
blended verdict declined it; the ground-truth label confirms fraud. The lower bars are
exact Shapley values over the four quantum features, giving a reviewer a per-feature
account of the circuit's output. The adjudication took 99 ms, off the authorization
path.

## Results

Trained on the ULB European Cardholder dataset (284,807 transactions, 0.1727% fraud),
evaluated on a held-out 20% stratified test split, seed 42. AUPRC is the reported
metric; at this prevalence AUC-ROC is close to uninformative.

| Metric | Classical sieve | Aegis mesh | Δ |
|---|---|---|---|
| Test AUPRC (full) | 0.8795 | 0.8808 | +0.0013 |
| Test AUPRC (ambiguity band) | 0.9402 | 0.9428 | +0.0026 |
| Traffic cleared synchronously | — | 99.40% | p50 2.2 ms |
| Fraud falling in the band | — | 65.3% of all fraud in 0.60% of traffic | — |

![Held-out test set evaluation](docs/evaluation-metrics.png)

The overall AUPRC difference is small. XGBoost is already strong on this dataset, and
because the mesh only reorders within the band, the ceiling on any change to the full
test metric is bounded by the band's size. The result worth reporting is the routing
one: 0.6% of transactions carry 65.3% of the fraud, which is what makes selective
quantum adjudication affordable at all.

On the synthetic twin (`--synthetic`), whose fraud mixture contains non-linear
signatures that trees fit poorly, the band AUPRC gain is larger (0.841 → 0.874). That
dataset is generated, not observed, so it indicates where the approach has headroom
rather than evidencing a result.

Every figure regenerates from `scripts/train.py` into `artifacts/metrics.json`, with
precision-recall curves, a latency histogram, a band score-shift scatter, and the
quantum loss curve written to `artifacts/plots/`.

## Limitations

- Circuits run on a local statevector simulator. Nothing here has been executed on
  Braket or on real hardware yet; `scripts/validate_braket.py` is written for that but
  has not been run against AWS.
- The adjudicator uses 4 qubits and 3 layers, trained on a few hundred ambiguous
  transactions. This is a small circuit on a small sample, not a demonstration of
  quantum advantage.
- Routing thresholds, blend weight, and verdict threshold are fitted on training and
  validation splits. The test split is untouched, but the band is narrow enough that
  band-level metrics rest on 344 transactions and 64 frauds.
- The dataset's features are PCA components, so SHAP attributions name `V14` and `V10`
  rather than anything a human recognises. Real deployment would need the underlying
  feature space.

## Running on Amazon Braket

Training stays local. Braket is used to validate that the trained circuits behave the
same on a noise-aware simulator and on hardware.

1. Enable Braket in the AWS console, including third-party device access. Use
   `us-east-1` for IonQ.
2. Create an IAM user with `AmazonBraketFullAccess`, then `aws configure`.
3. Uncomment the Braket lines in `requirements.txt` and reinstall.
4. Validate on the DM1 simulator: `python scripts/validate_braket.py --backend braket-dm1`
5. Optionally run a few adjudications on IonQ Aria. This costs roughly $30 per circuit
   at 1000 shots and is gated behind an explicit flag:
   `python scripts/validate_braket.py --backend ionq-aria --confirm --n 2`

The script writes local-vs-Braket score agreement to `artifacts/braket_validation_*.json`.

## Repo layout

```
aegis/
  classical.py    SMOTE + XGBoost + Platt calibration (the sieve)
  quantum.py      VQC + QAE dual-loss adjudicator (PennyLane, Braket-ready)
  qshap.py        exact Quantum SHAP
  mesh.py         routing, fusion, batch scoring
  data.py         OpenML loader, synthetic twin, stratified subsampling
  metrics.py      AUPRC evaluation and plots
scripts/
  train.py                end-to-end pipeline → artifacts/
  validate_braket.py      re-run trained circuits on Braket DM1 / IonQ Aria
  capture_screenshots.py  regenerate the README screenshots
app/
  server.py       FastAPI demo server (real models, async quantum worker)
  static/         dashboard, no external dependencies
docs/             README screenshots
artifacts/        metrics.json and plots are committed; models and demo pool are regenerated
```

## References

Nilson Report 2025 · Aite-Novarica 2019 · Mastercard 2012 · Deotte 2019 (IEEE-CIS) ·
Dal Pozzolo et al. 2015 (ULB dataset) · Karimi et al. 2024 (VQC) · Deloitte + AWS 2024
(hybrid QNN on Braket) · Sakhnenko et al. 2021 (QAE) · AWS Braket DM1 and IonQ Aria
documentation.

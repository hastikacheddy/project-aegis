"""Central configuration for the Aegis pipeline."""

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS, PLOTS):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class AegisConfig:
    seed: int = 42

    # --- Synchronous Sieve (XGBoost + SMOTE) ---
    smote_ratio: float = 0.10          # upsample fraud to 10% of majority
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.10

    # --- Ambiguity routing ---
    # The proposal's "high-ambiguity confidence margin delta" is realised as a
    # review-capacity budget, the way issuers actually deploy score tiers:
    # top decline_traffic of scores auto-decline, the next review_traffic are
    # routed to the Quantum Adjudicator, everything below auto-approves.
    # (A fixed 0.4<P<0.6 band is nearly empty at 0.172% prevalence once the
    # scores are calibrated — volume budgeting keeps the band meaningful.)
    decline_traffic: float = 0.0005    # 0.05% of traffic auto-declined
    review_traffic: float = 0.005      # 0.5% routed to quantum adjudication
    # training-set floors: if the band itself is too thin, supplement with the
    # hardest examples (nearest the band centre) so the Adjudicator has signal
    min_band_frauds: int = 12
    min_band_legit: int = 150

    # --- Quantum Adjudicator (VQC + QAE dual loss) ---
    n_qubits: int = 4                  # top-k fraud-salient features -> k qubits
    n_layers: int = 3
    n_trash: int = 2                   # QAE trash qubits (4 -> 2 compression)
    alpha: float = 0.6                 # L = alpha * L_VQC + (1 - alpha) * R_QAE
    epochs: int = 60
    batch_size: int = 64
    lr: float = 0.05
    max_quantum_train: int = 600       # stratified sub-sample cap (challenge rule)

    # --- Mesh fusion ---
    # final ambiguous score = w * quantum + (1 - w) * sieve; w picked on validation
    blend_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blend_grid"] = list(self.blend_grid)
        return d

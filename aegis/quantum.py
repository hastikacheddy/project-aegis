"""The Quantum Adjudicator: VQC + QAE trained under the dual loss

    L(theta, phi) = alpha * L_VQC(theta) + (1 - alpha) * R_QAE(phi)

- The Variational Quantum Classifier (VQC) learns known fraud typologies
  (supervised cross-entropy, class-weighted for the minority class).
- The Quantum Autoencoder (QAE) compresses the legitimate-transaction
  manifold; its reconstruction error R flags anomalies without ever
  overfitting to the tiny fraud class.

Circuits run on PennyLane. The device is swappable: local statevector for
development, Amazon Braket DM1 (noise-aware density matrix) for prototyping,
IonQ Aria for hardware execution with built-in debiasing/sharpening.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import QuantileTransformer

BRAKET_ARNS = {
    "braket-dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "ionq-aria": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
}


def make_device(n_qubits: int, backend: str = "local"):
    if backend == "local":
        return qml.device("default.qubit", wires=n_qubits)
    if backend in BRAKET_ARNS:
        # requires: pip install amazon-braket-pennylane-plugin + AWS credentials
        return qml.device(
            "braket.aws.qubit", device_arn=BRAKET_ARNS[backend], wires=n_qubits, shots=1000
        )
    raise ValueError(f"unknown backend '{backend}' (use local | braket-dm1 | ionq-aria)")


class QuantumAdjudicator:
    """Dual-model quantum scorer over the top-k fraud-salient features."""

    def __init__(
        self,
        feature_names: list[str],
        n_layers: int = 3,
        n_trash: int = 2,
        alpha: float = 0.6,
        backend: str = "local",
        seed: int = 42,
    ):
        self.feature_names = list(feature_names)
        self.n_qubits = len(feature_names)
        self.n_layers = n_layers
        self.n_trash = n_trash
        self.alpha = alpha
        self.backend = backend
        self.seed = seed

        self.scaler: QuantileTransformer | None = None
        self.theta = None  # VQC weights
        self.phi = None    # QAE encoder weights
        self.legit_mu = 0.0
        self.legit_sd = 1.0
        self.loss_history: list[float] = []

        self._build_qnodes()

    # ------------------------------------------------------------------ circuits

    def _build_qnodes(self):
        dev = make_device(self.n_qubits, self.backend)
        wires = list(range(self.n_qubits))
        trash = wires[-self.n_trash:]
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        self._weight_shape = shape

        @qml.qnode(dev, interface="autograd")
        def vqc(x, theta):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(theta, wires=wires)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev, interface="autograd")
        def qae(x, phi):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(phi, wires=wires)
            return qml.probs(wires=trash)

        self._vqc = vqc
        self._qae = qae

    # ------------------------------------------------------------------ training

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, batch_size: int = 64,
            lr: float = 0.05, verbose: bool = True):
        """X: raw feature matrix (n, k) in original units; y: 0/1 labels."""
        rng = np.random.default_rng(self.seed)

        self.scaler = QuantileTransformer(
            n_quantiles=min(200, len(X)), output_distribution="uniform", random_state=self.seed
        )
        Xs = self.scaler.fit_transform(X) * np.pi

        theta = pnp.array(0.1 * rng.standard_normal(self._weight_shape), requires_grad=True)
        phi = pnp.array(0.1 * rng.standard_normal(self._weight_shape), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=lr)

        y = np.asarray(y, dtype=float)
        n_pos = max(y.sum(), 1.0)
        w_pos = float((len(y) - n_pos) / n_pos)

        def cost(th, ph, xb, yb):
            z = self._vqc(xb, th)                       # (B,)
            p = pnp.clip((1.0 - z) / 2.0, 1e-6, 1 - 1e-6)
            bce = -pnp.mean(w_pos * yb * pnp.log(p) + (1.0 - yb) * pnp.log(1.0 - p))
            legit = xb[yb == 0]
            if len(legit) > 0:
                probs = self._qae(legit, ph)            # (B0, 2**n_trash)
                rec = pnp.mean(1.0 - probs[:, 0])       # trash-qubit infidelity
            else:
                rec = 0.0
            return self.alpha * bce + (1.0 - self.alpha) * rec

        n = len(Xs)
        self.loss_history = []
        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = Xs[idx], y[idx]
                (theta, phi), c = opt.step_and_cost(
                    lambda th, ph: cost(th, ph, xb, yb), theta, phi
                )
                epoch_loss += float(c)
                n_batches += 1
            self.loss_history.append(epoch_loss / max(n_batches, 1))
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"[quantum] epoch {epoch + 1:>3}/{epochs}  L = {self.loss_history[-1]:.4f}")

        self.theta = np.array(theta)
        self.phi = np.array(phi)

        # calibrate QAE anomaly score on the legitimate training manifold
        legit_err = self._qae_errors_scaled(Xs[y == 0])
        self.legit_mu = float(legit_err.mean())
        self.legit_sd = float(legit_err.std() + 1e-9)
        return self

    # ------------------------------------------------------------------ inference

    def _qae_errors_scaled(self, Xs: np.ndarray) -> np.ndarray:
        probs = np.atleast_2d(self._qae(Xs, self.phi))
        return 1.0 - probs[:, 0]

    def score_components(self, X: np.ndarray) -> dict:
        """X in raw units, shape (n, k). Returns per-sample components."""
        Xs = self.scaler.transform(np.atleast_2d(X)) * np.pi
        return self._score_scaled(Xs)

    def _score_scaled(self, Xs: np.ndarray) -> dict:
        Xs = np.atleast_2d(Xs)
        z = np.atleast_1d(self._vqc(Xs, self.theta))
        p_vqc = (1.0 - z) / 2.0
        err = self._qae_errors_scaled(Xs)
        anomaly = 1.0 / (1.0 + np.exp(-(err - self.legit_mu) / self.legit_sd))
        q_score = 0.5 * p_vqc + 0.5 * anomaly
        return {
            "q_score": q_score,
            "p_vqc": p_vqc,
            "qae_error": err,
            "anomaly": anomaly,
        }

    def q_scores(self, X: np.ndarray) -> np.ndarray:
        return self.score_components(X)["q_score"]

    # ------------------------------------------------------------------ persistence

    def save(self, directory: Path):
        directory = Path(directory)
        np.savez(directory / "quantum_weights.npz", theta=self.theta, phi=self.phi)
        joblib.dump(self.scaler, directory / "quantum_scaler.joblib")
        meta = {
            "feature_names": self.feature_names,
            "n_layers": self.n_layers,
            "n_trash": self.n_trash,
            "alpha": self.alpha,
            "seed": self.seed,
            "legit_mu": self.legit_mu,
            "legit_sd": self.legit_sd,
            "loss_history": self.loss_history,
        }
        (directory / "quantum_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, backend: str = "local") -> "QuantumAdjudicator":
        directory = Path(directory)
        meta = json.loads((directory / "quantum_meta.json").read_text())
        adj = cls(
            feature_names=meta["feature_names"],
            n_layers=meta["n_layers"],
            n_trash=meta["n_trash"],
            alpha=meta["alpha"],
            backend=backend,
            seed=meta["seed"],
        )
        weights = np.load(directory / "quantum_weights.npz")
        adj.theta = weights["theta"]
        adj.phi = weights["phi"]
        adj.scaler = joblib.load(directory / "quantum_scaler.joblib")
        adj.legit_mu = meta["legit_mu"]
        adj.legit_sd = meta["legit_sd"]
        adj.loss_history = meta.get("loss_history", [])
        return adj

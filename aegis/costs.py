"""Cost model for running the Aegis mesh.

Separates the two spends the architecture actually incurs:

  - the synchronous sieve, which runs on ordinary CPU (or GPU) compute and is
    billed by instance-hour;
  - the asynchronous adjudicator, which is billed either per quantum task and
    shot on a QPU, per minute on a managed simulator, or by instance-hour if
    the circuits are simulated on hardware you already rent.

All prices are list prices, no committed-use or credit discounts, and they
move; PRICING_DATE records when they were last checked against the source.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_DATE = "2026-09-14"

# --- Amazon Braket -----------------------------------------------------------
# verified against https://aws.amazon.com/braket/pricing/ on PRICING_DATE.
# Note: IonQ Aria (targeted by the original proposal) is no longer offered on
# Braket; IonQ's current devices are Forte-1 and Forte-Enterprise-1.

BRAKET_TASK_FEE = 0.30  # per quantum task, every QPU

QPU_SHOT_PRICE = {
    "ionq-forte": 0.08,
    "aqt-ibex-q1": 0.0235,
    "quera-aquila": 0.01,
    "iqm-emerald": 0.0016,
    "iqm-garnet": 0.00145,
    "rigetti-cepheus": 0.000425,
}

QPU_RESERVATION_HOURLY = {
    "ionq-forte": 7000.0,
    "aqt-ibex-q1": 4800.0,
    "rigetti-cepheus": 4100.0,
    "iqm-emerald": 4000.0,
    "iqm-garnet": 3000.0,
    "quera-aquila": 2500.0,
}

SIMULATOR_PER_MINUTE = {"SV1": 0.075, "DM1": 0.075}
SIMULATOR_MIN_BILLED_SECONDS = 3.0

# --- EC2 on-demand, us-east-1 ------------------------------------------------
# approximate list prices; verify in the AWS pricing calculator before quoting.
EC2_HOURLY = {
    "c6i.xlarge": 0.17,       # 4 vCPU, no GPU — what the sieve actually needs
    "g4dn.xlarge": 0.526,     # 1x T4
    "g5.xlarge": 1.006,       # 1x A10G
    "p4d.24xlarge": 21.958,   # 8x A100 40GB
    "p5.48xlarge": 55.04,     # 8x H100
}


@dataclass
class MeshWorkload:
    """Volume and shape of the work, independent of who is billing for it."""

    transactions: int              # transactions authorized over the period
    routed_fraction: float         # share sent to the adjudicator
    n_qubits: int = 4
    shots: int = 1000
    with_shap: bool = True
    sieve_ms: float = 2.2          # measured p50 per transaction
    circuit_ms: float = 45.0       # measured wall-clock per simulated circuit

    @property
    def adjudications(self) -> float:
        return self.transactions * self.routed_fraction

    @property
    def circuits_per_adjudication(self) -> int:
        """Circuit executions needed to adjudicate one transaction.

        The verdict itself costs two circuits (VQC + QAE). Exact Quantum SHAP
        evaluates every one of the 2^k feature coalitions, and each coalition
        is scored by both circuits. Parameter broadcasting collapses these into
        a single call on a simulator, but a QPU bills every one as its own
        task, so the attribution — not the classification — dominates.
        """
        verdict = 2
        shap = (2 ** self.n_qubits) * 2 if self.with_shap else 0
        return verdict + shap

    @property
    def total_circuits(self) -> float:
        return self.adjudications * self.circuits_per_adjudication


def qpu_cost(workload: MeshWorkload, device: str = "ionq-forte") -> dict:
    """Per-task + per-shot cost of adjudicating on a real QPU."""
    if device not in QPU_SHOT_PRICE:
        raise ValueError(f"unknown device '{device}'; known: {sorted(QPU_SHOT_PRICE)}")
    per_circuit = BRAKET_TASK_FEE + workload.shots * QPU_SHOT_PRICE[device]
    total = workload.total_circuits * per_circuit
    return {
        "device": device,
        "per_circuit": per_circuit,
        "per_adjudication": per_circuit * workload.circuits_per_adjudication,
        "per_transaction": total / workload.transactions if workload.transactions else 0.0,
        "total": total,
    }


def simulator_cost(workload: MeshWorkload, simulator: str = "DM1") -> dict:
    """Managed Braket simulator, billed per minute with a 3-second floor.

    The floor is what hurts: a 45 ms circuit is billed as 3 seconds, so the
    effective rate is ~67x the wall-clock rate unless circuits are batched
    into longer tasks.
    """
    rate_per_second = SIMULATOR_PER_MINUTE[simulator] / 60.0
    billed_seconds = max(workload.circuit_ms / 1000.0, SIMULATOR_MIN_BILLED_SECONDS)
    per_circuit = billed_seconds * rate_per_second
    total = workload.total_circuits * per_circuit
    return {
        "device": simulator,
        "per_circuit": per_circuit,
        "billed_seconds_per_circuit": billed_seconds,
        "per_adjudication": per_circuit * workload.circuits_per_adjudication,
        "per_transaction": total / workload.transactions if workload.transactions else 0.0,
        "total": total,
    }


def instance_cost(workload: MeshWorkload, instance: str = "c6i.xlarge",
                  parallel_circuits: int = 1) -> dict:
    """Simulating the circuits on compute you rent by the hour.

    At 4 qubits the statevector is 16 amplitudes, so this is a rounding error
    on any instance; the GPU entries exist to show the crossover is nowhere
    near this circuit size.
    """
    hourly = EC2_HOURLY[instance]
    circuit_hours = workload.total_circuits * (workload.circuit_ms / 1000.0) / 3600.0
    circuit_hours /= max(parallel_circuits, 1)
    sieve_hours = workload.transactions * (workload.sieve_ms / 1000.0) / 3600.0
    total = (circuit_hours + sieve_hours) * hourly
    return {
        "device": instance,
        "hourly": hourly,
        "adjudicator_hours": circuit_hours,
        "sieve_hours": sieve_hours,
        "per_transaction": total / workload.transactions if workload.transactions else 0.0,
        "total": total,
    }


def sieve_only_cost(workload: MeshWorkload, instance: str = "c6i.xlarge") -> dict:
    """The classical path on its own — the baseline every option is added to."""
    hours = workload.transactions * (workload.sieve_ms / 1000.0) / 3600.0
    total = hours * EC2_HOURLY[instance]
    return {
        "device": instance,
        "compute_hours": hours,
        "per_transaction": total / workload.transactions if workload.transactions else 0.0,
        "total": total,
    }


def fraud_value_at_stake(workload: MeshWorkload, fraud_rate: float,
                         band_fraud_share: float, avg_fraud_amount: float) -> dict:
    """Fraud value flowing through the band, as a ceiling on what adjudication
    can be worth. Catching *all* of it is the optimistic bound, not a forecast.
    """
    total_fraud = workload.transactions * fraud_rate
    band_fraud = total_fraud * band_fraud_share
    return {
        "fraud_transactions": total_fraud,
        "fraud_in_band": band_fraud,
        "value_in_band": band_fraud * avg_fraud_amount,
        "value_per_transaction": band_fraud * avg_fraud_amount / workload.transactions
        if workload.transactions else 0.0,
    }

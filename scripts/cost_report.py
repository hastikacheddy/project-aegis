"""Estimate what it costs to run the Aegis mesh, per backend and at volume.

Reads the measured routing fraction and latencies out of artifacts/metrics.json
so the estimate reflects the trained system rather than assumed parameters.

    python scripts/cost_report.py
    python scripts/cost_report.py --transactions 1000000000   # annual issuer volume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from aegis.config import ARTIFACTS, DATA_DIR, PLOTS
from aegis.costs import (EC2_HOURLY, PRICING_DATE, QPU_RESERVATION_HOURLY,
                         QPU_SHOT_PRICE, MeshWorkload, fraud_value_at_stake,
                         instance_cost, qpu_cost, sieve_only_cost,
                         simulator_cost)


def money(x: float) -> str:
    if x == 0:
        return "$0"
    if abs(x) >= 1_000_000_000:
        return f"${x / 1_000_000_000:,.1f}B"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:,.1f}M"
    if abs(x) >= 1000:
        return f"${x:,.0f}"
    if abs(x) >= 0.01:
        return f"${x:,.2f}"
    return f"${x:.2e}"


def avg_fraud_amount(default: float = 122.21) -> tuple[float, str]:
    """Mean value of a fraudulent transaction, from the dataset if cached."""
    cache = DATA_DIR / "creditcard.csv"
    if cache.exists():
        df = pd.read_csv(cache, usecols=["Amount", "Class"])
        return float(df.loc[df["Class"] == 1, "Amount"].mean()), "measured"
    return default, "assumed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", type=int, default=1_000_000,
                    help="authorization volume to price (default 1M)")
    ap.add_argument("--shots", type=int, default=1000)
    args = ap.parse_args()

    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    routed = metrics["mesh"]["routed_pct"]
    sieve_ms = metrics["sieve"]["latency"]["p50_ms"]
    n_qubits = metrics["quantum"]["n_qubits"]
    fraud_rate = metrics["dataset"]["fraud_rate"]
    band = metrics["ambiguity_band"]
    band_fraud_share = band["n_fraud"] / max(metrics["dataset"]["fraud"] * 0.2, 1)

    wl = MeshWorkload(transactions=args.transactions, routed_fraction=routed,
                      n_qubits=n_qubits, shots=args.shots, sieve_ms=sieve_ms)
    wl_noshap = MeshWorkload(transactions=args.transactions, routed_fraction=routed,
                             n_qubits=n_qubits, shots=args.shots, sieve_ms=sieve_ms,
                             with_shap=False)
    wl_all = MeshWorkload(transactions=args.transactions, routed_fraction=1.0,
                          n_qubits=n_qubits, shots=args.shots, sieve_ms=sieve_ms)

    print("=" * 78)
    print(f"AEGIS COST MODEL — {args.transactions:,} authorizations")
    print(f"list prices as of {PRICING_DATE}, no discounts or credits applied")
    print("=" * 78)
    print(f"  routed to adjudicator : {routed:.3%}  "
          f"({wl.adjudications:,.0f} adjudications)")
    print(f"  circuits per adjudication: {wl.circuits_per_adjudication}  "
          f"(2 verdict + {2 ** n_qubits * 2} for exact SHAP over "
          f"{2 ** n_qubits} coalitions)")
    print(f"  total circuit executions : {wl.total_circuits:,.0f}")
    print(f"  shots per circuit        : {args.shots}")

    sieve = sieve_only_cost(wl)
    print(f"\n--- classical sieve baseline ---")
    print(f"  {sieve['device']:<16} {sieve['compute_hours']:>10,.1f} h   "
          f"{money(sieve['total']):>12}   {money(sieve['per_transaction'])}/txn")

    rows = []
    print(f"\n--- adjudicator on a QPU (per-task + per-shot) ---")
    print(f"  {'device':<18}{'per circuit':>14}{'per adjudication':>20}{'total':>16}")
    for device in sorted(QPU_SHOT_PRICE, key=lambda d: QPU_SHOT_PRICE[d], reverse=True):
        c = qpu_cost(wl, device)
        rows.append(("qpu", device, c))
        print(f"  {device:<18}{money(c['per_circuit']):>14}"
              f"{money(c['per_adjudication']):>20}{money(c['total']):>16}")

    print(f"\n--- adjudicator on managed simulation / rented compute ---")
    print(f"  {'backend':<18}{'per adjudication':>20}{'total':>16}")
    sim = simulator_cost(wl, "DM1")
    rows.append(("simulator", "DM1", sim))
    print(f"  {'Braket DM1':<18}{money(sim['per_adjudication']):>20}"
          f"{money(sim['total']):>16}"
          f"   (3s minimum billing per task)")
    for inst in ["c6i.xlarge", "g4dn.xlarge", "g5.xlarge", "p4d.24xlarge"]:
        c = instance_cost(wl, inst)
        adj_only = c["total"] - sieve["total"]
        rows.append(("instance", inst, c))
        print(f"  {inst:<18}{money(adj_only / wl.adjudications):>20}"
              f"{money(adj_only):>16}")

    # what the routing decision is worth
    print(f"\n--- what selective routing buys (IonQ Forte, exact SHAP) ---")
    routed_cost = qpu_cost(wl, "ionq-forte")["total"]
    all_cost = qpu_cost(wl_all, "ionq-forte")["total"]
    print(f"  adjudicate {routed:.3%} of traffic : {money(routed_cost)}")
    print(f"  adjudicate 100% of traffic   : {money(all_cost)}")
    print(f"  ratio                        : {all_cost / routed_cost:,.0f}x")

    print(f"\n--- what exact SHAP costs (IonQ Forte) ---")
    shap_cost = qpu_cost(wl, "ionq-forte")["total"]
    noshap_cost = qpu_cost(wl_noshap, "ionq-forte")["total"]
    print(f"  verdict only ({wl_noshap.circuits_per_adjudication} circuits) : {money(noshap_cost)}")
    print(f"  with exact SHAP ({wl.circuits_per_adjudication} circuits): {money(shap_cost)}")
    print(f"  attribution is {(shap_cost - noshap_cost) / shap_cost:.1%} of the quantum bill")

    # economic ceiling
    amt, amt_source = avg_fraud_amount()
    value = fraud_value_at_stake(wl, fraud_rate, band_fraud_share, amt)
    print(f"\n--- fraud value flowing through the band ---")
    print(f"  mean fraud amount        : ${amt:,.2f} ({amt_source})")
    print(f"  fraud in band            : {value['fraud_in_band']:,.0f} transactions")
    print(f"  value at stake           : {money(value['value_in_band'])}")
    print(f"  ceiling per transaction  : {money(value['value_per_transaction'])}")
    print(f"  (this is the *entire* fraud value in the band; adjudication can only")
    print(f"   ever recover the part the sieve would have missed)")

    ceiling = max(value["value_per_transaction"], 1e-12)
    forte = qpu_cost(wl, "ionq-forte")
    gpu = instance_cost(wl, "g5.xlarge")
    gpu_adj = (gpu["total"] - sieve["total"]) / wl.transactions
    print(f"\n  QPU (IonQ Forte) per txn : {money(forte['per_transaction'])}"
          f"  = {forte['per_transaction'] / ceiling:,.0f}x the ceiling")
    print(f"  GPU simulation per txn   : {money(gpu_adj)}"
          f"  = 1/{ceiling / gpu_adj:,.0f} of the ceiling")

    report = {
        "pricing_date": PRICING_DATE,
        "transactions": args.transactions,
        "shots": args.shots,
        "routed_fraction": routed,
        "adjudications": wl.adjudications,
        "circuits_per_adjudication": wl.circuits_per_adjudication,
        "total_circuits": wl.total_circuits,
        "sieve": sieve,
        "qpu": {d: qpu_cost(wl, d) for d in QPU_SHOT_PRICE},
        "simulator_dm1": sim,
        "instances": {i: instance_cost(wl, i) for i in EC2_HOURLY},
        "routing_saving_ratio": all_cost / routed_cost,
        "shap_share_of_quantum_bill": (shap_cost - noshap_cost) / shap_cost,
        "fraud_value": value,
        "reservation_hourly": QPU_RESERVATION_HOURLY,
    }
    out = ARTIFACTS / "cost_model.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[cost] written to {out}")

    _plot(wl, sieve, value, args.transactions)


def _plot(wl, sieve, value, transactions) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.metrics import CYAN, DARK, GRID, PURPLE, RED, TEXT, _style

    options = [
        ("CPU sim\n(c6i)", instance_cost(wl, "c6i.xlarge")["total"] - sieve["total"], CYAN),
        ("GPU sim\n(g5)", instance_cost(wl, "g5.xlarge")["total"] - sieve["total"], CYAN),
        ("GPU sim\n(p4d)", instance_cost(wl, "p4d.24xlarge")["total"] - sieve["total"], CYAN),
        ("Braket\nDM1", simulator_cost(wl, "DM1")["total"], PURPLE),
        ("Rigetti\nCepheus", qpu_cost(wl, "rigetti-cepheus")["total"], RED),
        ("IQM\nGarnet", qpu_cost(wl, "iqm-garnet")["total"], RED),
        ("IonQ\nForte", qpu_cost(wl, "ionq-forte")["total"], RED),
    ]
    labels = [o[0] for o in options]
    values = [max(o[1], 1e-9) for o in options]
    colors = [o[2] for o in options]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=DARK)
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(value["value_in_band"], color="#fbbf24", linestyle="--", linewidth=1.4,
               label=f"fraud value in band ({money(value['value_in_band'])})")
    ax.set_yscale("log")
    _style(ax, f"Adjudicator cost per {transactions:,} authorizations (log scale)")
    ax.set_ylabel("cost (USD)")
    for i, v in enumerate(values):
        ax.text(i, v * 1.5, money(v), ha="center", color=TEXT, fontsize=8)
    leg = ax.legend(facecolor=DARK, edgecolor=GRID, fontsize=8)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    path = PLOTS / "cost_comparison.png"
    fig.savefig(path, dpi=150, facecolor=DARK)
    plt.close(fig)
    print(f"[cost] plot written to {path}")


if __name__ == "__main__":
    main()

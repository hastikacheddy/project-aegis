"""Derive the serverless build of the dashboard from the canonical one.

app/static/index.html is the single source of truth for markup, styling and
rendering. This script swaps only its data layer: instead of polling the
FastAPI backend, the page replays the records precomputed by
export_static_demo.py. Everything the viewer sees is still real model output.

    python scripts/export_static_demo.py --out build/static_demo
    python scripts/build_static_site.py --out build/static_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "app" / "static" / "index.html"

LIVE_CHIP = '<div class="chip live">LIVE FEED</div>'
REPLAY_CHIP = '<div class="chip live">RECORDED RUN</div>'

LIVE_FOOTNOTE_START = "  Demo feed is enriched to ~8% fraud for visibility"
REPLAY_FOOTNOTE = """  This is a recorded run, replayed in the browser. Every number shown &mdash; sieve probabilities, VQC and QAE
  scores, verdicts and Shapley attributions &mdash; was produced by the trained models on held-out transactions and
  written out ahead of time by <span class="mono">scripts/export_static_demo.py</span>; the page replays those records
  rather than scoring in real time, which is what lets it run on static hosting. The feed is also enriched to ~8% fraud
  and ~22% ambiguous cases for visibility, against a real prevalence of 0.172% fraud and 0.6% ambiguous, so the
  "cleared synchronously" figure here sits well below the 99.4% measured on the real distribution. To watch the models
  score live, run the FastAPI server from the repository."""

# everything from this marker to the end of the script block is the data layer
DATA_LAYER_START = "async function tick() {"
DATA_LAYER_END = "setInterval(tick, 1000);"

REPLAY_DRIVER = """// ---------------------------------------------------------------- replay driver
// Records precomputed by scripts/export_static_demo.py from the trained models.
// The rendering above is shared verbatim with the live server build.
let DATA = null;
let streamPos = 0;
const injectPos = { legit: 0, band: 0, fraud: 0 };
const MAX_TXNS = 48, MAX_ADJ = 6, ADJ_DELAY_MS = 1300, MAX_LAT = 200;

const S = {
  stats: { total: 0, approved: 0, declined: 0, queued: 0, adjudicated: 0,
           frauds_blocked: 0, false_declines: 0 },
  latencies: [], queue_depth: 0, txns: [], adjudications: [],
};

const nowTs = () => new Date().toTimeString().slice(0, 8);

function percentile(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}

function snapshot() {
  const st = S.stats;
  return {
    stats: Object.assign({}, st, {
      sync_cleared_pct: st.total ? 1 - st.queued / st.total : 1,
      latency_p50: percentile(S.latencies, 50),
      latency_p99: percentile(S.latencies, 99),
    }),
    queue_depth: S.queue_depth,
    txns: S.txns,
    adjudications: S.adjudications,
    band: DATA.band,
    blend_w: DATA.blend_w,
    quantum_features: DATA.quantum_features,
    metrics: DATA.metrics,
  };
}

function settle(txn, truth, declined) {
  const st = S.stats;
  if (declined) {
    st.declined++;
    if (truth === 1) st.frauds_blocked++; else st.false_declines++;
  } else {
    st.approved++;
  }
}

function resolveAdjudication(txn, rec) {
  const a = rec.adjudication, st = S.stats;
  st.adjudicated++;
  S.queue_depth = Math.max(0, S.queue_depth - 1);
  txn.final_score = a.final_score;
  txn.verdict = a.verdict;
  txn.status = a.verdict === "DECLINE" ? "Q_DECLINE" : "Q_APPROVE";
  settle(txn, rec.truth, a.verdict === "DECLINE");

  S.adjudications.unshift({
    id: rec.id, ts: nowTs(), amount: rec.amount, truth: rec.truth,
    p_sieve: rec.p_sieve, p_vqc: a.p_vqc, anomaly: a.anomaly,
    qae_error: a.qae_error, q_score: a.q_score, final_score: a.final_score,
    verdict: a.verdict, adjudication_ms: a.adjudication_ms, shap: a.shap,
  });
  if (S.adjudications.length > MAX_ADJ) S.adjudications.pop();
  render(snapshot());
}

function processRecord(rec) {
  const st = S.stats;
  st.total++;
  S.latencies.push(rec.latency_ms);
  if (S.latencies.length > MAX_LAT) S.latencies.shift();

  const txn = {
    id: rec.id, ts: nowTs(), amount: rec.amount, p_sieve: rec.p_sieve,
    latency_ms: rec.latency_ms, truth: rec.truth,
    status: rec.sieve_decision, final_score: null, verdict: null,
  };

  if (rec.ambiguous) {
    txn.status = "Q_PENDING";
    st.queued++;
    S.queue_depth++;
    setTimeout(() => resolveAdjudication(txn, rec), ADJ_DELAY_MS);
  } else {
    settle(txn, rec.truth, rec.sieve_decision === "DECLINE");
  }

  S.txns.unshift(txn);
  if (S.txns.length > MAX_TXNS) S.txns.pop();
  render(snapshot());
}

function inject(kind) {
  if (!DATA) return;
  const pool = DATA.inject[kind];
  processRecord(pool[injectPos[kind]++ % pool.length]);
}

function step() {
  processRecord(DATA.stream[streamPos++ % DATA.stream.length]);
  setTimeout(step, 500 + Math.random() * 900);
}

fetch("demo_data.json")
  .then(r => r.json())
  .then(d => { DATA = d; render(snapshot()); setTimeout(step, 600); })
  .catch(e => { $$("feed").innerHTML =
      '<tr><td colspan="7" class="empty">could not load demo_data.json</td></tr>'; });"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/static_demo")
    args = ap.parse_args()

    html = SOURCE.read_text(encoding="utf-8")

    if LIVE_CHIP not in html:
        sys.exit("[static] could not find the live-feed chip; source markup changed")
    html = html.replace(LIVE_CHIP, REPLAY_CHIP)

    # replace the live footnote paragraph
    start = html.index(LIVE_FOOTNOTE_START)
    end = html.index("</div>", start)
    html = html[:start] + REPLAY_FOOTNOTE + "\n" + html[end:]

    # swap the data layer, keeping every render function above it
    start = html.index(DATA_LAYER_START)
    end = html.index(DATA_LAYER_END, start) + len(DATA_LAYER_END)
    html = html[:start] + REPLAY_DRIVER + html[end:]

    for leftover in ("/api/state", "/api/inject"):
        if leftover in html:
            sys.exit(f"[static] '{leftover}' still referenced; the build would need a backend")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"[static] written {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

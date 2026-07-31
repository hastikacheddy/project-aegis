"""Capture dashboard screenshots for the README.

Drives the running demo (uvicorn app.server:app --port 8021), injects a mix of
transactions so the mesh has something interesting to show, then writes PNGs
into docs/.

    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BASE = "http://127.0.0.1:8021"


def inject(kind: str) -> None:
    req = urllib.request.Request(f"{BASE}/api/inject/{kind}", method="POST")
    with urllib.request.urlopen(req, timeout=10):
        pass


def main() -> None:
    from playwright.sync_api import sync_playwright

    DOCS.mkdir(exist_ok=True)

    try:
        urllib.request.urlopen(f"{BASE}/api/state", timeout=5)
    except Exception:
        sys.exit(f"[shots] no server at {BASE} — start it with:\n"
                 "  .venv\\Scripts\\python -m uvicorn app.server:app --port 8021")

    # seed some state: ambiguous txns produce quantum adjudications with SHAP,
    # frauds produce blocked-fraud counts
    print("[shots] seeding demo state ...")
    for kind in ["band", "fraud", "band", "fraud", "band", "legit", "band", "fraud"]:
        inject(kind)
        time.sleep(0.4)
    time.sleep(4)  # let the async quantum worker drain the queue

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000},
                                device_scale_factor=2)
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector(".adjcard", timeout=30_000)
        time.sleep(2)  # let the 1s poll paint a full frame

        # the dashboard repaints every second by replacing innerHTML, which
        # detaches elements mid-capture — freeze the poll on this rendered frame
        page.evaluate("for (let i = 1; i < 100000; i++) window.clearInterval(i);")
        time.sleep(0.5)

        shots = [
            ("dashboard-overview.png", None,
             "full dashboard: KPIs, mesh pipeline, live authorization stream"),
            ("quantum-adjudication.png", ".adjcard",
             "quantum adjudication card with exact SHAP attribution"),
        ]
        for name, selector, desc in shots:
            target = page.locator(selector).first if selector else page
            target.screenshot(path=str(DOCS / name))
            print(f"[shots] {name:<28} {desc}")

        # metrics panel is the last panel in the right-hand column
        metrics_panel = page.locator(".panel", has=page.locator("#metrics-grid")).first
        metrics_panel.scroll_into_view_if_needed()
        time.sleep(0.5)
        metrics_panel.screenshot(path=str(DOCS / "evaluation-metrics.png"))
        print("[shots] evaluation-metrics.png     held-out test set evaluation")

        browser.close()

    print(f"\n[shots] written to {DOCS}")


if __name__ == "__main__":
    main()

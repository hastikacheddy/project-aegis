"""Retry the OpenML download a few times (flaky networks happen at venues)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.data import CACHE, _fetch_openml

if __name__ == "__main__":
    for attempt in range(1, 5):
        try:
            df = _fetch_openml()
            df.to_csv(CACHE, index=False)
            print(f"[data] attempt {attempt}: success — {len(df):,} rows cached to {CACHE}")
            sys.exit(0)
        except Exception as exc:
            print(f"[data] attempt {attempt} failed: {exc!r}")
            time.sleep(10)
    print("[data] all attempts failed")
    sys.exit(1)

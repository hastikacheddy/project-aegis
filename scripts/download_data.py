"""Pre-fetch and cache the European Cardholder dataset (OpenML -> data/creditcard.csv)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.data import load_dataset

if __name__ == "__main__":
    df, source = load_dataset(prefer_real=True)
    print(f"[data] ready: source={source}, rows={len(df):,}")

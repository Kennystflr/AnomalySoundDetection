#!/usr/bin/env python3
"""
sort_chunks.py

Computes the delta between two CSV files: rows whose filename appears in
CSV B are removed from CSV A, and the remainder is written to a new file.

Usage:
    python sort_chunks.py --a path/to/a.csv --b path/to/b.csv --output path/to/delta.csv

The join key defaults to 'filename' but can be changed with --key.
All columns from CSV A are preserved in the output.
"""

import argparse
import pandas as pd
from pathlib import Path


def csv_delta(a_path: Path, b_path: Path, output_path: Path, key: str) -> None:
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    if key not in a.columns:
        raise ValueError(f"Key column '{key}' not found in {a_path}. Available: {list(a.columns)}")
    if key not in b.columns:
        raise ValueError(f"Key column '{key}' not found in {b_path}. Available: {list(b.columns)}")

    keys_b = set(b[key].dropna())
    delta = a[~a[key].isin(keys_b)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(output_path, index=False)

    removed = len(a) - len(delta)
    print(f"CSV A     : {len(a):>6} rows  ({a_path})")
    print(f"CSV B     : {len(b):>6} rows  ({b_path})")
    print(f"Removed   : {removed:>6} rows  (matched on '{key}')")
    print(f"Delta     : {len(delta):>6} rows  → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove rows from CSV A whose key appears in CSV B."
    )
    parser.add_argument("--a",      required=True,        help="Source CSV (rows to filter)")
    parser.add_argument("--b",      required=True,        help="Filter CSV (rows to remove)")
    parser.add_argument("--output", required=True,        help="Output CSV path")
    parser.add_argument("--key",    default="filename",   help="Column to join on (default: filename)")
    args = parser.parse_args()

    csv_delta(
        a_path      = Path(args.a),
        b_path      = Path(args.b),
        output_path = Path(args.output),
        key         = args.key,
    )


if __name__ == "__main__":
    main()

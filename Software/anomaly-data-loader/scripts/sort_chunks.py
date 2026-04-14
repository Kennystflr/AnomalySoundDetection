#!/usr/bin/env python3
"""
sort_chunks.py

Two modes, selected by `mode` in the config file:

  sort     – Move audio/mel/embedding files into three folders based on two
             distance thresholds:
               distance <= low           → normal/
               low < distance <= high    → doubt/
               distance > high           → anomaly/

  metadata – Write a metadata.csv with columns (filename, label) where
             normal chunks get label 0, anomaly chunks get label 1, and
             doubt chunks are excluded.
"""

import argparse
import csv
import yaml
import shutil
from pathlib import Path


# ── Default thresholds (override via CLI args) ────────────────────────────────
DEFAULT_LOW  = 0.3
DEFAULT_HIGH = 0.6

# ── Folder names ──────────────────────────────────────────────────────────────
FOLDER_NEAR = "normal"
FOLDER_MID  = "doubt"
FOLDER_FAR  = "anomaly"


def resolve_dest(distance: float, low: float, high: float) -> str:
    if distance < 0:
        return FOLDER_MID  # Treat negative distances as "doubt" (or could raise an error)
    elif distance <= low:
        return FOLDER_NEAR
    elif distance <= high:
        return FOLDER_MID
    else:
        return FOLDER_FAR


def move_if_exists(src: Path, dest_dir: Path, dry_run: bool) -> str:
    """Move src to dest_dir/src.name. Returns status string."""
    if not src.exists():
        return f"  [SKIP – not found] {src}"
    dest = dest_dir / src.name
    if dry_run:
        return f"  [DRY-RUN] {src}  →  {dest}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return f"  [MOVED] {src}  →  {dest}"


def process(csv_path: Path, low: float, high: float, dry_run: bool, verbose: bool) -> None:

    counters = {FOLDER_NEAR: 0, FOLDER_MID: 0, FOLDER_FAR: 0}
    skipped  = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        required = {"chunk_id", "audio_path", "mel_path",
                    "embedding_path", "label", "distance_to_ref"}
        if not required.issubset(reader.fieldnames or []):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV is missing columns: {missing}")

        for row in reader:
            try:
                distance = float(row["distance_to_ref"])
            except ValueError:
                print(f"[WARN] Non-numeric distance for chunk '{row['chunk_id']}' – skipping row.")
                skipped += 1
                continue

            dest_name = resolve_dest(distance, low, high)
            audio_dir = Path(row["audio_path"]).parent
            dest_dir  = audio_dir / dest_name

            if verbose:
                print(f"\nchunk_id={row['chunk_id']}  distance={distance:.4f}  → {dest_name}")

            for key in ("audio_path", "mel_path", "embedding_path"):
                src = Path(row[key])
                msg = move_if_exists(src, dest_dir, dry_run)
                if verbose:
                    print(msg)

            counters[dest_name] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(counters.values())
    print("\n" + "─" * 50)
    print(f"{'DRY-RUN ' if dry_run else ''}Summary")
    print("─" * 50)
    print(f"  Thresholds : low={low}  high={high}")
    print(f"  {FOLDER_NEAR:20s}: {counters[FOLDER_NEAR]:>5} chunks")
    print(f"  {FOLDER_MID:20s}: {counters[FOLDER_MID]:>5} chunks")
    print(f"  {FOLDER_FAR:20s}: {counters[FOLDER_FAR]:>5} chunks")
    print(f"  {'Skipped (bad rows)':20s}: {skipped:>5}")
    print(f"  {'Total processed':20s}: {total:>5}")
    print("─" * 50)


LABEL_MAP = {FOLDER_NEAR: 0, FOLDER_FAR: 1}  # doubt (FOLDER_MID) is excluded


def generate_metadata(csv_path: Path, low: float, high: float,
                      output_path: Path, verbose: bool) -> None:
    rows = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        required = {"chunk_id", "audio_path", "distance_to_ref"}
        if not required.issubset(reader.fieldnames or []):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV is missing columns: {missing}")

        excluded = 0
        for row in reader:
            try:
                distance = float(row["distance_to_ref"])
            except ValueError:
                print(f"[WARN] Non-numeric distance for chunk '{row['chunk_id']}' – skipping row.")
                excluded += 1
                continue

            dest_name = resolve_dest(distance, low, high)
            if dest_name not in LABEL_MAP:
                if verbose:
                    print(f"  [EXCLUDED – doubt] {row['chunk_id']}  distance={distance:.4f}")
                excluded += 1
                continue

            label = LABEL_MAP[dest_name]
            filename = Path(row["audio_path"]).name
            rows.append({"filename": filename, "label": label})

            if verbose:
                print(f"  {filename}  distance={distance:.4f}  → label={label}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(rows)

    normal_count  = sum(1 for r in rows if r["label"] == 0)
    anomaly_count = sum(1 for r in rows if r["label"] == 1)
    print("\n" + "─" * 50)
    print("Metadata Summary")
    print("─" * 50)
    print(f"  Thresholds : low={low}  high={high}")
    print(f"  {'Normal (label=0)':20s}: {normal_count:>5} chunks")
    print(f"  {'Anomaly (label=1)':20s}: {anomaly_count:>5} chunks")
    print(f"  {'Excluded (doubt)':20s}: {excluded:>5} chunks")
    print(f"  {'Output':20s}: {output_path}")
    print("─" * 50)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config["low"] >= config["high"]:
        raise ValueError(f"low ({config['low']}) must be strictly less than high ({config['high']})")

    if not Path(config["csv"]).exists():
        raise ValueError(f"CSV file not found: {config['csv']}")

    mode = config.get("mode", "sort")

    if mode == "metadata":
        output_path = Path(config.get("metadata_output", "metadata.csv"))
        generate_metadata(
            csv_path    = Path(config["csv"]),
            low         = config["low"],
            high        = config["high"],
            output_path = output_path,
            verbose     = config.get("verbose", False),
        )
    elif mode == "sort":
        process(
            csv_path    = Path(config["csv"]),
            low         = config["low"],
            high        = config["high"],
            dry_run     = config["dry_run"],
            verbose     = config["verbose"],
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'sort' or 'metadata'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
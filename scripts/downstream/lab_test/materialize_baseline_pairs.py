#!/usr/bin/env python
"""
Materialize baseline_pairs wide table for the lab test downstream task.

The script constructs exam-level (t -> t+1) pairs from cleaned_labels.parquet,
mirroring the horizon and split logic used by the downstream trainer. Each row
contains source exam features (X_*), target exam labels (Y_*), split flags, and
per-person test selection markers to support benchmarking baselines.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Construct baseline_pairs wide table from cleaned labels."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to data/downstream/lab_test/v1/cleaned_labels.parquet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config/downstream/lab_test_task_config_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for baseline_pairs.parquet",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/downstream/v1/benchmark"),
        help="Directory for run logs and metadata",
    )
    return parser.parse_args()


def load_label_order(config_path: Path) -> List[str]:
    """Load label_order from downstream task config."""
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    try:
        label_order = config["datamodule"]["label_processing"]["label_order"]
    except KeyError as exc:
        raise KeyError(
            "label_order not found in config. Expected path "
            "datamodule.label_processing.label_order"
        ) from exc

    if not isinstance(label_order, list) or not label_order:
        raise ValueError("label_order must be a non-empty list")

    return label_order


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    """Entry point."""
    args = parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_dir / "materialize_baseline_pairs.log"
    log_handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )
    logging.info("Loading label schema from %s", args.config)
    label_order = load_label_order(args.config)

    if not args.labels.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    logging.info("Reading cleaned labels from %s", args.labels)
    df = pd.read_parquet(args.labels)

    required_meta = ["exam_id", "person_id", "ExamDate", "split"]
    missing_meta = [col for col in required_meta if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing required metadata columns: {missing_meta}")

    missing_labels = [col for col in label_order if col not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")

    logging.info("Sorting rows by person_id, ExamDate, exam_id for deterministic ordering")
    df = df.sort_values(["person_id", "ExamDate", "exam_id"]).reset_index(drop=True)

    logging.info("Constructing X_* and Y_* columns via per-person shifts")
    pairs = pd.DataFrame({"person_id": df["person_id"]})

    # Source columns (X_)
    source_meta = ["exam_id", "ExamDate", "split"]
    for col in source_meta:
        pairs[f"X_{col}"] = df[col]
    for col in label_order:
        pairs[f"X_{col}"] = df[col]

    # Target columns (Y_) using shift(-1) within person
    grouped = df.groupby("person_id", sort=False)
    target_meta = ["exam_id", "ExamDate", "split"]
    for col in target_meta:
        pairs[f"Y_{col}"] = grouped[col].shift(-1)
    for col in label_order:
        pairs[f"Y_{col}"] = grouped[col].shift(-1)

    # Drop rows without a target exam
    before_drop = len(pairs)
    pairs = pairs[~pairs["Y_exam_id"].isna()].copy()
    logging.info("Dropped %d rows without next exam", before_drop - len(pairs))

    pairs["pair_index_in_person"] = pairs.groupby("person_id").cumcount()

    # Split flags based on Y_split (target exam)
    pairs["is_train_pair"] = pairs["Y_split"] == "TRAIN"
    pairs["is_val_pair"] = pairs["Y_split"] == "VAL"
    pairs["is_testf_pair"] = pairs["Y_split"] == "TESTF"

    # Mark final TESTF pair per person for test evaluation
    pairs["use_for_test"] = False
    testf_last_indices = (
        pairs[pairs["is_testf_pair"]].groupby("person_id", sort=False).tail(1).index
    )
    pairs.loc[testf_last_indices, "use_for_test"] = True

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing baseline pairs to %s", output_path)
    pairs.to_parquet(output_path, index=False)
    logging.info("Done. Rows: %d, Columns: %d", len(pairs), pairs.shape[1])

    logging.info("Computing SHA256 hashes for provenance")
    labels_hash = compute_sha256(args.labels)
    output_hash = compute_sha256(output_path)

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "labels_path": str(args.labels),
        "labels_sha256": labels_hash,
        "output_path": str(output_path),
        "output_sha256": output_hash,
        "rows": int(len(pairs)),
        "columns": int(pairs.shape[1]),
    }
    metadata_path = args.log_dir / "baseline_pairs_runs.jsonl"
    with metadata_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata) + "\n")
    logging.info("Recorded run metadata at %s", metadata_path)


if __name__ == "__main__":
    main()

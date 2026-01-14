#!/usr/bin/env python3
# scripts/data_preparation/v1/make_chronological_splits_v1.py

"""Generate chronological train/val/test manifests with optional sorting and audit logs."""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

timestamp_format = "%Y%m%d_%H%M%S"

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chronological data splits with optional manifest sorting."
    )
    parser.add_argument(
        "--input",
        default="data/processed/v1/mcinfo/exam_level",
        help="Input directory containing year-partitioned exam-level parquet files.",
    )
    parser.add_argument(
        "--output",
        default="data/splits/v1",
        help="Output directory for split manifests (core[/sorted]/...).",
    )
    parser.add_argument(
        "--metadata-output",
        default="config/splitting/v1",
        help="Directory to write YAML metadata about the splits.",
    )
    parser.add_argument(
        "--log-path",
        default="outputs/audit/v1/make_chronological_splits_logs",
        help="Directory for execute log and audit artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed included in metadata for reproducibility tracking.",
    )
    parser.add_argument(
        "--p-train",
        type=float,
        default=0.70,
        help="Percentile (0-1) defining the train/val cut date (default 0.70).",
    )
    parser.add_argument(
        "--p-val",
        type=float,
        default=0.85,
        help="Percentile (0-1) defining the val/test cut date (default 0.85).",
    )
    parser.add_argument(
        "--sort",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Sort manifests by person_id then ExamDate before writing (default: True).",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime(timestamp_format)
    log_path = log_dir / f"make_chronological_splits_{ts}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return log_path


def file_sha256(path: Path, enable: bool = True) -> Optional[str]:
    if not enable:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_source_hash(input_dir: Path) -> str:
    hasher = hashlib.sha256()
    if not input_dir.exists():
        logging.warning("Input directory %s does not exist for hashing.", input_dir)
        return hasher.hexdigest()

    for partition in sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("year=")):
        for parquet_file in sorted(partition.glob("*.parquet")):
            try:
                stat = parquet_file.stat()
            except FileNotFoundError:
                continue
            payload = f"{parquet_file}:{stat.st_size}:{int(stat.st_mtime)}"
            hasher.update(payload.encode())
    return hasher.hexdigest()


def load_exam_data(input_dir: Path) -> pd.DataFrame:
    logging.info("Loading exam data from %s", input_dir)
    columns = ["exam_id", "person_id", "ExamDate"]

    try:
        dataset = ds.dataset(str(input_dir), format="parquet", partitioning="hive")
        table = dataset.to_table(columns=columns)
        df = table.to_pandas()
        logging.info("Loaded %d rows via pyarrow.dataset", len(df))
        return df
    except Exception as exc:
        logging.warning("pyarrow.dataset load failed (%s); falling back to manual read.", exc)

    parts: List[pd.DataFrame] = []
    for partition in sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("year=")):
        for parquet_file in sorted(partition.glob("*.parquet")):
            part_df = pd.read_parquet(parquet_file, columns=columns)
            parts.append(part_df)
            logging.info("  Loaded %d rows from %s", len(part_df), parquet_file)

    if not parts:
        logging.warning("No exam data found under %s", input_dir)
        return pd.DataFrame(columns=columns)

    df = pd.concat(parts, ignore_index=True)
    logging.info("Concatenated %d rows from %d fragments", len(df), len(parts))
    return df


def compute_cut_dates(df: pd.DataFrame, p_train: float, p_val: float) -> Tuple[Any, Any]:
    if df.empty:
        raise ValueError("Cannot compute cut dates on empty DataFrame.")

    df_sorted = df.sort_values("ExamDate")
    train_val_cut = df_sorted["ExamDate"].quantile(p_train, interpolation="nearest")
    val_test_cut = df_sorted["ExamDate"].quantile(p_val, interpolation="nearest")

    logging.info("Train/val cut date (p=%.2f): %s", p_train, train_val_cut)
    logging.info("Val/test cut date (p=%.2f): %s", p_val, val_test_cut)

    return train_val_cut, val_test_cut


def create_splits(
    df: pd.DataFrame,
    train_val_cut,
    val_test_cut,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Creating splits using computed cut dates")
    train_df = df[df["ExamDate"] < train_val_cut]
    val_df = df[(df["ExamDate"] >= train_val_cut) & (df["ExamDate"] < val_test_cut)]
    test_df = df[df["ExamDate"] >= val_test_cut]

    total = len(df)
    logging.info("Train-SSL: %d exams (%.2f%%)", len(train_df), (len(train_df) / total * 100) if total else 0)
    logging.info("Val-SSL:   %d exams (%.2f%%)", len(val_df), (len(val_df) / total * 100) if total else 0)
    logging.info("Test-Future: %d exams (%.2f%%)", len(test_df), (len(test_df) / total * 100) if total else 0)

    return train_df, val_df, test_df


def verify_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, bool]:
    logging.info("Verifying split integrity")
    train_ids = set(train_df["exam_id"]) if not train_df.empty else set()
    val_ids = set(val_df["exam_id"]) if not val_df.empty else set()
    test_ids = set(test_df["exam_id"]) if not test_df.empty else set()

    overlaps = {
        "train_val_overlap": bool(train_ids & val_ids),
        "train_test_overlap": bool(train_ids & test_ids),
        "val_test_overlap": bool(val_ids & test_ids),
    }

    for key, value in overlaps.items():
        logging.info("  %s: %s", key, value)

    chronology = {
        "train_before_val": True,
        "val_before_test": True,
    }

    if not train_df.empty and not val_df.empty:
        chronology["train_before_val"] = train_df["ExamDate"].max() < val_df["ExamDate"].min()
    if not val_df.empty and not test_df.empty:
        chronology["val_before_test"] = val_df["ExamDate"].max() < test_df["ExamDate"].min()

    for key, value in chronology.items():
        logging.info("  %s: %s", key, value)

    passes = not any(overlaps.values()) and all(chronology.values())
    if passes:
        logging.info("Verification passed")
    else:
        logging.warning("Verification failed")
    chronology["verification_passed"] = passes
    chronology.update(overlaps)
    return chronology


def sort_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(by=["person_id", "ExamDate"])


def write_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    sort: bool,
) -> Dict[str, Path]:
    core_dir = output_dir / "core"
    target_dir = core_dir / "sorted" if sort else core_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train_ssl": train_df,
        "val_ssl": val_df,
        "test_future": test_df,
    }

    output_paths: Dict[str, Path] = {}
    for name, df in manifests.items():
        base_table = pa.Table.from_pandas(df)
        base_schema = base_table.schema

        out_df = sort_manifest(df) if sort else df
        table = pa.Table.from_pandas(out_df, schema=base_schema)
        output_path = target_dir / f"{name}.parquet"
        if sort:
            pq.write_table(table, output_path)
        else:
            pq.write_table(table, output_path, compression="snappy")
        logging.info("Wrote %s manifest to %s", name, output_path)
        output_paths[name] = output_path

    return output_paths


def write_metadata(
    metadata_dir: Path,
    train_val_cut,
    val_test_cut,
    train_count: int,
    val_count: int,
    test_count: int,
    total_count: int,
    source_hash: str,
    seed: int,
    sort: bool,
) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "split_metadata.yaml"

    def _fmt(value: Any) -> Any:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)

    metadata = {
        "cut_dates": {
            "train_val": _fmt(train_val_cut),
            "val_test": _fmt(val_test_cut),
        },
        "statistics": {
            "total_exams": int(total_count),
            "train_ssl_count": int(train_count),
            "val_ssl_count": int(val_count),
            "test_future_count": int(test_count),
            "train_percentage": round(train_count / total_count * 100, 2) if total_count else 0.0,
            "val_percentage": round(val_count / total_count * 100, 2) if total_count else 0.0,
            "test_percentage": round(test_count / total_count * 100, 2) if total_count else 0.0,
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_data_hash": source_hash,
        "random_seed": int(seed),
        "sorted_manifests": bool(sort),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        import yaml

        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logging.info("Wrote metadata to %s", metadata_path)
    return metadata_path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    metadata_dir = Path(args.metadata_output)
    log_dir = Path(args.log_path)
    sort = bool(args.sort)

    log_path = setup_logging(log_dir)
    logging.info("== make_chronological_splits_v1 started ==")
    logging.info("Project root: %s", project_root)
    logging.info("Input directory: %s", input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Metadata directory: %s", metadata_dir)
    logging.info("Log directory: %s", log_dir)
    logging.info("Log file: %s", log_path.name)
    logging.info("Seed: %d", args.seed)
    logging.info("p_train: %.4f", args.p_train)
    logging.info("p_val: %.4f", args.p_val)
    logging.info("Sort manifests: %s", sort)

    np.random.seed(args.seed)

    started_at = datetime.now(timezone.utc)
    t0 = time.time()

    source_hash = compute_source_hash(input_dir)
    logging.info("Source data hash: %s", source_hash)

    df = load_exam_data(input_dir)
    if df.empty:
        logging.warning("No exam data loaded; aborting early.")
        duration_s = time.time() - t0
        finished_at = datetime.now(timezone.utc)
        audit_obj = {
            "task": "make_chronological_splits_v1",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": round(duration_s, 3),
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "metadata_directory": str(metadata_dir),
            "source_data_hash": source_hash,
            "seed": int(args.seed),
            "p_train": float(args.p_train),
            "p_val": float(args.p_val),
            "sorted": sort,
            "splits": {},
            "verification": {},
            "output_files": {},
            "log_file": log_path.name,
        }
        ts = datetime.now().strftime(timestamp_format)
        audit_path = log_dir / f"make_chronological_splits_audit_{ts}.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)
        summary_path = log_dir / "make_chronological_splits_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)
        logging.info("Audit JSON: %s", audit_path.name)
        logging.info("Summary JSON: %s", summary_path.name)
        logging.info("== make_chronological_splits_v1 completed (no data) ==")
        return 0

    train_val_cut, val_test_cut = compute_cut_dates(df, args.p_train, args.p_val)
    train_df, val_df, test_df = create_splits(df, train_val_cut, val_test_cut)

    verification = verify_splits(train_df, val_df, test_df)
    if not verification.get("verification_passed", False):
        logging.error("Split verification failed; aborting without writing outputs.")
        return 1

    output_paths = write_manifests(train_df, val_df, test_df, output_dir, sort=sort)

    metadata_path = write_metadata(
        metadata_dir,
        train_val_cut,
        val_test_cut,
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
        source_hash,
        args.seed,
        sort,
    )

    output_file_hashes = {
        name: file_sha256(path)
        for name, path in output_paths.items()
    }

    duration_s = time.time() - t0
    finished_at = datetime.now(timezone.utc)

    audit_obj = {
        "task": "make_chronological_splits_v1",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round(duration_s, 3),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "metadata_directory": str(metadata_dir),
        "source_data_hash": source_hash,
        "seed": int(args.seed),
        "p_train": float(args.p_train),
        "p_val": float(args.p_val),
        "sorted": sort,
        "splits": {
            "train_ssl": {
                "rows": int(len(train_df)),
                "min_date": train_df["ExamDate"].min() if not train_df.empty else None,
                "max_date": train_df["ExamDate"].max() if not train_df.empty else None,
            },
            "val_ssl": {
                "rows": int(len(val_df)),
                "min_date": val_df["ExamDate"].min() if not val_df.empty else None,
                "max_date": val_df["ExamDate"].max() if not val_df.empty else None,
            },
            "test_future": {
                "rows": int(len(test_df)),
                "min_date": test_df["ExamDate"].min() if not test_df.empty else None,
                "max_date": test_df["ExamDate"].max() if not test_df.empty else None,
            },
        },
        "verification": verification,
        "output_files": {
            name: {
                "path": str(path),
                "sha256": output_file_hashes[name],
            }
            for name, path in output_paths.items()
        },
        "metadata_file": str(metadata_path),
        "log_file": log_path.name,
    }

    ts = datetime.now().strftime(timestamp_format)
    audit_path = log_dir / f"make_chronological_splits_audit_{ts}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)

    summary_path = log_dir / "make_chronological_splits_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)

    logging.info("== Summary ==")
    logging.info("  train rows: %s", f"{len(train_df):,}")
    logging.info("  val rows: %s", f"{len(val_df):,}")
    logging.info("  test rows: %s", f"{len(test_df):,}")
    logging.info("  metadata: %s", metadata_path)
    logging.info("  audit JSON: %s", audit_path.name)
    logging.info("  summary JSON: %s", summary_path.name)
    logging.info("== make_chronological_splits_v1 completed ==")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

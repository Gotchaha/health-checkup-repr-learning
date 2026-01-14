#!/usr/bin/env python3
# scripts/data_preparation/v1/drop_exam_duplicates_v1.py

"""Deduplicate exam-level mcinfo records by (person_id, ExamDate) with audit logs."""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate mcinfo exam-level data by (person_id, ExamDate)."
    )
    parser.add_argument(
        "--input",
        default="data/deidentified/v1/mcinfo/exam_level",
        help="Input directory containing year-partitioned exam-level parquet files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/v1/mcinfo/exam_level",
        help="Output directory for deduplicated year-partitioned parquet files.",
    )
    parser.add_argument(
        "--log-path",
        default="outputs/audit/v1/drop_duplicates_logs",
        help="Directory for execute log and audit artifacts.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Disable SHA256 file hashing for faster execution.",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"drop_exam_duplicates_{ts}.log"

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


def list_partitions(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        logging.warning("Input directory %s does not exist.", input_dir)
        return []
    partitions = [
        p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("year=")
    ]
    partitions.sort(key=lambda p: p.name)
    return partitions


def load_all_exam_data(
    input_dir: Path,
    compute_hashes: bool,
) -> tuple[pd.DataFrame, Dict[str, str], Optional[pa.Schema]]:
    logging.info("Loading exam data from %s", input_dir)

    tables: List[pa.Table] = []
    file_hashes: Dict[str, str] = {}
    schema: Optional[pa.Schema] = None

    for partition in list_partitions(input_dir):
        parquet_files = sorted(partition.glob("*.parquet"))
        if not parquet_files:
            logging.info("Partition %s has no parquet files; skipping.", partition.name)
            continue

        logging.info("Reading partition %s", partition.name)
        for parquet_file in parquet_files:
            if schema is None:
                logging.info("Capturing schema from %s", parquet_file.name)
            table = pq.read_table(parquet_file)
            if schema is None:
                schema = table.schema
            tables.append(table)

            file_hash = file_sha256(parquet_file, enable=compute_hashes)
            if file_hash:
                file_hashes[str(parquet_file)] = file_hash
            logging.info(
                "  loaded %d rows from %s",
                table.num_rows,
                parquet_file.name,
            )

    if not tables:
        logging.warning("No parquet data found under %s", input_dir)
        return pd.DataFrame(), file_hashes, schema

    combined_table = pa.concat_tables(tables)
    logging.info(
        "Total rows loaded: %d across %d files",
        combined_table.num_rows,
        len(tables),
    )
    df = combined_table.to_pandas()
    logging.info("Converted to pandas DataFrame with %d rows", len(df))
    return df, file_hashes, schema


def deduplicate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    logging.info("Checking duplicates among %d rows", len(df))
    dup_mask = df.duplicated(subset=["person_id", "ExamDate"], keep=False)
    duplicates = df[dup_mask]
    unique_pairs = duplicates[["person_id", "ExamDate"]].drop_duplicates()

    if not unique_pairs.empty:
        logging.warning(
            "Found %d unique (person_id, ExamDate) pairs with duplicates",
            len(unique_pairs),
        )
        logging.warning("Total duplicate rows: %d", len(duplicates))
        sample = unique_pairs.head(5)
        for _, row in sample.iterrows():
            person = row["person_id"]
            exam_date = row["ExamDate"]
            inst = duplicates[
                (duplicates["person_id"] == person)
                & (duplicates["ExamDate"] == exam_date)
            ]
            logging.info(
                "Sample duplicate -> person_id=%s ExamDate=%s instances=%d",
                person,
                exam_date,
                len(inst),
            )
    else:
        logging.info("No duplicate (person_id, ExamDate) pairs detected.")

    deduped = df.drop_duplicates(subset=["person_id", "ExamDate"], keep="first")
    removed = len(df) - len(deduped)
    stats = {
        "total_rows": int(len(df)),
        "unique_rows": int(len(deduped)),
        "duplicate_rows_removed": int(removed),
        "unique_duplicate_pairs": int(len(unique_pairs)),
        "duplicate_rate_percent": float((removed / len(df) * 100) if len(df) else 0.0),
    }

    logging.info("Deduplication summary: %s", stats)
    return deduped, stats


def save_partitions(
    df: pd.DataFrame,
    output_dir: Path,
    schema: Optional[pa.Schema],
    compute_hashes: bool,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: Dict[str, str] = {}

    if df.empty:
        logging.warning("Deduplicated DataFrame is empty; nothing to write.")
        return output_hashes

    # Group indices by year derived from ExamDate
    year_to_indices: Dict[int, List[int]] = {}
    for idx, value in df["ExamDate"].items():
        if hasattr(value, "year"):
            year = int(value.year)
        elif isinstance(value, str):
            year = int(value.split("-")[0])
        else:
            year = int(str(value)[:4])
        year_to_indices.setdefault(year, []).append(idx)

    for year, indices in sorted(year_to_indices.items()):
        part_df = df.loc[indices]
        year_dir = output_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        output_file = year_dir / "data.parquet"

        try:
            if "year" in part_df.columns:
                part_df = part_df.drop(columns=["year"])
                logging.info("Removed physical 'year' column for year=%d", year)

            if schema is not None and "year" in schema.names:
                schema_no_year = pa.schema(
                    [field for field in schema if field.name != "year"]
                )
            else:
                schema_no_year = schema

            table = pa.Table.from_pandas(part_df, schema=schema_no_year)
            pq.write_table(table, output_file, compression="snappy")
            logging.info(
                "Wrote %d rows to %s",
                table.num_rows,
                output_file,
            )

            file_hash = file_sha256(output_file, enable=compute_hashes)
            if file_hash:
                output_hashes[str(output_file)] = file_hash
        except Exception:
            logging.exception("Failed to write partition year=%d", year)
            raise

    logging.info(
        "Saved deduplicated data to %d year partitions",
        len(year_to_indices),
    )
    return output_hashes


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
    log_dir = Path(args.log_path)
    compute_hashes = not args.no_hash

    log_path = setup_logging(log_dir)
    logging.info("== drop_exam_duplicates_v1 started ==")
    logging.info("Project root: %s", project_root)
    logging.info("Input directory: %s", input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Log directory: %s", log_dir)
    logging.info("Log file: %s", log_path.name)
    logging.info("Hash computation: %s", "enabled" if compute_hashes else "disabled")

    started_at = datetime.now(timezone.utc)
    t0 = time.time()

    df, input_hashes, schema = load_all_exam_data(input_dir, compute_hashes)
    if df.empty:
        logging.warning("No data loaded; aborting without writing outputs.")
        duration_s = time.time() - t0
        finished_at = datetime.now(timezone.utc)
        audit_obj = {
            "task": "drop_exam_duplicates_v1",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": round(duration_s, 3),
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "input_file_hashes": input_hashes,
            "output_file_hashes": {},
            "deduplication": {
                "total_rows": 0,
                "unique_rows": 0,
                "duplicate_rows_removed": 0,
                "unique_duplicate_pairs": 0,
                "duplicate_rate_percent": 0.0,
            },
            "log_file": log_path.name,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = log_dir / f"drop_exam_duplicates_audit_{ts}.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)
        summary_path = log_dir / "drop_exam_duplicates_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)
        logging.info("Audit JSON: %s", audit_path.name)
        logging.info("Summary JSON: %s", summary_path.name)
        logging.info("== drop_exam_duplicates_v1 completed (no data) ==")
        return 0

    deduped_df, stats = deduplicate_data(df)
    output_hashes = save_partitions(deduped_df, output_dir, schema, compute_hashes)

    duration_s = time.time() - t0
    finished_at = datetime.now(timezone.utc)

    audit_obj = {
        "task": "drop_exam_duplicates_v1",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round(duration_s, 3),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "input_file_hashes": input_hashes,
        "output_file_hashes": output_hashes,
        "deduplication": stats,
        "log_file": log_path.name,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = log_dir / f"drop_exam_duplicates_audit_{ts}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)

    summary_path = log_dir / "drop_exam_duplicates_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)

    logging.info("== Summary ==")
    logging.info("  total_rows: %s", f"{stats['total_rows']:,}")
    logging.info("  unique_rows: %s", f"{stats['unique_rows']:,}")
    logging.info(
        "  duplicate_rows_removed: %s",
        f"{stats['duplicate_rows_removed']:,}",
    )
    logging.info(
        "  duplicate_rate_percent: %.6f%%",
        stats["duplicate_rate_percent"],
    )
    logging.info("Wrote output partitions to %s", output_dir)
    logging.info("Audit JSON: %s", audit_path.name)
    logging.info("Summary JSON: %s", summary_path.name)
    logging.info("== drop_exam_duplicates_v1 completed ==")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

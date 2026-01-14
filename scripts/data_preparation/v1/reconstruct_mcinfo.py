#!/usr/bin/env python3
# scripts/data_preparation/v1/reconstruct_mcinfo.py

"""Reconstruct mcinfo records into exam-level data with audit logging."""

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


EXAM_SCHEMA = pa.schema([
    ("exam_id", pa.string()),
    ("person_id", pa.string()),
    ("ExamDate", pa.date32()),
    ("tests", pa.list_(pa.struct([
        ("code", pa.string()),
        ("name", pa.string()),
        ("value_num", pa.float64()),
        ("value_cat", pa.string()),
        ("value_text", pa.string()),
        ("unit", pa.string()),
        ("type", pa.string()),
    ]))),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct mcinfo rows into exam-level records."
    )
    parser.add_argument(
        "--input",
        default="data/deidentified/v1/df_mcinfo_deidentified.parquet",
        help="Input Parquet path containing mcinfo records.",
    )
    parser.add_argument(
        "--output-base",
        default="data/deidentified/v1/mcinfo/exam_level",
        help="Directory where year-partitioned Parquet outputs are written.",
    )
    parser.add_argument(
        "--log-path",
        default="outputs/audit/v1/reconstruct_mcinfo_logs",
        help="Directory for execute log and audit JSON artifacts.",
    )
    parser.add_argument(
        "--master",
        default="data/external/test_code_master/Fmaster.csv",
        help="CSV mapping file for mcitem code to descriptive name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="Row batch size when scanning the parquet file.",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"reconstruct_mcinfo_{ts}.log"

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


def safe_json_loads(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logging.warning("JSON decoding failed for value: %s", value)
        return {}


def load_master_mapping(master_path: Path) -> Dict[str, str]:
    logging.info("Loading master data from %s ...", master_path)
    df_master = pd.read_csv(
        master_path,
        encoding="utf-8",
        dtype={
            "Item Name (JP)": "string",
            "Item Code": "string",
            "Data Type": "string",
        },
        keep_default_na=False,
    )
    if "CD/CO Encoding" in df_master.columns:
        df_master["CD/CO Encoding"] = df_master["CD/CO Encoding"].apply(safe_json_loads)

    mapping = df_master.set_index("Item Code")["Item Name (JP)"].to_dict()
    logging.info("Loaded master rows: %d", len(df_master))
    logging.info("Mapping sample: %s", list(mapping.items())[:5])
    return mapping


def generate_exam_id(person_id: str, exam_date: datetime) -> str:
    combined = f"{person_id}_{exam_date.strftime('%Y%m%d')}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def log_sample_overview(parquet_path: Path) -> Dict[str, Any]:
    logging.info("Reading sample data for quick inspection ...")
    parquet_file = pq.ParquetFile(str(parquet_path))
    try:
        first_batch = next(parquet_file.iter_batches())
    except StopIteration:
        logging.warning("Input parquet contains no row groups.")
        return {"rows_after_filter": 0}

    sample_df = pa.Table.from_batches([first_batch]).to_pandas()
    if "FlagError" in sample_df.columns:
        sample_df = sample_df[sample_df["FlagError"] == False]

    rows_after_filter = len(sample_df)
    logging.info("Sample rows after FlagError filter: %d", rows_after_filter)
    if rows_after_filter > 0:
        logging.info("Sample head:\n%s", sample_df.head().to_string(index=False))

    logging.info("Column dtypes observed in sample:")
    for col, dtype in sample_df.dtypes.items():
        logging.info("  %s: %s", col, dtype)

    unique_types: List[str] = []
    if "McItemResultType" in sample_df.columns:
        unique_types = sorted(sample_df["McItemResultType"].dropna().unique().tolist())
        logging.info("Unique McItemResultType: %s", unique_types)
    else:
        logging.warning("Column 'McItemResultType' not found in sample.")

    tests_distribution: Dict[str, float] = {}
    if {"AnonymousID", "McExamDt"}.issubset(sample_df.columns):
        sample_counts = (
            sample_df.groupby(["AnonymousID", "McExamDt"])
            .size()
            .reset_index(name="test_count")
        )
        if not sample_counts.empty:
            describe = sample_counts["test_count"].describe()
            tests_distribution = {k: float(v) for k, v in describe.to_dict().items()}
            logging.info(
                "Tests per exam distribution (from sample):\n%s",
                describe.to_string(),
            )

    head_records: List[Dict[str, Any]] = []
    if rows_after_filter > 0:
        head_records = sample_df.head().to_dict(orient="records")
        head_records = [
            {
                key: (
                    value.isoformat() if isinstance(value, pd.Timestamp) else value
                )
                for key, value in record.items()
            }
            for record in head_records
        ]

    return {
        "rows_after_filter": int(rows_after_filter),
        "unique_result_types": unique_types,
        "tests_per_exam_describe": tests_distribution,
        "head_records": head_records,
    }


def transform_to_exam_level(
    input_path: Path,
    output_base_path: Path,
    mcitemcd_to_name_map: Optional[Dict[str, str]] = None,
    chunk_size: int = 500000,
) -> Dict[str, Any]:
    start_time = time.time()

    parquet_file = pq.ParquetFile(str(input_path))
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else 0
    logging.info("Total rows in file: %d", total_rows)

    processed_rows = 0
    exam_count = 0
    test_count = 0
    chunk_index = 0
    year_tables: Dict[int, pa.Table] = {}

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk_index += 1
        chunk_df = pa.Table.from_batches([batch]).to_pandas()

        if "FlagError" in chunk_df.columns:
            chunk_df = chunk_df[chunk_df["FlagError"] == False]

        if "McExamDt" in chunk_df.columns:
            chunk_df["McExamDt"] = pd.to_datetime(
                chunk_df["McExamDt"], errors="coerce"
            )

        chunk_df = chunk_df.dropna(subset=["AnonymousID", "McExamDt"], how="any")

        logging.info("Processing chunk %d ...", chunk_index)
        if chunk_df.empty:
            logging.info("  Chunk empty after filtering; skipping.")
            continue

        test_count += len(chunk_df)

        grouped = chunk_df.groupby(["AnonymousID", "McExamDt"])
        year_exam_rows: Dict[int, List[Dict[str, Any]]] = {}

        for (person_id, exam_date), group in grouped:
            exam_date = pd.Timestamp(exam_date)
            exam_id = generate_exam_id(str(person_id), exam_date)

            tests: List[Dict[str, Any]] = []
            for _, row in group.iterrows():
                result_type = row.get("McItemResultType")

                value_num = None
                value_cat = None
                value_text = None

                if result_type == "PQ":
                    value_num = row.get("McItemResultValue")
                elif result_type in {"CD", "CO"}:
                    value_cat = row.get("McItemResultCd")
                    if pd.isna(value_cat):
                        value_cat = None
                elif result_type == "ST":
                    value_text = row.get("McItemResultText")
                    if pd.isna(value_text):
                        value_text = None

                test_code = row.get("McItemCd")
                test_name = "Unknown"
                if mcitemcd_to_name_map and test_code in mcitemcd_to_name_map:
                    test_name = mcitemcd_to_name_map[test_code]

                unit = row.get("McItemUnit")
                if pd.isna(unit):
                    unit = None

                tests.append(
                    {
                        "code": test_code,
                        "name": test_name,
                        "value_num": value_num,
                        "value_cat": value_cat,
                        "value_text": value_text,
                        "unit": unit,
                        "type": result_type,
                    }
                )

            exam_row = {
                "exam_id": exam_id,
                "person_id": person_id,
                "ExamDate": exam_date,
                "tests": tests,
            }

            year = int(exam_date.year)
            year_exam_rows.setdefault(year, []).append(exam_row)
            exam_count += 1

        for year, exam_rows in year_exam_rows.items():
            exam_df = pd.DataFrame(exam_rows)
            table = pa.Table.from_pandas(exam_df, schema=EXAM_SCHEMA)
            if year in year_tables:
                year_tables[year] = pa.concat_tables([year_tables[year], table])
            else:
                year_tables[year] = table

        processed_rows += len(chunk_df)
        pct = (processed_rows / total_rows * 100) if total_rows else 0.0
        logging.info(
            "Processed %d/%d rows (%.2f%%)",
            processed_rows,
            total_rows,
            pct,
        )

        for year, table in list(year_tables.items()):
            if table.num_rows > 100000:
                partition_path = output_base_path / f"year={year}"
                partition_path.mkdir(parents=True, exist_ok=True)
                output_path = partition_path / "data.parquet"

                if output_path.exists():
                    existing_table = pq.read_table(output_path)
                    table = pa.concat_tables([existing_table, table])

                pq.write_table(table, output_path, compression="snappy")
                logging.info("Written %d exams for year %d", table.num_rows, year)
                del year_tables[year]

    for year, table in year_tables.items():
        partition_path = output_base_path / f"year={year}"
        partition_path.mkdir(parents=True, exist_ok=True)
        output_path = partition_path / "data.parquet"

        if output_path.exists():
            existing_table = pq.read_table(output_path)
            table = pa.concat_tables([existing_table, table])

        pq.write_table(table, output_path, compression="snappy")
        logging.info("Written %d exams for year %d", table.num_rows, year)

    duration_s = time.time() - start_time
    logging.info("Transformation complete in %.2f seconds", duration_s)
    logging.info("Total exams: %d", exam_count)
    logging.info("Total tests: %d", test_count)

    return {
        "total_rows": int(total_rows),
        "filtered_rows": int(test_count),
        "exam_count": int(exam_count),
        "test_count": int(test_count),
        "chunks": int(chunk_index),
        "duration_s": round(duration_s, 2),
    }


def validate_transformation(
    output_base_path: Path,
    original_test_count: int,
) -> Dict[str, Any]:
    logging.info("Validating transformation ...")

    partition_dirs = sorted(
        [
            p
            for p in output_base_path.iterdir()
            if p.is_dir() and p.name.startswith("year=")
        ],
        key=lambda p: p.name,
    )

    transformed_exam_count = 0
    transformed_test_count = 0
    per_year: List[Dict[str, Any]] = []

    for partition_dir in partition_dirs:
        year_str = partition_dir.name.replace("year=", "")
        try:
            year = int(year_str)
        except ValueError:
            logging.warning("Unexpected partition name: %s", partition_dir.name)
            continue

        parquet_files = sorted(
            [p for p in partition_dir.iterdir() if p.suffix == ".parquet"],
            key=lambda p: p.name,
        )

        year_exam_count = 0
        year_test_count = 0

        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            file_exam_count = len(df)
            year_exam_count += file_exam_count

            file_test_count = int(sum(len(tests) for tests in df["tests"]))
            year_test_count += file_test_count

            transformed_exam_count += file_exam_count
            transformed_test_count += file_test_count

        logging.info(
            "Year %d: %d exams, %d tests",
            year,
            year_exam_count,
            year_test_count,
        )
        per_year.append(
            {
                "year": year,
                "exam_count": int(year_exam_count),
                "test_count": int(year_test_count),
            }
        )

    logging.info("Total transformed exams: %d", transformed_exam_count)
    logging.info("Total transformed tests: %d", transformed_test_count)

    counts_match = transformed_test_count == original_test_count
    if counts_match:
        logging.info("Test counts match original.")
    else:
        logging.warning(
            "Test counts do not match: original=%d, transformed=%d",
            original_test_count,
            transformed_test_count,
        )

    sample_snapshot: Dict[str, Any] = {}
    for partition_dir in partition_dirs:
        parquet_files = sorted(
            [p for p in partition_dir.iterdir() if p.suffix == ".parquet"],
            key=lambda p: p.name,
        )
        if not parquet_files:
            continue
        sample_df = pq.read_table(parquet_files[0]).to_pandas()
        if sample_df.empty:
            continue

        sample_snapshot["year"] = partition_dir.name.replace("year=", "")
        headline = sample_df[["exam_id", "person_id", "ExamDate"]].head()
        logging.info("Sample of transformed data:\n%s", headline.to_string(index=False))
        sample_snapshot["headline"] = [
            {
                key: (
                    value.isoformat() if isinstance(value, pd.Timestamp) else value
                )
                for key, value in record.items()
            }
            for record in headline.to_dict(orient="records")
        ]

        first_tests = sample_df.iloc[0]["tests"][:5]
        logging.info("Nested tests structure (first 5 tests of first exam):")
        for idx, test in enumerate(first_tests, start=1):
            logging.info("  %d: %s", idx, test)
        sample_snapshot["first_exam_tests"] = first_tests
        break

    logging.info("Validation complete.")
    return {
        "total_exams": int(transformed_exam_count),
        "total_tests": int(transformed_test_count),
        "test_count_match": bool(counts_match),
        "per_year": per_year,
        "sample": sample_snapshot,
    }


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

    input_path = Path(args.input)
    output_base_path = Path(args.output_base)
    log_dir = Path(args.log_path)
    master_path = Path(args.master)
    chunk_size = args.chunk_size

    output_base_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(log_dir)
    logging.info("== reconstruct_mcinfo_v1 started ==")
    logging.info("Project root: %s", project_root)
    logging.info("Input path: %s", input_path)
    logging.info("Output base: %s", output_base_path)
    logging.info("Master mapping: %s", master_path)
    logging.info("Chunk size: %d", chunk_size)
    logging.info("Log directory: %s", log_dir)
    logging.info("Log file: %s", log_path.name)

    started_at = datetime.now(timezone.utc)
    t0 = time.time()

    sample_info = log_sample_overview(input_path)
    mapping = load_master_mapping(master_path)

    transform_stats = transform_to_exam_level(
        input_path=input_path,
        output_base_path=output_base_path,
        mcitemcd_to_name_map=mapping,
        chunk_size=chunk_size,
    )

    validation_stats = validate_transformation(
        output_base_path=output_base_path,
        original_test_count=transform_stats["test_count"],
    )

    duration_s = time.time() - t0
    finished_at = datetime.now(timezone.utc)

    audit_obj = {
        "task": "reconstruct_mcinfo_v1",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round(duration_s, 3),
        "input_path": str(input_path),
        "output_base_path": str(output_base_path),
        "master_path": str(master_path),
        "log_directory": str(log_dir),
        "log_file": log_path.name,
        "chunk_size": int(chunk_size),
        "sample_overview": sample_info,
        "transformation": transform_stats,
        "validation": validation_stats,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = log_dir / f"reconstruct_mcinfo_audit_{ts}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)

    summary_path = log_dir / "reconstruct_mcinfo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2, default=json_default)

    logging.info("== Summary ==")
    logging.info(
        "  total_rows: %s",
        f"{transform_stats.get('total_rows', 0):,}",
    )
    logging.info(
        "  filtered_rows: %s",
        f"{transform_stats.get('filtered_rows', 0):,}",
    )
    logging.info(
        "  exams: %s",
        f"{transform_stats.get('exam_count', 0):,}",
    )
    logging.info(
        "  tests: %s",
        f"{transform_stats.get('test_count', 0):,}",
    )
    for item in validation_stats.get("per_year", []):
        logging.info(
            "  year %s -> exams=%s, tests=%s",
            item["year"],
            f"{item['exam_count']:,}",
            f"{item['test_count']:,}",
        )

    logging.info("Audit JSON: %s", audit_path.name)
    logging.info("Summary JSON: %s", summary_path.name)
    logging.info("== reconstruct_mcinfo_v1 completed ==")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

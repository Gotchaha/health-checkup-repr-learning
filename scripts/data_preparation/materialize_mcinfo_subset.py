#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize the mcinfo subset needed by a split, perfectly aligned to the split
manifest row order, for O(1)-style positional access during training.

Design goals (research/academic code style):
- Manifest-led "left join": we only keep rows present in the split manifest.
- Output Parquet contains ONLY one payload column: `tests`.
- Row group size is small (default 4096) to reduce per-sample random-read latency.
- Strong provenance/auditability:
  * JSON audit log under outputs/audit/materialize_split/
  * Parquet footer key-value metadata includes hashes for reproducibility.

Assumptions:
- mcinfo is Hive-partitioned by year: <mcinfo_dir>/year=YYYY/*.parquet
- `exam_id` uniquely identifies an exam-level row in mcinfo.
- Manifest Parquet contains at least: ["exam_id", "ExamDate"].
- We compute years from manifest["ExamDate"] and restrict scans per year.

Notes:
- This script supports full in-memory materialization (default) and can optionally
  fall back to manifest chunking if memory becomes tight (not needed on large-memory nodes).
- Training code can verify alignment by reading Parquet footer metadata only,
  without scanning the data columns.
"""

import os
import sys
import json
import math
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# -----------------------------
# Utilities (hashing, logging)
# -----------------------------

def sha256_file(path: Path, bufsize: int = 2**20) -> str:
    """Compute sha256 of a file in a streaming fashion."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_exam_id_order(exam_id_array: pa.ChunkedArray, chunk_len: int = 1_000_000) -> str:
    """
    Compute a content-order hash of the exam_id sequence (manifest order).
    We do not materialize a huge Python list; we hash chunk-wise to bound memory.
    """
    # Normalize to a flat, contiguous Arrow array for predictable chunk slicing
    arr = pc.cast(exam_id_array, pa.large_string())  # safe for very long strings
    # Concatenate chunks to simplify slicing in big strides
    if len(arr.chunks) > 1:
        arr = pa.chunked_array([pa.concat_arrays(arr.chunks)], type=arr.type)

    h = hashlib.sha256()
    n = len(arr)
    start = 0
    while start < n:
        end = min(start + chunk_len, n)
        # Join with '\n' to avoid ambiguity; encode once here.
        # Using to_pylist() in chunks is acceptable since we bound chunk_len.
        chunk_list = arr.slice(start, end - start).to_pylist()
        h.update(("\n".join(chunk_list)).encode("utf-8"))
        start = end
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Core materialization logic
# -----------------------------

def read_manifest(manifest_path: Path) -> pa.Table:
    """Read split manifest with minimal columns and compute year."""
    tbl = pq.read_table(str(manifest_path), columns=["exam_id", "ExamDate"])
    # Compute year for per-year scanning
    year = pc.year(tbl["ExamDate"])
    tbl = tbl.append_column("year", year)
    return tbl


def scan_mcinfo_by_year(mcinfo_dir: Path, year: int, exam_ids: pa.Array) -> pa.Table:
    """
    Vectorized scan within a single year partition:
    Keep only rows with exam_id in the given set. Project `exam_id` and `tests`.
    """
    y_dir = mcinfo_dir / f"year={int(year)}"
    if not y_dir.exists():
        # Return empty table if the partition doesn't exist (we will validate later).
        return pa.table({"exam_id": pa.array([], type=pa.string()),
                         "tests": pa.array([], type=pa.null())})
    dset = ds.dataset(str(y_dir), format="parquet", partitioning="hive")
    filt = pc.is_in(ds.field("exam_id"), value_set=exam_ids)
    # Only project what we need for alignment + payload
    t = dset.to_table(filter=filt, columns=["exam_id", "tests"])
    return t


def materialize_tests_table(manifest: pa.Table, mcinfo_dir: Path,
                            chunk_size: int | None = None) -> pa.Table:
    """
    Manifest-led left-join materialization that returns a Table with columns:
    - tests  (aligned exactly to manifest row order)
    Notes:
    - We gather rows per year using vectorized is_in on exam_id.
    - We then reorder to match manifest order exactly.
    - We validate 1:1 coverage and uniqueness along the way.
    """
    n = manifest.num_rows
    if n == 0:
        return pa.table({"tests": pa.array([], type=pa.null())})

    # Optionally support chunking over manifest (not needed if you have big memory).
    starts = [0] if not chunk_size else list(range(0, n, chunk_size))
    parts: list[pa.Table] = []

    for start in starts:
        end = n if not chunk_size else min(start + chunk_size, n)
        m = manifest.slice(start, end - start)

        # Validate manifest exam_id uniqueness within this slice (defensive)
        if pc.count_distinct(m["exam_id"]).as_py() != m.num_rows:
            raise RuntimeError("Duplicate exam_id found in manifest slice; cannot align 1:1.")

        years = pc.unique(m["year"]).to_pylist()
        year_tables = []
        for y in years:
            mask = pc.equal(m["year"], pa.scalar(y, pa.int32()))
            m_y = m.filter(mask)
            # Arrow array of exam_ids for this year
            exam_ids_y = m_y["exam_id"]
            t_y = scan_mcinfo_by_year(mcinfo_dir, int(y), exam_ids_y)
            if t_y.num_rows:
                year_tables.append(t_y)

        if not year_tables:
            raise RuntimeError(
                f"No mcinfo rows found for manifest rows [{start}:{end}]. "
                f"Check mcinfo_dir or manifest time window."
            )

        T = pa.concat_tables(year_tables, promote_options="default") if len(year_tables) > 1 else year_tables[0]

        # Validate: exam_id uniqueness in gathered rows (defensive against data issues)
        if pc.count_distinct(T["exam_id"]).as_py() != T.num_rows:
            # Pinpoint duplicates for better error messages
            # (group_by + count is heavy; keep message short to avoid large prints)
            raise RuntimeError("Duplicate exam_id detected inside mcinfo for the selected window.")

        # Reorder to EXACTLY match manifest order
        # We want rows in the order of m["exam_id"]
        # Approach: indices = index_in(m.exam_id, value_set=T.exam_id)
        # Then T_aligned = T.take(indices)
        indices = pc.index_in(m["exam_id"], value_set=T["exam_id"])
        if pc.any(pc.equal(indices, pa.scalar(-1, pa.int32()))).as_py():
            # Some manifest ids are missing in gathered T
            missing_mask = pc.equal(indices, pa.scalar(-1, pa.int32()))
            # Extract a tiny preview of missing IDs for debugging
            missing_ids = pc.filter(m["exam_id"], missing_mask).to_pylist()[:10]
            raise RuntimeError(f"Missing exam_id in mcinfo: sample={missing_ids} ...")

        T_aligned = T.take(indices)  # order matches m.exam_id
        parts.append(T_aligned.select(["tests"]))

    return pa.concat_tables(parts, promote_options="default") if len(parts) > 1 else parts[0]


def build_footer_metadata(manifest_path: Path,
                          manifest_sha256: str,
                          exam_id_order_sha256: str,
                          mcinfo_dir: Path,
                          row_group_size: int,
                          compression: str) -> dict:
    """Construct key-value metadata for Parquet footer."""
    meta = {
        "created_by": "materialize_mcinfo_subset.py",
        "created_at": iso_now(),
        "pyarrow_version": pa.__version__,
        "method": "manifest_left_join",
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": manifest_sha256,
        "exam_id_order_sha256": exam_id_order_sha256,
        "mcinfo_dir": str(mcinfo_dir),
        "partitioning": "hive(year=YYYY)",
        "row_group_size": str(row_group_size),
        "compression": compression,
    }
    # Parquet expects {str: bytes} or {str: str}; keep strings.
    return meta


def write_parquet_with_metadata(out_path: Path,
                                table: pa.Table,
                                metadata: dict,
                                row_group_size: int,
                                compression: str) -> None:
    """
    Write a Parquet file with small row groups and key-value metadata in the footer.
    We call ParquetWriter and write in slices of `row_group_size` to create small groups.
    """
    # Build a minimal schema with only `tests` and attach metadata
    schema = table.schema
    schema = schema.with_metadata({k: str(v) for k, v in metadata.items()})
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(str(out_path), schema=schema,
                              compression=None if compression == "none" else compression)
    try:
        n = table.num_rows
        # Create row groups by writing in chunks of `row_group_size`
        for start in range(0, n, row_group_size):
            end = min(start + row_group_size, n)
            writer.write_table(table.slice(start, end - start))
    finally:
        writer.close()


# -----------------------------
# Main entry
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Materialize mcinfo subset (tests only) for a split.")
    ap.add_argument("--mcinfo-dir", required=True,
                    help="Root of year-partitioned mcinfo Parquet, e.g., data/processed/mcinfo/exam_level")
    ap.add_argument("--manifest", required=True,
                    help="Sorted manifest Parquet for the target split")
    ap.add_argument("--out", required=True,
                    help="Output Parquet path (will contain only `tests` column)")
    ap.add_argument("--log-dir", default="outputs/audit/materialize_split",
                    help="Directory to write the JSON audit log (default: outputs/audit/materialize_split)")
    ap.add_argument("--row-group-size", type=int, default=4096,
                    help="Parquet row group size for random-access friendly reads (default: 4096)")
    ap.add_argument("--compression", default="zstd",
                    choices=["zstd", "snappy", "gzip", "none"],
                    help="Parquet compression codec (default: zstd)")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="Optional: manifest chunk size to bound memory; 0 means no chunking (default)")
    args = ap.parse_args()

    t0 = datetime.now(timezone.utc)

    mcinfo_dir = Path(args.mcinfo_dir)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    log_dir = Path(args.log_dir)

    # Derive a log file name from output file stem + timestamp
    ts = t0.strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{out_path.stem}_{ts}.json"

    # Compute manifest sha and order hash up front
    manifest_sha = sha256_file(manifest_path)
    manifest_tbl = read_manifest(manifest_path)
    exam_order_sha = sha256_exam_id_order(manifest_tbl["exam_id"])

    # Build materialized table (tests only), aligned to manifest order
    table = materialize_tests_table(
        manifest=manifest_tbl,
        mcinfo_dir=mcinfo_dir,
        chunk_size=(None if args.chunk_size <= 0 else args.chunk_size),
    )

    # Sanity: row count must match manifest
    if table.num_rows != manifest_tbl.num_rows:
        raise RuntimeError(
            f"Row count mismatch: materialized {table.num_rows} vs manifest {manifest_tbl.num_rows}"
        )

    # Write Parquet with footer metadata
    footer_meta = build_footer_metadata(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        exam_id_order_sha256=exam_order_sha,
        mcinfo_dir=mcinfo_dir,
        row_group_size=args.row_group_size,
        compression=args.compression,
    )
    write_parquet_with_metadata(
        out_path=out_path,
        table=table.select(["tests"]),
        metadata=footer_meta,
        row_group_size=args.row_group_size,
        compression=args.compression,
    )

    # Hash the output file for audit
    out_sha = sha256_file(out_path)

    t1 = datetime.now(timezone.utc)
    audit = {
        "script": "materialize_mcinfo_subset.py",
        "version": "1",
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "duration_sec": (t1 - t0).total_seconds(),
        "inputs": {
            "mcinfo_dir": str(mcinfo_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_rows": manifest_tbl.num_rows,
            "exam_id_order_sha256": exam_order_sha,
        },
        "output": {
            "out_path": str(out_path),
            "rows_written": table.num_rows,
            "row_group_size": args.row_group_size,
            "compression": args.compression,
            "out_sha256": out_sha,
        },
        "parameters": {
            "chunk_size": (None if args.chunk_size <= 0 else args.chunk_size),
        },
        "environment": {
            "python": sys.version.split()[0],
            "pyarrow": pa.__version__,
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        },
        "method": "manifest_left_join",
        "notes": [
            "Output Parquet contains only the `tests` column.",
            "Parquet footer stores manifest path/hash and exam_id order hash for runtime verification.",
        ],
    }

    write_json(log_path, audit)
    print(f"[OK] Wrote {table.num_rows} rows to {out_path}")
    print(f"[OK] Audit log: {log_path}")


if __name__ == "__main__":
    main()

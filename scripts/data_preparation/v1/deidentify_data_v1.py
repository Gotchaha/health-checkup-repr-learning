# scripts/data_preparation/v1/deidentify_data_v1.py
"""
De-identify healthcare data (v1): ID mapping + date jitter + PHI scrubbing.

This v1 script aligns with:
- New deidentifier module: src/data/cleaning/v1/deidentifier.py
- New patterns file format: {"meta": {...}, "patterns": {...}}
- No "interview" flow (dropped in v1)
- Simpler, no chunked processing for mcinfo (sufficient memory)
- Default IO & logs rooted under v1 paths

Defaults (can be overridden via CLI):
- Inputs:
    data/normalized/v1/df_mcinfo_cleaned.parquet
    data/normalized/v1/result_per_exam_cleaned_dedup.parquet
- Outputs:
    data/deidentified/v1/df_mcinfo_deidentified.parquet
    data/deidentified/v1/result_per_exam_deidentified.parquet
- Patterns:
    config/cleaning/v1/deidentification/phi_patterns.yaml
- Logs (runtime + audit):
    outputs/audit/v1/deidentification/
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
import yaml

# Add project root to path to enable imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import v1 deidentifier
from src.data.cleaning.v1.deidentifier import (
    PHIScrubber,
    process_mcinfo_data,
    process_result_data,
)


# ------------------------- Logging ------------------------- #
def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging to file (timestamped) and console."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"deidentify_v1_{ts}.log")

    logger = logging.getLogger("deidentify_v1")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logging initialized")
    logger.info("Log file: %s", logfile)
    return logger


# ------------------------- Helpers ------------------------- #
def load_patterns_meta(patterns_file: str) -> Dict[str, Any]:
    """Load YAML and return (meta, patterns). Raise on invalid structure."""
    with open(patterns_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "patterns" not in data:
        raise ValueError("Invalid patterns file: top-level key 'patterns' is required.")
    meta = data.get("meta", {})
    patterns = data["patterns"]
    return {"meta": meta, "patterns": patterns}


def make_json_serializable(obj):
    """Best-effort conversion for audit JSON."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def save_audit_data(stats: Dict[str, Any],
                    config: Dict[str, Any],
                    patterns_meta: Dict[str, Any],
                    log_dir: str) -> str:
    """Save audit JSON including meta/version info for patterns."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_file = os.path.join(log_dir, f"deidentification_audit_v1_{ts}.json")

    # Aggregate summary
    total_original = int(sum(s.get("original_rows", 0) for s in stats.values()))
    total_kept = int(sum(s.get("kept_rows", 0) for s in stats.values()))
    total_removed = int(sum(s.get("removed_rows", 0) for s in stats.values()))
    phi_total = int(stats.get("result", {}).get("phi_scrubbing", {}).get("total_replacements", 0))

    audit_payload = {
        "timestamp": datetime.now().isoformat(),
        "version": "v1",
        "config": make_json_serializable(config),
        "patterns_meta": make_json_serializable(patterns_meta.get("meta", {})),
        "stats": make_json_serializable(stats),
        "summary": {
            "total_original_rows": total_original,
            "total_kept_rows": total_kept,
            "total_removed_rows": total_removed,
            "phi_replacements_total": phi_total,
        },
    }

    try:
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Fallback to simplified audit if serialization fails
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "version": "v1",
                    "summary": audit_payload["summary"],
                    "error": f"Failed to serialize full audit: {e}",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    return audit_file


# ------------------------- File processors ------------------------- #
def process_mcinfo_file(input_path: str,
                        output_path: str,
                        mapping: pd.DataFrame) -> Dict[str, Any]:
    """
    Process mcinfo file (ID mapping + date jitter). No chunking (v1).
    """
    logger = logging.getLogger("deidentify_v1")
    logger.info(f"Processing mcinfo data: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded mcinfo data: {len(df):,} rows")

    result, stats = process_mcinfo_data(df, mapping)
    logger.info(f"Processed mcinfo data: {len(result):,} rows kept, {stats['removed_rows']:,} rows removed")

    result.to_parquet(output_path, index=False)
    logger.info(f"Saved deidentified mcinfo data to: {output_path}")

    return stats


def process_result_file(input_path: str,
                        output_path: str,
                        mapping: pd.DataFrame,
                        phi_scrubber: PHIScrubber) -> Dict[str, Any]:
    """
    Process result file (ID mapping + date jitter + PHI scrubbing).
    """
    logger = logging.getLogger("deidentify_v1")
    logger.info(f"Processing result data: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded result data: {len(df):,} rows")

    result, stats = process_result_data(df, mapping, phi_scrubber)
    logger.info(f"Processed result data: {len(result):,} rows kept, {stats['removed_rows']:,} rows removed")

    result.to_parquet(output_path, index=False)
    logger.info(f"Saved deidentified result data to: {output_path}")

    return stats


# ------------------------- CLI ------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="De-identify healthcare data files (v1)")
    # Inputs
    parser.add_argument(
        "--mcinfo",
        default="data/normalized/v1/df_mcinfo_cleaned.parquet",
        help="Path to input mcinfo parquet",
    )
    parser.add_argument(
        "--result",
        default="data/normalized/v1/result_per_exam_cleaned_dedup.parquet",
        help="Path to input result parquet",
    )
    parser.add_argument(
        "--mapping",
        default="data/private_backup/id_mapping.parquet",
        help="Path to ID mapping parquet (contains original_id, new_id, date_offset, keep_record)",
    )
    # Patterns
    parser.add_argument(
        "--patterns",
        default="config/cleaning/v1/deidentification/phi_patterns.yaml",
        help="Path to v1 PHI patterns YAML (with meta + patterns)",
    )
    # Outputs
    parser.add_argument(
        "--output-dir",
        default="data/deidentified/v1/",
        help="Directory for deidentified outputs",
    )
    # Logs
    parser.add_argument(
        "--log-dir",
        default="outputs/audit/v1/deidentification/",
        help="Directory to store runtime logs and audit JSON",
    )
    # Skips
    parser.add_argument(
        "--skip-mcinfo",
        action="store_true",
        help="Skip processing mcinfo data",
    )
    parser.add_argument(
        "--skip-result",
        action="store_true",
        help="Skip processing result data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_dir)
    logger.info("Starting de-identification process (v1)")

    # Resolve outputs
    os.makedirs(args.output_dir, exist_ok=True)
    out_mcinfo = os.path.join(args.output_dir, "df_mcinfo_deidentified.parquet")
    out_result = os.path.join(args.output_dir, "result_per_exam_deidentified.parquet")

    # Load mapping
    try:
        logger.info(f"Loading ID mapping: {args.mapping}")
        mapping = pd.read_parquet(args.mapping)
        logger.info(f"Loaded mapping rows: {len(mapping):,}")
        filtered_ids = int((~mapping["keep_record"]).sum()) if "keep_record" in mapping.columns else 0
        logger.info(f"IDs with keep_record=False: {filtered_ids:,}")
    except Exception as e:
        logger.exception(f"Failed to load ID mapping: {e}")
        return 1

    # Load patterns (for meta & PHIScrubber)
    try:
        logger.info(f"Loading PHI patterns: {args.patterns}")
        pat = load_patterns_meta(args.patterns)  # validates structure
        phi_scrubber = PHIScrubber(args.patterns)
        cat_count = len(phi_scrubber.compiled_patterns)
        pat_count = sum(len(v) for v in phi_scrubber.compiled_patterns.values())
        logger.info(f"Initialized PHIScrubber: {cat_count} categories, {pat_count} patterns")
    except Exception as e:
        logger.exception(f"Failed to initialize PHI patterns: {e}")
        return 1

    stats: Dict[str, Any] = {}

    # 1) mcinfo (ID mapping + date jitter)
    if not args.skip_mcinfo and os.path.exists(args.mcinfo):
        try:
            stats["mcinfo"] = process_mcinfo_file(args.mcinfo, out_mcinfo, mapping)
        except Exception as e:
            logger.exception(f"Failed to process mcinfo: {e}")
            stats["mcinfo"] = {"error": str(e)}
    else:
        if args.skip_mcinfo:
            logger.info("Skipping mcinfo as requested")
        else:
            logger.warning(f"Mcinfo file not found: {args.mcinfo}")
        stats["mcinfo"] = {"error": "Skipped or file not found"}

    # 2) result (ID mapping + date jitter + PHI scrubbing)
    if not args.skip_result and os.path.exists(args.result):
        try:
            stats["result"] = process_result_file(args.result, out_result, mapping, phi_scrubber)
        except Exception as e:
            logger.exception(f"Failed to process result: {e}")
            stats["result"] = {"error": str(e)}
    else:
        if args.skip_result:
            logger.info("Skipping result as requested")
        else:
            logger.warning(f"Result file not found: {args.result}")
        stats["result"] = {"error": "Skipped or file not found"}

    # Save audit JSON (includes patterns meta)
    audit_file = save_audit_data(
        stats=stats,
        config={
            "inputs": {
                "mcinfo": args.mcinfo,
                "result": args.result,
            },
            "outputs": {
                "mcinfo": out_mcinfo,
                "result": out_result,
            },
            "mapping_file": args.mapping,
            "patterns_file": args.patterns,
        },
        patterns_meta=pat,
        log_dir=args.log_dir,
    )

    # Console summary
    logger.info("\nDe-identification process complete (v1)\n")
    print("\n" + "=" * 60)
    print("DE-IDENTIFICATION PROCESS SUMMARY (v1)")
    print("=" * 60)

    if "mcinfo" in stats and "original_rows" in stats["mcinfo"]:
        kept = stats["mcinfo"]["kept_rows"]
        orig = stats["mcinfo"]["original_rows"]
        pct = (kept / orig * 100) if orig else 0.0
        print(f"Mcinfo data: {kept:,}/{orig:,} rows kept ({pct:.1f}%)")

    if "result" in stats and "original_rows" in stats["result"]:
        kept = stats["result"]["kept_rows"]
        orig = stats["result"]["original_rows"]
        pct = (kept / orig * 100) if orig else 0.0
        print(f"Result data: {kept:,}/{orig:,} rows kept ({pct:.1f}%)")
        # PHI stats
        if "phi_scrubbing" in stats["result"]:
            phi_stats = stats["result"]["phi_scrubbing"]
            print(f"  PHI replacements: {phi_stats['total_replacements']:,} in {phi_stats['processed_fields']:,} fields")
            for category, count in phi_stats["replacements_by_category"].items():
                if count > 0:
                    print(f"    {category}: {count:,}")

    print("\nOutput files:")
    for label, path in (("mcinfo", out_mcinfo), ("result", out_result)):
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {label}: {path} ({size_mb:.1f} MB)")

    print(f"\nAudit JSON: {audit_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

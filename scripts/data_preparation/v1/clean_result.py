# scripts/data_preparation/v1/clean_result.py

import os
import sys
import json
import time
import argparse
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yaml

# ---------------------------------------------------------------------
# Make project root importable (scripts/data_preparation/v1 -> project)
# ---------------------------------------------------------------------
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use precompiled cleaner and batch utilities from the lib
from src.data.cleaning.v1.result_text_cleaner import build_cleaner, clean_series


# -------------------------
# Utilities
# -------------------------
def setup_logging(log_dir: Path) -> Path:
    """Configure console + file logging. Returns the log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"clean_result_{ts}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return log_path


def file_sha256(path: Path, enable: bool = True) -> str | None:
    """Compute SHA256 for a file (None if disabled)."""
    if not enable:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent_dir(p: Path) -> None:
    """Create parent directories for a path if they do not exist."""
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)


def quantiles(series: pd.Series, qs=(0.5, 0.9, 0.99)) -> dict:
    """Return quantiles as {p50, p90, p99} from a numeric Series."""
    qv = series.quantile(list(qs), interpolation="linear").to_dict()
    return {f"p{int(q*100)}": float(v) for q, v in qv.items()}


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean ResultText using v1 pipeline and emit audit logs."
    )
    parser.add_argument(
        "--input",
        default="data/raw/result_per_exam.parquet",
        help="Input Parquet path.",
    )
    parser.add_argument(
        "--output",
        default="data/normalized/v1/result_per_exam_cleaned.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--config",
        default="config/cleaning/v1/result_text_clean_v1.yaml",
        help="Cleaning YAML config path.",
    )
    parser.add_argument(
        "--audit-dir",
        default="outputs/audit/v1/clean_result_logs",
        help="Directory for JSON and LOG audit artifacts.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Disable SHA256 computation for speed.",
    )
    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)
    audit_dir = Path(args.audit_dir)

    ensure_parent_dir(output_path)
    audit_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(audit_dir)
    logging.info("== clean_result_v1 started ==")
    logging.info(f"Project root: {project_root}")
    logging.info(f"Input:  {input_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Config: {config_path}")
    logging.info(f"Audit:  {audit_dir}")
    logging.info(f"Log:    {log_path.name}")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # Load cleaning config
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    # Optional file hashes
    input_hash = file_sha256(input_path, enable=not args.no_hash)
    config_hash = file_sha256(config_path, enable=not args.no_hash)
    if input_hash:
        logging.info(f"input_sha256:  {input_hash}")
    if config_hash:
        logging.info(f"config_sha256: {config_hash}")

    # Load data (expects 'ResultText' column)
    logging.info("Loading input parquet...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    if "ResultText" not in df.columns:
        logging.error("Input file missing 'ResultText' column.")
        return 2

    total = len(df)
    logging.info(f"Rows: {total:,}")

    # Build precompiled cleaner and run batch cleaning
    logging.info("Building precompiled cleaner…")
    cleaner = build_cleaner(cfg)

    logging.info("Applying cleaning pipeline (precompiled + batch)…")
    s_orig = df["ResultText"].astype("object")
    s_clean, stats = clean_series(s_orig, cleaner, progress=False)

    # Normalize empty results to NA for consistency
    s_clean = s_clean.where(~s_clean.fillna("").astype(str).str.strip().eq(""), other=pd.NA)

    # Derive "emptied by clean"
    pre_non_empty_mask = s_orig.notna() & s_orig.astype(str).str.strip().ne("")
    post_non_empty_mask = s_clean.notna() & s_clean.astype(str).str.strip().ne("")
    emptied_by_clean = int((pre_non_empty_mask & ~post_non_empty_mask).sum())

    # Write output parquet
    df["ResultText"] = s_clean.astype("string")  # Arrow-friendly dtype
    logging.info("Writing cleaned parquet...")
    df.to_parquet(output_path, engine="pyarrow", index=False)

    finished_at = datetime.now(timezone.utc).isoformat()
    duration_s = time.time() - t0

    # Map batch stats into the audit structure
    non_empty_before = total - int(stats.get("empty_before", 0))
    non_empty_after = total - int(stats.get("empty_after", 0))

    audit_obj = {
        "task": "clean_result_v1",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 3),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config_path": str(config_path),
        "input_sha256": input_hash,
        "config_sha256": config_hash,
        "stats": {
            "total": int(stats.get("total", total)),
            "non_empty_before": int(non_empty_before),
            "non_empty_after": int(non_empty_after),
            "emptied_by_clean": int(emptied_by_clean),
            "modified_count": int(stats.get("modified_count", 0)),
            "modified_rate": float(stats.get("modified_rate", 0.0)),
            "length_delta": stats.get("length_delta", {}),
        },
    }

    # Write audit JSONs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = audit_dir / f"clean_result_audit_{ts}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2)

    summary_path = audit_dir / "clean_result_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2)

    # Log summary
    logging.info("== Summary ==")
    logging.info(f"  total:               {audit_obj['stats']['total']:,}")
    logging.info(f"  non_empty_before:    {audit_obj['stats']['non_empty_before']:,}")
    logging.info(f"  non_empty_after:     {audit_obj['stats']['non_empty_after']:,}")
    logging.info(f"  emptied_by_clean:    {audit_obj['stats']['emptied_by_clean']:,}")
    logging.info(f"  modified_count:      {audit_obj['stats']['modified_count']:,} "
                 f"({audit_obj['stats']['modified_rate']:.3%})")
    ld = audit_obj["stats"]["length_delta"]
    if ld:
        logging.info(
            "  length_delta: mean={mean:.2f}, p50={p50:.0f}, p90={p90:.0f}, p99={p99:.0f}, "
            "min={min}, max={max}".format(**ld)
        )
    logging.info(f"Wrote: {output_path}")
    logging.info(f"Audit JSON: {audit_path.name}")
    logging.info(f"Summary JSON: {summary_path.name}")
    logging.info("== clean_result_v1 completed ==")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

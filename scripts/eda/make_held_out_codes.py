# scripts/eda/make_held_out_codes.py
"""
Compute test frequency on the train_ssl dataset, identify the bottom 15% least
frequent test codes, verify their presence in test_future, and save the final
list to the config file.

Run after generating the chronological splits:

    python scripts/eda/make_held_out_codes.py

Optional flags:

    --percentile <float>   Percentile threshold for "rare" codes (default: 15.0)
    --min-count <int>      Minimum occurrence count to include a code (default: 1)

Assumed repo-relative layout:
    • data/splits/core/train_ssl.parquet
    • data/splits/core/test_future.parquet
    • data/processed/mcinfo/exam_level/<year>/data.parquet
    • outputs/eda/train_test_frequency/        (artifacts, git-ignored)
    • config/splitting/held_out_codes.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401
    """Return parsed command-line args."""
    p = argparse.ArgumentParser(
        prog="make_held_out_codes.py",
        description="Compute test frequency and generate held-out codes list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--exams-root",
        type=Path,
        default=Path("data/processed/mcinfo/exam_level"),
        help="Folder with <year>/data.parquet partitions",
    )
    p.add_argument(
        "--splits-root",
        type=Path,
        default=Path("data/splits/core"),
        help="Folder with train_ssl.parquet and test_future.parquet manifests",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/eda/train_test_frequency"),
        help="Where PNG and raw Parquet go (will be created)",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config/splitting"),
        help="Folder to store held_out_codes.yaml",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=15.0,
        help="Percentile threshold for 'rare' codes (0.0-100.0)",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum occurrence count to include a code",
    )
    
    return p.parse_args()

# -----------------------------------------------------------------------------
# Data loading and processing
# -----------------------------------------------------------------------------

def load_manifest(file_path: Path) -> pa.Table:
    """Load a manifest file and return a PyArrow Table."""
    if not file_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {file_path}")
    
    return pq.read_table(file_path)

def extract_test_code_frequencies(manifest: pa.Table, exams_root: Path) -> list[tuple[str, int]]:
    """Extract test code frequencies from exams matching the manifest."""
    logger.info(f"Processing exams for {manifest.num_rows:,} records from manifest")
    
    # Create a set of (person_id, ExamDate) tuples for fast lookup using pandas
    # This ensures consistent date type handling
    manifest_df = manifest.to_pandas()
    manifest_keys = set()
    
    for _, row in manifest_df.iterrows():
        manifest_keys.add((row["person_id"], row["ExamDate"]))
    
    logger.info(f"Created {len(manifest_keys):,} unique keys from manifest")
    
    # Initialize counter for test codes
    code_counter = Counter()
    
    # Process the dataset in chunks to avoid memory issues
    dataset = ds.dataset(exams_root, format="parquet", partitioning="hive")
    
    # Process in batches
    batch_size = 10000
    num_exams_processed = 0
    num_matching_exams = 0
    
    for batch in dataset.to_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        df = table.to_pandas()
        
        # Filter exams that match the manifest
        for _, row in df.iterrows():
            num_exams_processed += 1
            if num_exams_processed % 100000 == 0:
                logger.info(f"Processed {num_exams_processed:,} exams...")
                if num_matching_exams > 0:
                    logger.info(f"Found {num_matching_exams} matching exams so far")
            
            key = (row["person_id"], row["ExamDate"])
            if key in manifest_keys:
                num_matching_exams += 1
                # Extract test codes from this exam
                tests = row["tests"]
                
                for test in tests:
                    code = test["code"]
                    code_counter[code] += 1
    
    logger.info(f"Processed {num_exams_processed:,} exams, found {num_matching_exams:,} matching exams")
    
    # Convert to list of tuples (code, count) sorted by count
    code_counts = sorted(code_counter.items(), key=lambda x: x[1])
    
    return code_counts

def identify_rare_codes(
    code_counts: list[tuple[str, int]], 
    percentile: float,
    min_count: int
) -> list[str]:
    """Identify the bottom percentile of least frequent test codes."""
    logger.info(f"Identifying bottom {percentile}% of test codes")
    
    # Filter by minimum count
    filtered_counts = [(code, count) for code, count in code_counts if count >= min_count]
    
    if not filtered_counts:
        logger.warning(f"No codes with count >= {min_count}")
        return []
    
    # Calculate cutoff index
    total_codes = len(filtered_counts)
    cutoff_idx = max(1, int(total_codes * percentile / 100))
    
    # Extract rare codes
    rare_codes = [code for code, _ in filtered_counts[:cutoff_idx]]
    
    logger.info(f"Identified {len(rare_codes)} rare codes out of {total_codes} total codes")
    return rare_codes

def verify_test_future_presence(rare_codes: list[str], test_manifest: pa.Table, exams_root: Path) -> list[str]:
    """Verify which rare codes also appear in the test_future set."""
    logger.info("Verifying presence of rare codes in test_future set")
    
    if not rare_codes:
        logger.warning("No rare codes to verify in test set")
        return []
    
    # Create a set of (person_id, ExamDate) tuples for fast lookup using pandas
    # This ensures consistent date type handling
    test_df = test_manifest.to_pandas()
    test_keys = set()
    
    for _, row in test_df.iterrows():
        test_keys.add((row["person_id"], row["ExamDate"]))
    
    logger.info(f"Created {len(test_keys):,} unique keys from test manifest")
    
    # Collect all codes in test_future
    test_codes = set()
    
    # Process the dataset in chunks
    dataset = ds.dataset(exams_root, format="parquet", partitioning="hive")
    
    num_exams_processed = 0
    num_matching_exams = 0
    
    for batch in dataset.to_batches(batch_size=10000):
        table = pa.Table.from_batches([batch])
        df = table.to_pandas()
        
        # Filter exams that match the test manifest
        for _, row in df.iterrows():
            num_exams_processed += 1
            if num_exams_processed % 100000 == 0:
                logger.info(f"Processed {num_exams_processed:,} exams in test verification...")
            
            key = (row["person_id"], row["ExamDate"])
            if key in test_keys:
                num_matching_exams += 1
                # Extract test codes from this exam
                tests = row["tests"]
                
                for test in tests:
                    code = test["code"]
                    test_codes.add(code)
    
    logger.info(f"Test verification: Processed {num_exams_processed:,} exams, found {num_matching_exams:,} matching exams")
    
    # Filter rare_codes to only those present in test_future
    verified_codes = [code for code in rare_codes if code in test_codes]
    
    logger.info(f"Found {len(verified_codes)} rare codes that also appear in test_future")
    return verified_codes

def sha_folder(path: Path) -> str:
    """Return SHA-256 digest of a folder listing (filenames + size)."""
    h = hashlib.sha256()
    for fp in sorted(path.rglob("*.parquet")):
        stat = fp.stat()
        h.update(f"{fp.relative_to(path)}::{stat.st_size}".encode())
    return h.hexdigest()[:12]

# -----------------------------------------------------------------------------
# Artifact generation
# -----------------------------------------------------------------------------

def save_frequency_artifacts(
    code_counts: list[tuple[str, int]],
    verified_rare_codes: list[str],
    out_dir: Path,
    config_dir: Path,
    percentile: float,
    min_count: int,
    exams_root: Path,
) -> None:
    """Save artifacts including raw data, visualization, and config file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle empty code_counts
    if not code_counts:
        logger.warning("No code frequency data to save")
        # Create empty Parquet with schema
        tbl = pa.Table.from_pydict({
            "code": pa.array([], type=pa.string()),
            "count": pa.array([], type=pa.int64()),
            "is_rare": pa.array([], type=pa.bool_())
        })
        pq.write_table(tbl, out_dir / "test_code_frequency.parquet")
        
        # Create config with empty list
        payload = {
            "held_out_codes": [],
            "percentile": float(percentile),
            "min_count": int(min_count),
            "total_codes": 0,
            "num_held_out": 0,
            "exams_root_sha": sha_folder(exams_root),
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warning": "No test codes found in the training data."
        }
        
        with open(config_dir / "held_out_codes.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        
        logger.info("Empty config file saved due to no code data")
        return
    
    # 1. Raw Parquet with code frequencies
    codes, counts = zip(*code_counts)
    tbl = pa.Table.from_pydict({
        "code": codes,
        "count": counts,
        "is_rare": [code in verified_rare_codes for code in codes]
    })
    pq.write_table(tbl, out_dir / "test_code_frequency.parquet")
    
    # 2. PNG plot - histogram of test code frequencies
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Use log scale for better visualization
        counts_np = np.array(counts)
        plt.hist(counts_np, bins=50, log=True)
        
        # Mark the cutoff point
        cutoff_index = int(len(counts) * percentile / 100)
        if cutoff_index < len(counts):
            cutoff_value = counts[cutoff_index]
            plt.axvline(x=cutoff_value, color='r', linestyle='--', 
                      label=f'{percentile}% cutoff: {cutoff_value} occurrences')
        
        plt.xlabel('Number of occurrences (log scale)')
        plt.ylabel('Number of test codes (log scale)')
        plt.title('Distribution of Test Code Frequencies in Train-SSL')
        plt.xscale('log')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "test_code_frequency.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
    
    # 3. Config YAML with the final list of held-out codes
    # Make sure all codes are explicitly treated as strings in the YAML
    payload = {
        "held_out_codes": [str(code) for code in verified_rare_codes],  # Force string type
        "percentile": float(percentile),
        "min_count": int(min_count),
        "total_codes": len(code_counts),
        "num_held_out": len(verified_rare_codes),
        "exams_root_sha": sha_folder(exams_root),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    
    # Use a custom YAML dumper to force consistent string formatting
    class StringDumper(yaml.SafeDumper):
        """Custom YAML dumper that forces consistent string formatting."""
        pass
    
    # Add a string representer that always uses quotes
    def represent_str(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    
    StringDumper.add_representer(str, represent_str)
    
    with open(config_dir / "held_out_codes.yaml", "w", encoding="utf-8") as f:
        yaml.dump(payload, f, Dumper=StringDumper, sort_keys=False)
    
    # 4. Log summary to stdout
    logger.info(f"Total unique test codes: {len(code_counts):,}")
    logger.info(f"Held-out codes: {len(verified_rare_codes):,} ({percentile:.1f}%)")
    logger.info(f"Config saved to: {config_dir / 'held_out_codes.yaml'}")
    logger.info(f"Frequency data saved to: {out_dir / 'test_code_frequency.parquet'}")
    logger.info(f"Visualization saved to: {out_dir / 'test_code_frequency.png'}")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute the main script logic."""
    args = parse_args()
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load train_ssl manifest
    logger.info("Loading Train-SSL manifest")
    train_manifest_path = args.splits_root / "train_ssl.parquet"
    train_manifest = load_manifest(train_manifest_path)
    
    # 2. Extract test code frequencies from train_ssl exams
    code_counts = extract_test_code_frequencies(train_manifest, args.exams_root)
    
    # 3. Identify rare codes
    rare_codes = identify_rare_codes(code_counts, args.percentile, args.min_count)
    
    # 4. Load test_future manifest
    logger.info("Loading Test-Future manifest")
    test_manifest_path = args.splits_root / "test_future.parquet"
    test_manifest = load_manifest(test_manifest_path)
    
    # 5. Verify which rare codes appear in test_future
    verified_rare_codes = verify_test_future_presence(rare_codes, test_manifest, args.exams_root)
    
    # 6. Save artifacts
    save_frequency_artifacts(
        code_counts,
        verified_rare_codes,
        args.out_dir,
        args.config_dir,
        args.percentile,
        args.min_count,
        args.exams_root,
    )

if __name__ == "__main__":
    main()
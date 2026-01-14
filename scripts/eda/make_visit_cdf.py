"""scripts/eda/make_visit_cdf.py

Compute the per‑person visit‑count CDF on the *exam‑level* dataset and record
thresholds that will be re‑used by downstream split‑generation steps.

Run it once after you have the final, de‑identified Parquet partitions:

    python scripts/eda/make_visit_cdf.py

Optional flags to **override** the automatic choices:

    --cold-start-K <int>    Fixed K for cold‑start (skip 80‑percentile rule)
    --sparse-max   <int>    Upper bound of sparse bucket
    --dense-min    <int>    Lower bound of dense bucket

If any of these are omitted, sensible defaults are derived from the CDF.

Assumed repo‑relative layout:
    • data/processed/mcinfo/exam_level/<year>/data.parquet
    • outputs/eda/visit_cdf/            (artefacts, git‑ignored)
    • config/splitting/visit_thresholds.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401
    """Return parsed command‑line args."""
    p = argparse.ArgumentParser(
        prog="make_visit_cdf.py",
        description="Compute visit‑count CDF and write threshold YAML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--exams-root",
        type=Path,
        default=Path("data/processed/mcinfo/exam_level"),
        help="Folder with <year>/data.parquet partitions",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/eda/visit_cdf"),
        help="Where PNG and raw Parquet go (will be created)",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config/splitting"),
        help="Folder to store visit_thresholds.yaml",
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of persons to sample for a quick run (1.0 = all)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")

    # Override knobs -----------------------------------------------------------
    p.add_argument("--cold-start-K", type=int, default=None)
    p.add_argument("--sparse-max", type=int, default=None)
    p.add_argument("--dense-min", type=int, default=None)

    return p.parse_args()

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def compute_visit_counts(exams_root: Path, sample_frac: float, seed: int) -> np.ndarray:
    """Return sorted numpy array of visit counts per *person_id*."""
    dataset = ds.dataset(exams_root, format="parquet", partitioning="hive")
    if "person_id" not in dataset.schema.names:
        raise RuntimeError("Column 'person_id' not found in exam‑level dataset")

    tbl = dataset.to_table(columns=["person_id"])

    if 0.0 < sample_frac < 1.0:
        idx = np.random.default_rng(seed).choice(
            len(tbl), size=int(len(tbl) * sample_frac), replace=False
        )
        tbl = tbl.take(pa.array(idx))

    grp = tbl.group_by("person_id").aggregate([("person_id", "count")])
    counts = grp.column(1).to_numpy()
    return np.sort(counts)


def calc_cdf(counts: np.ndarray) -> np.ndarray:
    """Empirical CDF for sorted counts."""
    return np.arange(1, len(counts) + 1) / len(counts)


def sha_folder(path: Path) -> str:
    """Return SHA‑256 digest of a folder listing (filenames + size)."""
    h = hashlib.sha256()
    for fp in sorted(path.rglob("*.parquet")):
        stat = fp.stat()
        h.update(f"{fp.relative_to(path)}::{stat.st_size}".encode())
    return h.hexdigest()[:12]

# -----------------------------------------------------------------------------
# Artefact writers
# -----------------------------------------------------------------------------

def save_artifacts(
    counts_sorted: np.ndarray,
    cdf: np.ndarray,
    out_dir: Path,
    config_dir: Path,
    overrides: dict[str, int | None],
    exams_root: Path,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw Parquet -----------------------------------------------------------
    tbl = pa.Table.from_pydict({"visit_cnt": counts_sorted, "cdf": cdf})
    pq.write_table(tbl, out_dir / "visit_count_raw.parquet")

    # 2. PNG plot --------------------------------------------------------------
    import matplotlib.pyplot as plt  # local import to keep headless CI happy

    plt.figure()
    plt.step(counts_sorted, cdf, where="post")
    plt.xlabel("Visits per person (n_i)")
    plt.ylabel("Empirical CDF F(k)")
    plt.xscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "visit_count_cdf.png", dpi=300)
    plt.close()

    # 3. Threshold YAML --------------------------------------------------------
    auto_K = counts_sorted[np.searchsorted(cdf, 0.80)]
    cold_start_K = overrides["cold_start_K"] or int(auto_K)
    sparse_max = overrides["sparse_max"] or int(counts_sorted[np.searchsorted(cdf, 0.20)])
    dense_min = overrides["dense_min"] or int(counts_sorted[np.searchsorted(cdf, 0.85)])

    cold_start_pct = float(cdf[np.searchsorted(counts_sorted, cold_start_K)])

    payload = {
        "cold_start_K": int(cold_start_K),
        "sparse_max": int(sparse_max),
        "dense_min": int(dense_min),
        "cold_start_pct": round(cold_start_pct, 4),
        "id_col": "person_id",
        "seed": seed,
        "exams_root_sha": sha_folder(exams_root),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    with open(config_dir / "visit_thresholds.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    # 4. Log quick summary to stdout ------------------------------------------
    print(json.dumps(payload, indent=2))

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    counts_sorted = compute_visit_counts(args.exams_root, args.sample_frac, args.seed)
    cdf = calc_cdf(counts_sorted)

    overrides = {
        "cold_start_K": args.cold_start_K,
        "sparse_max": args.sparse_max,
        "dense_min": args.dense_min,
    }

    save_artifacts(
        counts_sorted,
        cdf,
        args.out_dir,
        args.config_dir,
        overrides,
        args.exams_root,
        args.seed,
    )


if __name__ == "__main__":
    main()

"""Cluster incident-anchored SSL representations per anchor."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster incident-anchored SSL representations per anchor."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to clustering config YAML.",
    )
    parser.add_argument(
        "--only-anchor",
        type=str,
        default=None,
        help="Optional anchor_code or anchor name to process.",
    )
    parser.add_argument(
        "--only-embedding",
        type=str,
        default=None,
        help="Optional embedding column to process.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: expected dict at root, got {type(cfg)}")
    return cfg


def _to_numpy_embeddings(arr: pa.Array) -> np.ndarray:
    """
    Convert an Arrow embedding column to a dense float32 NumPy array.

    Note: pq.read_table returns a ChunkedArray; we normalize to a single chunk first.
    """
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    if pa.types.is_fixed_size_list(arr.type):
        if arr.null_count:
            raise ValueError("Embedding column contains nulls")
        list_size = arr.type.list_size
        flat = arr.values.to_numpy(zero_copy_only=False)
        flat = np.asarray(flat, dtype=np.float32)
        return flat.reshape((len(arr), list_size))

    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        if arr.null_count:
            raise ValueError("Embedding column contains nulls")
        py = arr.to_pylist()
        if not py:
            return np.empty((0, 0), dtype=np.float32)
        list_size = len(py[0])
        if any(len(x) != list_size for x in py):
            raise ValueError("Embedding column has inconsistent list lengths")
        return np.asarray(py, dtype=np.float32)

    raise ValueError(f"Unsupported embedding column type: {arr.type}")


def _select_columns(path: Path, columns: List[str]) -> pa.Table:
    schema = pq.read_schema(path)
    available = set(schema.names)
    cols = [c for c in columns if c in available]
    return pq.read_table(path, columns=cols)


def _merge_tables(case_table: pa.Table, control_table: pa.Table) -> pa.Table:
    # PyArrow >= 16: `promote` is superseded by `promote_options`.
    return pa.concat_tables([case_table, control_table], promote_options="default")


def _ensure_unique_ids(table: pa.Table, id_col: str) -> None:
    if id_col not in table.column_names:
        raise ValueError(f"Missing id column: {id_col}")
    ids = table[id_col].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids found in {id_col}")


def _ensure_anchor_code(table: pa.Table, expected_code: str) -> None:
    if "anchor_code" not in table.column_names:
        raise ValueError("Missing anchor_code column for sanity check")
    codes = set(table["anchor_code"].to_pylist())
    if len(codes) != 1 or expected_code not in codes:
        raise ValueError(f"Anchor code mismatch: expected {expected_code}, got {codes}")


def _write_parquet(table: pa.Table, path: Path, compression: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)


def _cluster_summary(
    labels: np.ndarray,
    cohort: List[Any] | None,
    age_at_index: List[Any] | None,
    gender: List[Any] | None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    unique, counts = np.unique(labels, return_counts=True)
    summary["cluster_sizes"] = {str(k): int(v) for k, v in zip(unique, counts)}

    if cohort is not None:
        cohort_arr = np.array(cohort)
        case_mask = cohort_arr == "CASE"
        summary["case_rate_overall"] = float(case_mask.mean()) if len(cohort_arr) else 0.0
        summary["case_rate_by_cluster"] = {}
        for k in unique:
            k_mask = labels == k
            if k_mask.any():
                summary["case_rate_by_cluster"][str(k)] = float(case_mask[k_mask].mean())

    if age_at_index is not None:
        ages = np.array(age_at_index, dtype=float)
        summary["age_at_index_mean_by_cluster"] = {}
        for k in unique:
            k_mask = labels == k
            if k_mask.any():
                summary["age_at_index_mean_by_cluster"][str(k)] = float(np.nanmean(ages[k_mask]))

    if gender is not None:
        gender_arr = np.array(gender, dtype=str)
        summary["gender_ratio_by_cluster"] = {}
        for k in unique:
            k_mask = labels == k
            if k_mask.any():
                vals, cnts = np.unique(gender_arr[k_mask], return_counts=True)
                total = cnts.sum()
                summary["gender_ratio_by_cluster"][str(k)] = {
                    str(v): float(c / total) for v, c in zip(vals, cnts)
                }

    return summary


def _write_summary(path: Path, payload: Dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _build_covariates(
    age_at_index: List[Any],
    gender: List[Any],
    center_age: bool,
    drop_missing: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    age = np.array(age_at_index, dtype=float)
    gender_arr = np.array(gender, dtype=str)
    missing_age = np.isnan(age)
    missing_gender = ~np.isin(gender_arr, ["M", "F"])

    keep = ~(missing_age | missing_gender)
    if not keep.all() and not drop_missing:
        raise ValueError("Missing age/gender values present but drop_missing is false")

    age = age[keep]
    gender_arr = gender_arr[keep]
    is_female = (gender_arr == "F").astype(float)
    if center_age:
        age = age - np.mean(age)

    X = np.column_stack([np.ones_like(age), age, is_female])
    return X, keep


def _residualize_embeddings(
    embeddings: np.ndarray,
    age_at_index: List[Any],
    gender: List[Any],
    alpha: float,
    center_age: bool,
    drop_missing: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    X, keep = _build_covariates(age_at_index, gender, center_age, drop_missing)
    E = embeddings[keep]
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, E)
    resid = E - model.predict(X)
    return resid, keep


def main() -> int:
    args = parse_args()
    setup_logging()

    cfg = load_config(Path(args.config))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    np.random.seed(seed)

    data_cfg = cfg["data"]
    anchors = data_cfg["anchors"]
    if args.only_anchor:
        anchors = [
            a for a in anchors
            if a.get("anchor_code") == args.only_anchor or a.get("name") == args.only_anchor
        ]
        if not anchors:
            raise ValueError(f"No anchor matched: {args.only_anchor}")

    embedding_cols = data_cfg.get("embedding_cols", [])
    if args.only_embedding:
        if args.only_embedding not in embedding_cols:
            raise ValueError(f"Unknown embedding column: {args.only_embedding}")
        embedding_cols = [args.only_embedding]

    meta_cols = data_cfg.get("meta_cols", [])
    id_col = data_cfg.get("id_col", "episode_id")

    preprocess_cfg = cfg["preprocess"]
    pca_cfg = preprocess_cfg.get("pca", {})
    pca_enabled = bool(pca_cfg.get("enabled", True))
    n_components = int(pca_cfg.get("n_components", 100))
    whiten = bool(pca_cfg.get("whiten", False))

    clustering_cfg = cfg["clustering"]
    k_values = clustering_cfg.get("k_values", [10])
    n_init = int(clustering_cfg.get("n_init", 20))
    max_iter = int(clustering_cfg.get("max_iter", 300))

    output_cfg = cfg["output"]
    output_dir_raw = Path(output_cfg.get("output_dir_raw", "outputs/clustering/raw"))
    output_dir_resid = Path(output_cfg.get("output_dir_resid", "outputs/clustering/resid"))
    compression = output_cfg.get("compression", "zstd")
    overwrite = bool(output_cfg.get("overwrite", False))
    assignments_template = output_cfg.get("assignments_filename_template", "{anchor_code}__{embedding_col}__k{k}.parquet")
    summary_template = output_cfg.get("summary_filename_template", "{anchor_code}__summary.json")
    pca2_template = output_cfg.get("pca2_filename_template", "{anchor_code}__{embedding_col}__pca2.parquet")
    export_pca2 = bool(output_cfg.get("export_pca2", False))
    log_dir = Path(output_cfg.get("log_dir", output_dir_raw))

    sanity_cfg = cfg.get("sanity_checks", {})
    require_unique_id = bool(sanity_cfg.get("require_unique_id", True))
    require_single_anchor_code = bool(sanity_cfg.get("require_single_anchor_code", True))
    report_case_rate = bool(sanity_cfg.get("report_case_rate", True))

    audit_entries: List[Dict[str, Any]] = []

    resid_cfg = preprocess_cfg.get("residualize", {})
    resid_enabled = bool(resid_cfg.get("enabled", False))
    resid_alpha = float(resid_cfg.get("alpha", 1.0))
    resid_center_age = bool(resid_cfg.get("center_age", True))
    resid_drop_missing = bool(resid_cfg.get("drop_missing", True))

    variants = [("raw", output_dir_raw, False)]
    if resid_enabled:
        variants.append(("resid", output_dir_resid, True))

    for anchor in anchors:
        anchor_code = anchor.get("anchor_code")
        case_path = Path(anchor["case_repr_path"])
        control_path = Path(anchor["control_repr_path"])

        base_cols = list(set(meta_cols + embedding_cols))
        case_table = _select_columns(case_path, base_cols)
        control_table = _select_columns(control_path, base_cols)
        table = _merge_tables(case_table, control_table)

        if require_unique_id:
            _ensure_unique_ids(table, id_col)
        if require_single_anchor_code:
            _ensure_anchor_code(table, anchor_code)

        available_meta_cols = [c for c in meta_cols if c in table.column_names]
        cohort = table["cohort"].to_pylist() if "cohort" in table.column_names else None
        age_at_index = table["age_at_index"].to_pylist() if "age_at_index" in table.column_names else None
        gender = table["Gender"].to_pylist() if "Gender" in table.column_names else None

        for variant_name, variant_dir, do_resid in variants:
            summary_payload: Dict[str, Any] = {
                "anchor_code": anchor_code,
                "variant": variant_name,
                "inputs": {
                    "case_repr_path": str(case_path),
                    "control_repr_path": str(control_path),
                },
                "rows": table.num_rows,
                "residualize": {
                    "enabled": do_resid,
                    "alpha": resid_alpha,
                    "center_age": resid_center_age,
                    "drop_missing": resid_drop_missing,
                },
            }

            for emb_col in embedding_cols:
                if emb_col not in table.column_names:
                    raise ValueError(f"Missing embedding column: {emb_col}")

                embeddings = _to_numpy_embeddings(table[emb_col])
                mask = np.ones(len(embeddings), dtype=bool)
                if do_resid:
                    if age_at_index is None or gender is None:
                        raise ValueError("Residualize requires age_at_index and Gender columns")
                    embeddings, mask = _residualize_embeddings(
                        embeddings,
                        age_at_index,
                        gender,
                        resid_alpha,
                        resid_center_age,
                        resid_drop_missing,
                    )
                    summary_payload["rows"] = int(mask.sum())

                if preprocess_cfg.get("standardize", True):
                    scaler = StandardScaler()
                    embeddings = scaler.fit_transform(embeddings)
                else:
                    scaler = None

                if pca_enabled:
                    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
                    reduced = pca.fit_transform(embeddings)
                else:
                    pca = None
                    reduced = embeddings

                if export_pca2:
                    if pca_enabled and reduced.shape[1] >= 2:
                        pca2 = reduced[:, :2]
                    else:
                        pca2 = embeddings[:, :2]
                    pca2_table = pa.table(
                        {
                            id_col: table[id_col].filter(pa.array(mask)),
                            "pc1": pa.array(pca2[:, 0], type=pa.float32()),
                            "pc2": pa.array(pca2[:, 1], type=pa.float32()),
                            **{c: table[c].filter(pa.array(mask)) for c in available_meta_cols if c != id_col},
                        }
                    )
                    pca2_path = variant_dir / pca2_template.format(
                        anchor_code=anchor_code,
                        embedding_col=emb_col,
                    )
                    _write_parquet(pca2_table, pca2_path, compression, overwrite)

                if pca_enabled:
                    summary_payload.setdefault("pca", {})[emb_col] = {
                        "n_components": int(n_components),
                        "explained_variance_ratio": [
                            float(v) for v in pca.explained_variance_ratio_[:10]
                        ],
                        "explained_variance_total": float(
                            np.sum(pca.explained_variance_ratio_)
                        ),
                    }

                summary_payload.setdefault("clustering", {})[emb_col] = {}
                for k in k_values:
                    kmeans = KMeans(
                        n_clusters=int(k),
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=seed,
                    )
                    labels = kmeans.fit_predict(reduced)

                    cluster_table = pa.table(
                        {
                            id_col: table[id_col].filter(pa.array(mask)),
                            "cluster_id": pa.array(labels, type=pa.int64()),
                            **{c: table[c].filter(pa.array(mask)) for c in available_meta_cols if c != id_col},
                        }
                    )
                    out_path = variant_dir / assignments_template.format(
                        anchor_code=anchor_code,
                        embedding_col=emb_col,
                        k=k,
                    )
                    _write_parquet(cluster_table, out_path, compression, overwrite)

                    summary = {
                        "k": int(k),
                        "inertia": float(kmeans.inertia_),
                    }
                    if report_case_rate or age_at_index is not None or gender is not None:
                        if do_resid:
                            cohort_masked = [c for c, keep in zip(cohort, mask) if keep] if cohort is not None else None
                            age_masked = [a for a, keep in zip(age_at_index, mask) if keep] if age_at_index is not None else None
                            gender_masked = [g for g, keep in zip(gender, mask) if keep] if gender is not None else None
                            summary.update(_cluster_summary(labels, cohort_masked, age_masked, gender_masked))
                        else:
                            summary.update(_cluster_summary(labels, cohort, age_at_index, gender))

                    summary_payload["clustering"][emb_col][str(k)] = summary
                    logger.info(
                        "Anchor %s | %s | %s | k=%s: wrote %s rows to %s",
                        anchor_code,
                        emb_col,
                        variant_name,
                        k,
                        int(mask.sum()),
                        out_path,
                    )

            summary_path = variant_dir / summary_template.format(anchor_code=anchor_code)
            _write_summary(summary_path, summary_payload, overwrite)

            audit_entries.append(
                {
                    "anchor_code": anchor_code,
                    "variant": variant_name,
                    "rows": int(mask.sum()) if do_resid else table.num_rows,
                    "summary_path": str(summary_path),
                }
            )

    audit_payload = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "anchors": audit_entries,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_path = log_dir / f"audit_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with audit_path.open("w") as f:
        json.dump(audit_payload, f, ensure_ascii=True, indent=2)
    logger.info("Wrote audit log to %s", audit_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

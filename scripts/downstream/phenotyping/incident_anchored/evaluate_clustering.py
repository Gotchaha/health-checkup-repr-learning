"""Evaluate and visualize incident-anchored clustering results."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate incident-anchored clustering outputs."
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
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["raw", "resid", "both"],
        help="Evaluate raw, resid, or both variants.",
    )
    parser.add_argument(
        "--only-k",
        type=int,
        default=None,
        help="Optional single k value to evaluate.",
    )
    parser.add_argument(
        "--silhouette-subsample",
        type=int,
        default=5000,
        help="Subsample size for silhouette score.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
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
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        flat = arr.values.to_numpy(zero_copy_only=False)
        flat = np.asarray(flat, dtype=np.float32)
        return flat.reshape((len(arr), arr.type.list_size))
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
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
    return pa.concat_tables([case_table, control_table], promote_options="default")


def _ensure_unique_ids(table: pa.Table, id_col: str) -> None:
    ids = table[id_col].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids found in {id_col}")


def _plot_scatter(
    pc: np.ndarray,
    color_values: np.ndarray,
    title: str,
    path: Path,
    cmap: str = "viridis",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(pc[:, 0], pc[:, 1], c=color_values, s=6, alpha=0.6, cmap=cmap)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_categorical(
    pc: np.ndarray,
    labels: List[Any],
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels_arr = np.array(labels)
    uniq = np.unique(labels_arr)
    plt.figure(figsize=(6, 5))
    for val in uniq:
        mask = labels_arr == val
        plt.scatter(pc[mask, 0], pc[mask, 1], s=6, alpha=0.6, label=str(val))
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _cluster_stats(
    labels: np.ndarray,
    cohort: List[Any] | None,
    age_at_index: List[Any] | None,
    gender: List[Any] | None,
) -> Dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    stats: Dict[str, Any] = {
        "cluster_sizes": {str(k): int(v) for k, v in zip(unique, counts)},
    }

    if cohort is not None:
        cohort_arr = np.array(cohort)
        case_mask = cohort_arr == "CASE"
        stats["case_rate_overall"] = float(case_mask.mean()) if len(cohort_arr) else 0.0
        stats["case_rate_by_cluster"] = {}
        for k in unique:
            k_mask = labels == k
            if k_mask.any():
                stats["case_rate_by_cluster"][str(k)] = float(case_mask[k_mask].mean())

    if age_at_index is not None:
        ages = np.array(age_at_index, dtype=float)
        stats["age_at_index_mean_by_cluster"] = {}
        stats["age_at_index_quantiles_by_cluster"] = {}
        for k in unique:
            k_mask = labels == k
            if k_mask.any():
                vals = ages[k_mask]
                stats["age_at_index_mean_by_cluster"][str(k)] = float(np.nanmean(vals))
                stats["age_at_index_quantiles_by_cluster"][str(k)] = {
                    "p25": float(np.nanquantile(vals, 0.25)),
                    "p50": float(np.nanquantile(vals, 0.50)),
                    "p75": float(np.nanquantile(vals, 0.75)),
                }

    if gender is not None:
        gender_arr = np.array(gender, dtype=str)
        stats["gender_ratio_by_cluster"] = {}
        for k in unique:
            k_mask = labels == k
            if k_mask.any():
                vals, cnts = np.unique(gender_arr[k_mask], return_counts=True)
                total = cnts.sum()
                stats["gender_ratio_by_cluster"][str(k)] = {
                    str(v): float(c / total) for v, c in zip(vals, cnts)
                }

    return stats


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
    seed = args.seed if args.seed is not None else int(cfg.get("experiment", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

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

    id_col = data_cfg.get("id_col", "episode_id")
    meta_cols = data_cfg.get("meta_cols", [])

    preprocess_cfg = cfg["preprocess"]
    standardize = bool(preprocess_cfg.get("standardize", True))
    pca_cfg = preprocess_cfg.get("pca", {})
    n_components = int(pca_cfg.get("n_components", 100))
    whiten = bool(pca_cfg.get("whiten", False))

    clustering_cfg = cfg["clustering"]
    k_values = clustering_cfg.get("k_values", [10])
    if args.only_k is not None:
        if args.only_k not in k_values:
            raise ValueError(f"k={args.only_k} not in config k_values")
        k_values = [args.only_k]

    output_cfg = cfg["output"]
    output_dir_raw = Path(output_cfg.get("output_dir_raw", "outputs/clustering/raw"))
    output_dir_resid = Path(output_cfg.get("output_dir_resid", "outputs/clustering/resid"))
    log_dir = Path(output_cfg.get("log_dir", output_dir_raw))
    pca2_template = output_cfg.get("pca2_filename_template", "{anchor_code}__{embedding_col}__pca2.parquet")
    assignments_template = output_cfg.get("assignments_filename_template", "{anchor_code}__{embedding_col}__k{k}.parquet")

    resid_cfg = preprocess_cfg.get("residualize", {})
    resid_enabled = bool(resid_cfg.get("enabled", False))
    resid_alpha = float(resid_cfg.get("alpha", 1.0))
    resid_center_age = bool(resid_cfg.get("center_age", True))
    resid_drop_missing = bool(resid_cfg.get("drop_missing", True))

    variants = [("raw", output_dir_raw, False)]
    if resid_enabled:
        variants.append(("resid", output_dir_resid, True))
    if args.variant and args.variant != "both":
        variants = [v for v in variants if v[0] == args.variant]
        if not variants:
            raise ValueError(f"Variant '{args.variant}' not available (residualize disabled)")

    audit_entries: List[Dict[str, Any]] = []

    for anchor in anchors:
        anchor_code = anchor.get("anchor_code")
        case_path = Path(anchor["case_repr_path"])
        control_path = Path(anchor["control_repr_path"])

        base_cols = list(set(meta_cols + embedding_cols))
        case_table = _select_columns(case_path, base_cols)
        control_table = _select_columns(control_path, base_cols)
        table = _merge_tables(case_table, control_table)

        if id_col not in table.column_names:
            raise ValueError(f"Missing id column: {id_col}")
        _ensure_unique_ids(table, id_col)

        cohort = table["cohort"].to_pylist() if "cohort" in table.column_names else None
        age_at_index = table["age_at_index"].to_pylist() if "age_at_index" in table.column_names else None
        gender = table["Gender"].to_pylist() if "Gender" in table.column_names else None

        for variant_name, variant_dir, do_resid in variants:
            figures_dir = variant_dir / "figures"
            metrics_dir = variant_dir / "analysis"

            anchor_metrics: Dict[str, Any] = {
                "anchor_code": anchor_code,
                "variant": variant_name,
                "rows": table.num_rows,
                "embeddings": {},
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

                scaler = StandardScaler() if standardize else None
                if scaler is not None:
                    embeddings = scaler.fit_transform(embeddings)

                pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
                reduced = pca.fit_transform(embeddings)

                anchor_metrics["embeddings"][emb_col] = {
                    "pca": {
                        "n_components": int(n_components),
                        "explained_variance_ratio": [
                            float(v) for v in pca.explained_variance_ratio_[:10]
                        ],
                        "explained_variance_total": float(
                            np.sum(pca.explained_variance_ratio_)
                        ),
                    },
                    "k_metrics": {},
                }

                pca2_path = variant_dir / pca2_template.format(
                    anchor_code=anchor_code,
                    embedding_col=emb_col,
                )
                if not pca2_path.exists():
                    pca2 = reduced[:, :2]
                    pca2_table = pa.table(
                        {
                            id_col: table[id_col].filter(pa.array(mask)),
                            "pc1": pa.array(pca2[:, 0], type=pa.float32()),
                            "pc2": pa.array(pca2[:, 1], type=pa.float32()),
                            **{c: table[c].filter(pa.array(mask)) for c in meta_cols if c in table.column_names and c != id_col},
                        }
                    )
                    pca2_path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pca2_table, pca2_path, compression=output_cfg.get("compression", "zstd"))
                else:
                    pca2_table = pq.read_table(pca2_path).combine_chunks()

                pc1_all = np.asarray(pca2_table["pc1"].to_pylist(), dtype=np.float32)
                pc2_all = np.asarray(pca2_table["pc2"].to_pylist(), dtype=np.float32)
                pc_all = np.column_stack([pc1_all, pc2_all])

                ids_all = table[id_col].to_pylist()
                if do_resid:
                    ids_kept = [val for val, keep in zip(ids_all, mask) if keep]
                else:
                    ids_kept = ids_all
                id_to_idx = {val: idx for idx, val in enumerate(ids_kept)}

                for k in k_values:
                    assignments_path = variant_dir / assignments_template.format(
                        anchor_code=anchor_code,
                        embedding_col=emb_col,
                        k=k,
                    )
                    assignments = _select_columns(assignments_path, [id_col, "cluster_id", "cohort", "age_at_index", "Gender"])
                    assignments = assignments.combine_chunks()
                    ids = assignments[id_col].to_pylist()
                    if len(ids) != len(set(ids)):
                        raise ValueError(f"Duplicate ids in assignments: {assignments_path}")

                    indices = [id_to_idx[i] for i in ids]
                    labels = np.array(assignments["cluster_id"].to_pylist())
                    reduced_subset = reduced[indices]

                    metrics: Dict[str, Any] = {}
                    metrics["calinski_harabasz"] = float(calinski_harabasz_score(reduced_subset, labels))
                    metrics["davies_bouldin"] = float(davies_bouldin_score(reduced_subset, labels))

                    n = len(labels)
                    if n > 1:
                        sample_n = min(int(args.silhouette_subsample), n)
                        if sample_n < n:
                            sample_idx = rng.choice(n, size=sample_n, replace=False)
                        else:
                            sample_idx = np.arange(n)
                        metrics["silhouette"] = float(
                            silhouette_score(reduced_subset[sample_idx], labels[sample_idx])
                        )

                    stats = _cluster_stats(
                        labels,
                        assignments["cohort"].to_pylist() if "cohort" in assignments.column_names else cohort,
                        assignments["age_at_index"].to_pylist() if "age_at_index" in assignments.column_names else age_at_index,
                        assignments["Gender"].to_pylist() if "Gender" in assignments.column_names else gender,
                    )
                    metrics.update(stats)
                    anchor_metrics["embeddings"][emb_col]["k_metrics"][str(k)] = metrics

                    pc = pc_all[indices]

                    fig_base = figures_dir / anchor_code / emb_col
                    _plot_categorical(
                        pc,
                        labels,
                        f"{anchor_code} {emb_col} {variant_name} k={k} (clusters)",
                        fig_base / f"k{k}_clusters.png",
                    )
                    if "cohort" in assignments.column_names:
                        _plot_categorical(
                            pc,
                            assignments["cohort"].to_pylist(),
                            f"{anchor_code} {emb_col} {variant_name} k={k} (cohort)",
                            fig_base / f"k{k}_cohort.png",
                        )
                    if "age_at_index" in assignments.column_names:
                        age_vals = np.array(assignments["age_at_index"].to_pylist(), dtype=float)
                        _plot_scatter(
                            pc,
                            age_vals,
                            f"{anchor_code} {emb_col} {variant_name} k={k} (age)",
                            fig_base / f"k{k}_age.png",
                        )

                    logger.info(
                        "Anchor %s | %s | %s | k=%s: CH=%.3f DB=%.3f sil=%.3f",
                        anchor_code,
                        emb_col,
                        variant_name,
                        k,
                        metrics["calinski_harabasz"],
                        metrics["davies_bouldin"],
                        metrics.get("silhouette", float("nan")),
                    )

            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metrics_dir / f"metrics_{anchor_code}.json"
            with metrics_path.open("w") as f:
                json.dump(anchor_metrics, f, ensure_ascii=True, indent=2)
            audit_entries.append({"anchor_code": anchor_code, "variant": variant_name, "metrics_path": str(metrics_path)})

    log_dir.mkdir(parents=True, exist_ok=True)
    audit_path = log_dir / f"audit_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with audit_path.open("w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": args.config,
                "anchors": audit_entries,
                "silhouette_subsample": int(args.silhouette_subsample),
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    logger.info("Wrote audit log to %s", audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

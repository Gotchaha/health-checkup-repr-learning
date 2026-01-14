#!/usr/bin/env python
"""
Compute heuristic baselines (Last-Value and Mean-of-Past) for the lab test task.

Given baseline_pairs.parquet, the script evaluates two naïve predictors on the
validation and test splits:
    - Last-Value (LV): predicts Y_k(t+1) using X_k(t)
    - Mean-of-Past (MoP): predicts Y_k(t+1) using the per-person historical mean
Both baselines report regression and binary metrics aligned with downstream eval.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run heuristic baselines for lab tests.")
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to data/downstream/lab_test/v1/baseline_pairs.parquet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config/downstream/lab_test_task_config_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/downstream/v1/benchmark"),
        help="Directory where run artifacts will be stored",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to timestamp if not provided",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    """Compute SHA256 for a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_config(config_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Load label order and task types from config."""
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    try:
        label_order = config["datamodule"]["label_processing"]["label_order"]
        task_type_cfg = config["model"]["task_types"]
    except KeyError as exc:
        raise KeyError("Config missing label_order or task_types") from exc

    task_type_map = {}
    for task in task_type_cfg.get("regression", []):
        task_type_map[task] = "regression"
    for task in task_type_cfg.get("binary", []):
        task_type_map[task] = "binary"

    return label_order, task_type_map


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    metrics = {}
    diff = y_pred - y_true
    metrics["mae"] = float(np.mean(np.abs(diff)))
    metrics["rmse"] = float(np.sqrt(np.mean(diff**2)))

    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        metrics["r2"] = float("nan")
    else:
        metrics["r2"] = float(1.0 - ss_res / ss_tot)

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics["pearson_r"] = float("nan")

    metrics["n"] = int(len(y_true))
    return metrics


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute binary metrics with probability scores."""
    metrics = {"n": int(len(y_true))}

    if len(np.unique(y_true)) < 2:
        metrics.update(
            {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
             "precision": float("nan"), "recall": float("nan")}
        )
        return metrics

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["auprc"] = float("nan")

    y_pred = (y_score >= 0.5).astype(int)
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return metrics


def aggregate_summary(
    task_results: Dict[str, Dict[str, Dict[str, float]]],
    regression_tasks: List[str],
    binary_tasks: List[str],
) -> Dict[str, float]:
    """Aggregate mean MAE and macro AUROC."""
    reg_maes = [
        task_results["regression"][task]["mae"]
        for task in regression_tasks
        if task in task_results["regression"]
    ]
    bin_aurocs = [
        task_results["binary"][task]["auroc"]
        for task in binary_tasks
        if task in task_results["binary"]
    ]

    summary = {
        "mean_mae": float(np.nanmean(reg_maes)) if reg_maes else float("nan"),
        "macro_auroc": float(np.nanmean(bin_aurocs)) if bin_aurocs else float("nan"),
    }
    return summary


def ensure_columns(df: pd.DataFrame, label_order: List[str]) -> None:
    """Ensure expected pair columns exist."""
    required_cols = [
        "person_id",
        "pair_index_in_person",
        "is_train_pair",
        "is_val_pair",
        "is_testf_pair",
        "use_for_test",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"baseline_pairs missing columns: {missing}")

    missing_y = [f"Y_{col}" for col in label_order if f"Y_{col}" not in df.columns]
    missing_x = [f"X_{col}" for col in label_order if f"X_{col}" not in df.columns]
    if missing_y or missing_x:
        raise ValueError(f"Missing prediction columns: X -> {missing_x}, Y -> {missing_y}")


def compute_mop_columns(df: pd.DataFrame, label_order: List[str]) -> pd.DataFrame:
    """Compute Mean-of-Past columns per task."""
    df = df.sort_values(["person_id", "pair_index_in_person"]).reset_index(drop=True)
    for task in label_order:
        x_col = f"X_{task}"
        mop_col = f"MoP_{task}"
        df[mop_col] = (
            df.groupby("person_id", sort=False)[x_col]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def run_baseline(
    df_split: pd.DataFrame,
    regression_tasks: List[str],
    binary_tasks: List[str],
    prefix: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute baseline metrics for a split."""
    results = {"regression": {}, "binary": {}}

    for task in regression_tasks:
        y_true = df_split[f"Y_{task}"].to_numpy()
        y_pred = df_split[f"{prefix}_{task}"].to_numpy()
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() == 0:
            continue
        metrics = evaluate_regression(y_true[mask], y_pred[mask])
        results["regression"][task] = metrics

    for task in binary_tasks:
        y_true = df_split[f"Y_{task}"].to_numpy()
        y_pred = df_split[f"{prefix}_{task}"].to_numpy()
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() == 0:
            continue
        y_true_bin = (y_true[mask] > 0.5).astype(int)
        metrics = evaluate_binary(y_true_bin, y_pred[mask])
        results["binary"][task] = metrics

    return results


def flatten_results(
    baseline_name: str,
    split_name: str,
    task_type: str,
    task_name: str,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    """Prepare row for CSV summary."""
    row = {
        "baseline": baseline_name,
        "split": split_name,
        "task_type": task_type,
        "task": task_name,
    }
    row.update(metrics)
    return row


def main() -> None:
    """Main entry point."""
    args = parse_args()

    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"heuristics_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )
    logging.info("Starting heuristic baselines run: %s", run_name)

    label_order, task_type_map = load_config(args.config)
    regression_tasks = [task for task in label_order if task_type_map.get(task) == "regression"]
    binary_tasks = [task for task in label_order if task_type_map.get(task) == "binary"]
    logging.info("Regression tasks: %d | Binary tasks: %d", len(regression_tasks), len(binary_tasks))

    logging.info("Reading baseline pairs from %s", args.pairs)
    pairs = pd.read_parquet(args.pairs)
    ensure_columns(pairs, label_order)

    logging.info("Computing Mean-of-Past columns")
    pairs = compute_mop_columns(pairs, label_order)

    df_train = pairs[pairs["is_train_pair"]].copy()
    df_val = pairs[pairs["is_val_pair"]].copy()
    df_test = pairs[pairs["use_for_test"]].copy()
    logging.info(
        "Split sizes | train=%d | val=%d | test=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    baselines = {}
    csv_rows: List[Dict[str, object]] = []
    for baseline_name, prefix in [("LV", "X"), ("MoP", "MoP")]:
        logging.info("Evaluating %s baseline on validation split", baseline_name)
        val_results = run_baseline(df_val, regression_tasks, binary_tasks, prefix)
        logging.info("Evaluating %s baseline on test split", baseline_name)
        test_results = run_baseline(df_test, regression_tasks, binary_tasks, prefix)

        baseline_entry = {
            "val": {
                "regression": val_results["regression"],
                "binary": val_results["binary"],
                "summary": aggregate_summary(val_results, regression_tasks, binary_tasks),
            },
            "test": {
                "regression": test_results["regression"],
                "binary": test_results["binary"],
                "summary": aggregate_summary(test_results, regression_tasks, binary_tasks),
            },
        }
        baselines[baseline_name] = baseline_entry

        for split_name, split_results in [("val", val_results), ("test", test_results)]:
            for task, metrics in split_results["regression"].items():
                csv_rows.append(flatten_results(baseline_name, split_name, "regression", task, metrics))
            for task, metrics in split_results["binary"].items():
                csv_rows.append(flatten_results(baseline_name, split_name, "binary", task, metrics))

    metrics_json = {
        "config": {
            "pairs_path": str(args.pairs),
            "config_path": str(args.config),
            "label_order": label_order,
            "regression_tasks": regression_tasks,
            "binary_tasks": binary_tasks,
            "n_train_pairs": int(len(df_train)),
            "n_val_pairs": int(len(df_val)),
            "n_test_pairs": int(len(df_test)),
        },
        "baselines": baselines,
    }

    json_path = run_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    logging.info("Saved metrics JSON to %s", json_path)

    csv_path = run_dir / "metrics_summary.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logging.info("Saved metrics summary CSV to %s", csv_path)

    metadata = {
        "run_name": run_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pairs_path": str(args.pairs),
        "pairs_sha256": compute_sha256(args.pairs),
        "config_path": str(args.config),
        "config_sha256": compute_sha256(args.config),
        "run_dir": str(run_dir),
    }
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Recorded run metadata at %s", metadata_path)

    logging.info("Baseline heuristics completed successfully")


if __name__ == "__main__":
    main()

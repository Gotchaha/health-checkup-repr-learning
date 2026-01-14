"""Train an FT-Transformer baseline for the Grade5 downstream task.

This script implements the protocol-C step from `tmp/grade5_benchmark/grade5_benchmark.py`:
- Assumes protocol A/B already produced split-wise artifacts under `<run_dir>/features_wide/`.
- Trains an FT-Transformer model using the fixed train/val/test splits.
- Writes predictions to `<run_dir>/preds/fttransformer/` in the same format as other baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import delu
from rtdl_revisiting_models import FTTransformer

# Make project root importable (scripts are invoked from project root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


NUM_CLASS = 5


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _safe_pkg_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _cat_cardinalities_from_feature_list(
    feature_list: dict[str, Any],
    cat_columns: list[str],
) -> list[int]:
    vocab = feature_list.get("cat_features", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("feature_list.json missing cat_features.vocab")

    cardinalities = []
    for col in cat_columns:
        if col not in vocab:
            raise ValueError(f"Column {col!r} missing from feature_list.json vocab")
        values = vocab[col]
        if not isinstance(values, list):
            raise ValueError(f"feature_list.json vocab[{col!r}] must be a list")
        cardinalities.append(int(len(values) + 2))  # missing=0, unknown=1, known>=2
    return cardinalities


def _cardinality_summary(cardinalities: list[int]) -> dict[str, float | int]:
    if not cardinalities:
        return {"n_cat_features": 0, "min": 0, "max": 0, "mean": 0.0}
    arr = np.asarray(cardinalities, dtype=np.int64)
    return {
        "n_cat_features": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


@dataclass(frozen=True)
class SplitArtifacts:
    x_cont: torch.Tensor
    x_cat: torch.Tensor | None
    y: torch.Tensor
    index_df: pd.DataFrame


def _load_split(
    run_dir: Path, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    feature_dir = run_dir / "features_wide"
    xnum_path = feature_dir / f"X_num_{split}.parquet"
    xcat_path = feature_dir / f"X_cat_{split}.parquet"
    y_path = feature_dir / f"y_{split}.npy"
    idx_path = feature_dir / f"index_{split}.parquet"

    for p in [xnum_path, xcat_path, y_path, idx_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    x_num_df = pd.read_parquet(xnum_path)
    x_cat_df = pd.read_parquet(xcat_path)
    y = np.load(y_path).astype(np.int64)
    index_df = pd.read_parquet(idx_path)

    if not (len(x_num_df) == len(x_cat_df) == len(index_df) == len(y)):
        raise ValueError(f"Split {split}: artifact lengths do not match")

    if "exam_id" not in x_num_df.columns or "exam_id" not in x_cat_df.columns:
        raise ValueError(f"Split {split}: missing exam_id in X_num/X_cat")

    if not (x_num_df["exam_id"].values == index_df["exam_id"].values).all():
        raise ValueError(f"Split {split}: X_num exam_id does not align with index")
    if not (x_cat_df["exam_id"].values == index_df["exam_id"].values).all():
        raise ValueError(f"Split {split}: X_cat exam_id does not align with index")

    return x_num_df, x_cat_df, y, index_df


def _prepare_tensors(
    x_num_df: pd.DataFrame,
    x_cat_df: pd.DataFrame,
    y: np.ndarray,
    index_df: pd.DataFrame,
    device: torch.device,
) -> SplitArtifacts:
    x_cont = torch.as_tensor(
        x_num_df.drop(columns=["exam_id"]).to_numpy(dtype=np.float32),
        device=device,
    )

    cat_columns = [c for c in x_cat_df.columns if c != "exam_id"]
    if cat_columns:
        x_cat = torch.as_tensor(
            x_cat_df[cat_columns].to_numpy(dtype=np.int64),
            device=device,
        )
    else:
        x_cat = None

    y_t = torch.as_tensor(y.astype(np.int64), device=device)
    return SplitArtifacts(x_cont=x_cont, x_cat=x_cat, y=y_t, index_df=index_df)


def _apply_model(model: FTTransformer, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["x_cont"], batch.get("x_cat"))


@torch.no_grad()
def _evaluate_val_loss(
    model: FTTransformer, data: dict[str, torch.Tensor], batch_size: int
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in delu.iter_batches(data, batch_size):
        logits = _apply_model(model, batch)
        loss = F.cross_entropy(logits, batch["y"], reduction="mean")
        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        raise ValueError("Empty validation set")
    return float(np.mean(losses))


@torch.no_grad()
def _predict_proba(
    model: FTTransformer, data: dict[str, torch.Tensor], batch_size: int
) -> np.ndarray:
    model.eval()
    out: list[torch.Tensor] = []
    for batch in delu.iter_batches(data, batch_size):
        logits = _apply_model(model, batch)
        out.append(logits)

    logits_all = torch.cat(out, dim=0)
    proba = torch.softmax(logits_all, dim=1)
    return proba.detach().cpu().numpy().astype(np.float32)


def _save_pred(
    run_dir: Path,
    model_name: str,
    split: str,
    exam_id: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_class: int,
) -> Path:
    out_dir = run_dir / "preds" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if y_proba.shape != (len(y_true), num_class):
        raise ValueError(f"Unexpected y_proba shape: {y_proba.shape}")

    df = pd.DataFrame(
        {
            "exam_id": exam_id.astype(str),
            "y_true": y_true.astype(np.int64),
            "y_pred": y_pred.astype(np.int64),
        }
    )
    for k in range(num_class):
        df[f"proba_{k}"] = y_proba[:, k].astype(np.float32)

    pred_path = out_dir / f"pred_{split}.parquet"
    df.to_parquet(pred_path, index=False)
    return pred_path


def _update_model_meta(run_dir: Path, model_name: str, meta: dict[str, Any]) -> Path:
    out_dir = run_dir / "preds" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "model_meta.json"

    if meta_path.exists():
        with meta_path.open("r") as f:
            prev = json.load(f)
    else:
        prev = {}
    prev.update(meta)

    with meta_path.open("w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False)
    return meta_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to a Grade5 benchmark run directory containing features_wide/.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=1_000_000_000)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    try:
        run_dir: Path = args.run_dir
        feature_dir = run_dir / "features_wide"
        if not feature_dir.exists():
            raise FileNotFoundError(f"Missing: {feature_dir}")

        run_meta_path = run_dir / "run_metadata.json"
        if not run_meta_path.exists():
            raise FileNotFoundError(f"Missing: {run_meta_path}")
        run_meta = _load_json(run_meta_path)
        seed = int(run_meta["seed"])

        delu.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        requested_device = torch.device(args.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available; falling back to CPU")
            device = torch.device("cpu")
        else:
            device = requested_device

        logging.info("run_dir=%s", run_dir)
        logging.info("seed=%d", seed)
        logging.info("device=%s", str(device))

        feature_list_path = feature_dir / "feature_list.json"
        if not feature_list_path.exists():
            raise FileNotFoundError(f"Missing: {feature_list_path}")
        feature_list = _load_json(feature_list_path)

        x_num_train_df, x_cat_train_df, y_train_np, idx_train = _load_split(run_dir, "train")
        x_num_val_df, x_cat_val_df, y_val_np, idx_val = _load_split(run_dir, "val")
        x_num_test_df, x_cat_test_df, y_test_np, idx_test = _load_split(run_dir, "test")

        if len(y_train_np) == 0:
            raise ValueError("Empty training set")
        if len(y_val_np) == 0:
            raise ValueError("Empty validation set")

        for split_name, y_arr in [("train", y_train_np), ("val", y_val_np), ("test", y_test_np)]:
            if y_arr.min() < 0 or y_arr.max() >= NUM_CLASS:
                raise ValueError(f"Split {split_name}: y values must be in [0, {NUM_CLASS - 1}]")

        cat_columns = [c for c in x_cat_train_df.columns if c != "exam_id"]
        cat_cardinalities = _cat_cardinalities_from_feature_list(feature_list, cat_columns)

        train = _prepare_tensors(x_num_train_df, x_cat_train_df, y_train_np, idx_train, device)
        val = _prepare_tensors(x_num_val_df, x_cat_val_df, y_val_np, idx_val, device)
        test = _prepare_tensors(x_num_test_df, x_cat_test_df, y_test_np, idx_test, device)

        n_cont_features = int(train.x_cont.shape[1])
        model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=NUM_CLASS,
            **FTTransformer.get_default_kwargs(),
        ).to(device)
        optimizer = model.make_default_optimizer()

        logging.info("n_cont_features=%d", n_cont_features)
        logging.info("n_cat_features=%d", len(cat_cardinalities))

        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(args.patience, mode="min")

        best_val_loss = math.inf
        best_epoch = -1
        best_state: dict[str, Any] | None = None

        timer.run()
        t0 = time.time()

        data_train = {"x_cont": train.x_cont, "x_cat": train.x_cat, "y": train.y}
        data_val = {"x_cont": val.x_cont, "x_cat": val.x_cat, "y": val.y}

        logging.info(
            "Starting training: batch_size=%d max_epochs=%d patience=%d",
            args.batch_size,
            args.max_epochs,
            args.patience,
        )

        for epoch in range(int(args.max_epochs)):
            model.train()
            for batch in delu.iter_batches(data_train, args.batch_size, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                logits = _apply_model(model, batch)
                loss = F.cross_entropy(logits, batch["y"], reduction="mean")
                loss.backward()
                optimizer.step()

            val_loss = _evaluate_val_loss(model, data_val, batch_size=args.eval_batch_size)
            logging.info("epoch=%d val_loss=%.6f time=%s", epoch, val_loss, str(timer))

            early_stopping.update(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                best_state = {
                    "epoch": best_epoch,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }

            if early_stopping.should_stop():
                logging.info("Early stopping triggered at epoch=%d", epoch)
                break

        train_time_sec = float(time.time() - t0)

        if best_state is None:
            raise RuntimeError("No best checkpoint captured")

        model.load_state_dict(best_state["model_state_dict"], strict=True)

        out_dir = run_dir / "preds" / "fttransformer"
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = out_dir / "model.pt"
        torch.save(
            {
                "model": "fttransformer",
                "seed": seed,
                "num_class": NUM_CLASS,
                "n_cont_features": n_cont_features,
                "cat_cardinalities": cat_cardinalities,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "state_dict": best_state["model_state_dict"],
            },
            ckpt_path,
        )

        val_proba = _predict_proba(
            model,
            {"x_cont": val.x_cont, "x_cat": val.x_cat, "y": val.y},
            batch_size=args.eval_batch_size,
        )
        val_pred = val_proba.argmax(axis=1).astype(np.int64)
        _save_pred(
            run_dir,
            model_name="fttransformer",
            split="val",
            exam_id=idx_val["exam_id"].values,
            y_true=y_val_np,
            y_pred=val_pred,
            y_proba=val_proba,
            num_class=NUM_CLASS,
        )

        test_proba = _predict_proba(
            model,
            {"x_cont": test.x_cont, "x_cat": test.x_cat, "y": test.y},
            batch_size=args.eval_batch_size,
        )
        test_pred = test_proba.argmax(axis=1).astype(np.int64)
        _save_pred(
            run_dir,
            model_name="fttransformer",
            split="test",
            exam_id=idx_test["exam_id"].values,
            y_true=y_test_np,
            y_pred=test_pred,
            y_proba=test_proba,
            num_class=NUM_CLASS,
        )

        meta = {
            "model": "fttransformer",
            "seed": seed,
            "device": str(device),
            "num_class": NUM_CLASS,
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "patience": int(args.patience),
            "max_epochs": int(args.max_epochs),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "train_time_sec": float(train_time_sec),
            "data": {
                "n_train": int(len(y_train_np)),
                "n_val": int(len(y_val_np)),
                "n_test": int(len(y_test_np)),
                "n_cont_features": int(n_cont_features),
                "cat_cardinalities_summary": _cardinality_summary(cat_cardinalities),
            },
            "artifacts": {
                "checkpoint": str(ckpt_path.relative_to(run_dir)),
                "pred_val": str((out_dir / "pred_val.parquet").relative_to(run_dir)),
                "pred_test": str((out_dir / "pred_test.parquet").relative_to(run_dir)),
            },
            "package_versions": {
                "numpy": _safe_pkg_version("numpy"),
                "pandas": _safe_pkg_version("pandas"),
                "pyarrow": _safe_pkg_version("pyarrow"),
                "torch": _safe_pkg_version("torch"),
                "delu": _safe_pkg_version("delu"),
                "rtdl_revisiting_models": _safe_pkg_version("rtdl-revisiting-models"),
            },
        }

        meta_path = _update_model_meta(run_dir, "fttransformer", meta)

        logging.info("Saved checkpoint: %s", ckpt_path)
        logging.info("Updated meta: %s", meta_path)
        logging.info("Done")
        return 0
    except Exception:
        logging.exception("FT-Transformer benchmark failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

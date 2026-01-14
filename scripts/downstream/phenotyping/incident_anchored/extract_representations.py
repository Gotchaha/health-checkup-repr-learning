"""Extract SSL representations for incident-anchored phenotyping manifests."""

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
import torch
import yaml
from torch.utils.data import DataLoader

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.downstream.phenotyping.incident_anchored import (
    build_incident_anchored_collate_fn,
    build_incident_anchored_dataset,
    build_incident_anchored_sampler,
)
from src.models import MedicalSSLModel, create_embedders_from_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SSL representations for incident-anchored manifests."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to incident-anchored representation config YAML.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional input name to process (matches inputs[].name).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: expected dict at root, got {type(cfg)}")
    return cfg


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_ssl_backbone(cfg: Dict[str, Any], device: torch.device):
    ckpt_path = cfg["ssl_backbone"]["checkpoint_path"]
    checkpoint = torch.load(ckpt_path, map_location=device)

    embedders = create_embedders_from_config(
        checkpoint["config"]["model"]["embedders"],
        device=device,
    )
    embedders_state = checkpoint.get("embedders_state_dict", {})
    embedders.text.load_state_dict(embedders_state.get("text", {}))
    embedders.categorical.load_state_dict(embedders_state.get("categorical", {}))
    embedders.numerical.load_state_dict(embedders_state.get("numerical", {}))

    vocab_sizes = embedders.get_vocab_sizes()
    model = MedicalSSLModel(
        config=checkpoint["config"]["model"],
        text_vocab_size=vocab_sizes["text"],
        cat_vocab_size=vocab_sizes["categorical"],
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    embedders.eval()

    return model, embedders


def build_dataloader(
    cfg: Dict[str, Any],
    input_item: Dict[str, Any],
    embedders,
):
    data_cfg = cfg["data"]

    dataset = build_incident_anchored_dataset(
        manifest_path=input_item["manifest_path"],
        mcinfo_materialized_path=input_item["mcinfo_materialized_path"],
        demographics_path=data_cfg["demographics_path"],
        use_result=data_cfg.get("use_result", True),
        use_pretokenized_result=data_cfg.get("use_pretokenized_result", True),
        result_tokenized_path=data_cfg.get("result_tokenized_path"),
        use_interview=data_cfg.get("use_interview", False),
        interview_path=data_cfg.get("interview_path"),
        mcinfo_rg_cache_size=data_cfg.get("mcinfo_rg_cache_size", 2),
        mcinfo_validate_footer=data_cfg.get("mcinfo_validate_footer", True),
        manifest_extra_cols=data_cfg.get("manifest_extra_cols"),
        manifest_meta_key="manifest_meta",
    )

    sampler = build_incident_anchored_sampler(
        manifest_path=input_item["manifest_path"],
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=bool(data_cfg.get("shuffle", False)),
    )

    collate_builder = build_incident_anchored_collate_fn(
        config=cfg,
        manifest_meta_tensor_keys=data_cfg.get("manifest_meta_tensor_keys"),
        manifest_meta_key="manifest_meta",
    )

    def collate_wrapper(batch):
        return collate_builder(
            batch=batch,
            code_embedder=embedders.categorical,
            text_embedder=embedders.text,
            device="cpu",
        )

    num_workers = int(data_cfg.get("num_workers", 0))
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    persistent_workers = bool(data_cfg.get("persistent_workers", False))

    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
        "persistent_workers": persistent_workers,
        "collate_fn": collate_wrapper,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataset, dataloader


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if torch.is_tensor(sub_value):
                    value[sub_key] = sub_value.to(device)
    return batch


def build_episode_rows(
    batch: Dict[str, Any],
    segment_lengths: List[int],
    episode_template: str,
    include_meta: bool,
) -> List[Dict[str, Any]]:
    meta = batch.get("manifest_meta", {})

    rows: List[Dict[str, Any]] = []
    cursor = 0

    def _to_python(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        return value

    for length in segment_lengths:
        if length <= 0:
            cursor += 0
            continue
        row: Dict[str, Any] = {}
        base = {key: _to_python(values[cursor]) for key, values in meta.items()}

        if episode_template:
            try:
                row["episode_id"] = episode_template.format(**base)
            except KeyError as exc:
                raise KeyError(
                    f"Missing key for episode_id template: {exc}"
                ) from exc

        if include_meta:
            row.update(base)

        rows.append(row)
        cursor += length

    return rows


def build_fixed_size_list(array_2d: np.ndarray) -> pa.Array:
    if array_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array_2d.shape}")
    flat = pa.array(array_2d.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, array_2d.shape[1])


def write_output(
    output_path: Path,
    rows: List[Dict[str, Any]],
    general_repr: np.ndarray,
    next_repr: np.ndarray,
    compression: str,
) -> None:
    if len(rows) != general_repr.shape[0] or len(rows) != next_repr.shape[0]:
        raise ValueError("Row count and embedding count mismatch")

    columns: Dict[str, pa.Array] = {}
    if rows:
        keys = list(rows[0].keys())
        for key in keys:
            columns[key] = pa.array([r.get(key) for r in rows])

    columns["general_representation"] = build_fixed_size_list(general_repr)
    columns["next_prediction"] = build_fixed_size_list(next_repr)

    table = pa.table(columns)
    pq.write_table(table, output_path, compression=compression)


def write_audit_log(log_dir: Path, payload: Dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"audit_{ts}.json"
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path


def main() -> int:
    args = parse_args()
    setup_logging()

    cfg = load_config(Path(args.config))
    device = resolve_device(cfg.get("device", "cpu"))

    model, embedders = load_ssl_backbone(cfg, device)

    inputs = cfg.get("inputs", [])
    if args.only:
        inputs = [item for item in inputs if item.get("name") == args.only]
        if not inputs:
            raise ValueError(f"No input matched name: {args.only}")

    output_cfg = cfg.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    overwrite = bool(output_cfg.get("overwrite", False))
    compression = output_cfg.get("compression", "zstd")
    filename_template = output_cfg.get("filename_template", "{name}.parquet")
    log_dir = Path(output_cfg.get("log_dir", output_dir))

    episode_cfg = cfg.get("data", {}).get("episode_id", {})
    episode_enabled = bool(episode_cfg.get("enabled", True))
    episode_template = episode_cfg.get("template", "{cohort}|{anchor_code}|{person_id}|{index_date}")

    audit_entries: List[Dict[str, Any]] = []

    for item in inputs:
        name = item.get("name", "input")
        output_path = output_dir / filename_template.format(name=name)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {output_path}")

        dataset, dataloader = build_dataloader(cfg, item, embedders)

        all_rows: List[Dict[str, Any]] = []
        general_list: List[np.ndarray] = []
        next_list: List[np.ndarray] = []

        with torch.inference_mode():
            for batch in dataloader:
                if not batch:
                    continue
                batch = move_batch_to_device(batch, device)
                outputs = model(
                    batch,
                    embedders.categorical,
                    embedders.numerical,
                    embedders.text,
                )

                segment_lengths = batch.get("segment_lengths", [])
                rows = build_episode_rows(
                    batch=batch,
                    segment_lengths=segment_lengths,
                    episode_template=episode_template if episode_enabled else "",
                    include_meta=bool(output_cfg.get("include_manifest_meta", True)),
                )

                general_repr = outputs.general_representation.detach().cpu().numpy()
                next_repr = outputs.next_prediction.detach().cpu().numpy()

                if len(rows) != general_repr.shape[0]:
                    raise ValueError(
                        f"Episode/meta count mismatch: rows={len(rows)} repr={general_repr.shape[0]}"
                    )

                all_rows.extend(rows)
                general_list.append(general_repr)
                next_list.append(next_repr)

        if not all_rows:
            logger.warning("No rows produced for %s", name)
            dataset.close()
            continue

        general_concat = np.concatenate(general_list, axis=0)
        next_concat = np.concatenate(next_list, axis=0)

        write_output(
            output_path=output_path,
            rows=all_rows,
            general_repr=general_concat,
            next_repr=next_concat,
            compression=compression,
        )

        dataset.close()

        audit_entries.append(
            {
                "name": name,
                "manifest_path": item.get("manifest_path"),
                "mcinfo_materialized_path": item.get("mcinfo_materialized_path"),
                "output_path": str(output_path),
                "rows": len(all_rows),
            }
        )

        logger.info("Wrote %d rows to %s", len(all_rows), output_path)

    audit_payload = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "checkpoint_path": cfg.get("ssl_backbone", {}).get("checkpoint_path"),
        "inputs": audit_entries,
    }
    audit_path = write_audit_log(log_dir, audit_payload)
    logger.info("Wrote audit log to %s", audit_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

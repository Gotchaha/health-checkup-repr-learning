"""
Export SSL representations for a manifest to a Parquet file.

This script runs the frozen SSL backbone over complete-individual batches and
writes exam-level post_causal embeddings with optional manifest columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

# Make project root importable (script lives in scripts/downstream/)
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.chdir(project_root)

if not (project_root / "src").exists() or not (project_root / "config").exists():
    raise RuntimeError(f"Not in project root directory. Current: {project_root}")

from src.models import HealthExamDataset, MedicalSSLModel, collate_exams, create_embedders_from_config


class CompletePersonBatchSampler:
    """
    Person-aware sampler that keeps complete individuals in each batch.

    Optionally filters by split when a manifest column is present.
    """

    def __init__(
        self,
        manifest_path: Path,
        batch_size: int,
        split: str = "all",
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.split = split
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._skip_summary_logged = False
        self.persons = self._build_person_index_map()

    def _build_person_index_map(self) -> List[Tuple[str, List[int]]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        df = pd.read_parquet(self.manifest_path).reset_index(drop=True)
        if "person_id" not in df.columns:
            raise ValueError("Manifest missing required column: person_id")

        split_col_present = "split" in df.columns
        split_filter = self.split.lower()
        if split_filter != "all" and not split_col_present:
            raise ValueError("Manifest has no split column but --split is not 'all'")

        person_groups: List[Tuple[str, List[int]]] = []
        for person_id, person_group in df.groupby("person_id", sort=False):
            exam_indices = person_group.index.tolist()
            if split_filter != "all":
                splits = person_group["split"].astype(str).str.lower().unique().tolist()
                if len(splits) != 1:
                    raise ValueError(
                        f"Person {person_id} has multiple splits: {sorted(splits)}"
                    )
                if splits[0] != split_filter:
                    continue
            if exam_indices:
                person_groups.append((person_id, exam_indices))

        return person_groups

    def __iter__(self) -> Iterator[List[int]]:
        persons = self.persons.copy()
        if self.shuffle:
            import random

            random.shuffle(persons)

        person_idx = 0
        skipped_persons = 0
        skipped_total_exams = 0
        skipped_max_exams = 0

        while person_idx < len(persons):
            current_batch: List[int] = []
            while person_idx < len(persons):
                _, exam_indices = persons[person_idx]
                slots_available = self.batch_size - len(current_batch)

                if len(exam_indices) <= slots_available:
                    current_batch.extend(exam_indices)
                    person_idx += 1
                else:
                    break

            if len(current_batch) == 0:
                if not self._skip_summary_logged:
                    exams = len(persons[person_idx][1])
                    skipped_persons += 1
                    skipped_total_exams += exams
                    skipped_max_exams = max(skipped_max_exams, exams)
                person_idx += 1
                continue

            if self.drop_last and len(current_batch) < self.batch_size:
                break

            yield current_batch

        if (not self._skip_summary_logged) and skipped_persons > 0:
            logging.getLogger(__name__).warning(
                "Skipping %d persons (total_exams=%d, max_exams=%d, batch_size=%d)",
                skipped_persons,
                skipped_total_exams,
                skipped_max_exams,
                self.batch_size,
            )
            self._skip_summary_logged = True


class ManifestAugmentedDataset(HealthExamDataset):
    """
    HealthExamDataset wrapper that attaches extra manifest columns to each sample.
    """

    def __init__(self, extra_columns: Sequence[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.extra_columns = list(extra_columns)
        self._validate_columns()

    def _validate_columns(self) -> None:
        missing = [c for c in self.extra_columns if c not in self.manifest.schema.names]
        if missing:
            raise ValueError(f"Manifest missing columns for export: {missing}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = super().__getitem__(idx)
        if not self.extra_columns:
            return result
        manifest_row = self.manifest.slice(idx, 1).to_pydict()
        for col in self.extra_columns:
            result[col] = manifest_row[col][0]
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SSL representations to Parquet.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_path", type=str, default=None, help="Output Parquet path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory root")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--include_columns", type=str, default=None, help="Comma-separated manifest columns to include")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum batches to export")
    parser.add_argument("--no_timestamp", action="store_true", help="Disable timestamp in output dir")
    return parser.parse_args()


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("repr_export")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device)
    return batch_data


def unpack_individual_sequences(
    individual_emb: torch.Tensor,
    segment_lengths: Sequence[int],
) -> torch.Tensor:
    total = int(sum(segment_lengths))
    if total == 0:
        return individual_emb.new_empty((0, individual_emb.size(-1)))
    D = individual_emb.size(-1)
    exam_embeddings = torch.zeros(total, D, device=individual_emb.device, dtype=individual_emb.dtype)
    idx = 0
    for i, length in enumerate(segment_lengths):
        exam_embeddings[idx:idx + length] = individual_emb[i, :length]
        idx += length
    return exam_embeddings


def build_t_index(segment_lengths: Sequence[int]) -> List[int]:
    t_index: List[int] = []
    for length in segment_lengths:
        t_index.extend(list(range(length)))
    return t_index


def export_collate_fn(
    batch: List[Dict[str, Any]],
    code_embedder,
    text_embedder,
    config: Dict[str, Any],
    include_columns: Sequence[str],
) -> Dict[str, Any]:
    extra_columns: Dict[str, List[Any]] = defaultdict(list)
    for sample in batch:
        for col in include_columns:
            if col in sample:
                extra_columns[col].append(sample[col])

    outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=config,
        device="cpu",
    )
    outputs["export_columns"] = dict(extra_columns)
    return outputs


def resolve_output_paths(
    config: Dict[str, Any],
    output_dir: Optional[str],
    output_path: Optional[str],
    log_dir: Optional[str],
    no_timestamp: bool,
) -> Tuple[Path, Path, Path]:
    output_root = Path(output_dir) if output_dir else Path("outputs/downstream/repr_export")
    log_root = Path(log_dir) if log_dir else Path("outputs/downstream/repr_export")
    run_name = config.get("experiment_name", "repr_export")
    if not no_timestamp:
        run_name = f"{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_out_dir = output_root / run_name
    run_log_dir = log_root / run_name

    if output_path:
        out_path = Path(output_path)
    else:
        out_path = run_out_dir / "representations.parquet"

    audit_path = run_log_dir / "audit.json"
    return out_path, run_log_dir, audit_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)

    out_path, log_dir, audit_path = resolve_output_paths(
        config=config,
        output_dir=args.output_dir,
        output_path=args.output_path,
        log_dir=args.log_dir,
        no_timestamp=args.no_timestamp,
    )
    logger = setup_logging(log_dir)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    data_cfg = config["data"]
    manifest_path = Path(data_cfg["manifest_path"])

    include_columns = []
    if args.include_columns:
        include_columns = [c.strip() for c in args.include_columns.split(",") if c.strip()]
    else:
        include_columns = ["exam_id", "person_id", "ExamDate"]
    required_cols = ["exam_id", "person_id", "ExamDate"]
    for col in required_cols:
        if col not in include_columns:
            include_columns.append(col)

    manifest_cols = set(pq.read_schema(manifest_path).names)
    missing_required = [c for c in required_cols if c not in manifest_cols]
    if missing_required:
        raise ValueError(f"Manifest missing required columns: {missing_required}")
    include_columns = [c for c in include_columns if c in manifest_cols]

    dataset = ManifestAugmentedDataset(
        extra_columns=include_columns,
        split_name="repr_export",
        manifest_path=str(manifest_path),
        mcinfo_dir=data_cfg["mcinfo_dir"],
        demographics_path=data_cfg["demographics_path"],
        use_result=data_cfg.get("use_result", True),
        result_path=data_cfg.get("result_path"),
        use_interview=data_cfg.get("use_interview", False),
        interview_path=data_cfg.get("interview_path"),
        use_pretokenized_result=data_cfg.get("use_pretokenized_result", False),
        result_tokenized_path=data_cfg.get("result_tokenized_path", None),
        mcinfo_materialized_path=data_cfg.get("mcinfo_materialized_path"),
        mcinfo_rg_cache_size=data_cfg.get("mcinfo_rg_cache_size", 2),
        mcinfo_validate_footer=data_cfg.get("mcinfo_validate_footer", True),
    )

    batch_size = args.batch_size or config["datamodule"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 4)

    sampler = CompletePersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        split=args.split,
        shuffle=False,
        drop_last=False,
    )

    checkpoint_path = config["ssl_backbone"]["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    embedders_cfg = checkpoint["config"]["model"]["embedders"]
    embedders_cpu = create_embedders_from_config(embedders_cfg, device="cpu")
    embedders_device = create_embedders_from_config(embedders_cfg, device=str(device))
    embedders_state = checkpoint["embedders_state_dict"]
    embedders_device.text.load_state_dict(embedders_state["text"])
    embedders_device.categorical.load_state_dict(embedders_state["categorical"])
    embedders_device.numerical.load_state_dict(embedders_state["numerical"])

    vocab_sizes = embedders_device.get_vocab_sizes()
    ssl_model = MedicalSSLModel(
        config=checkpoint["config"]["model"],
        text_vocab_size=vocab_sizes["text"],
        cat_vocab_size=vocab_sizes["categorical"],
        device=str(device),
    ).to(device)
    ssl_model.load_state_dict(checkpoint["model_state_dict"])
    ssl_model.eval()
    embedders_device.eval()

    logger.info("Exporting with device=%s, dtype=%s", device, args.dtype)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Split: %s | batch_size=%d | num_workers=%d", args.split, batch_size, num_workers)
    training_cfg = config.get("training", {})
    masking_keys = ["p_mlm", "p_mcm", "p_cvr", "p_mcc"]
    masking_values = {k: float(training_cfg.get(k, 0.0) or 0.0) for k in masking_keys}
    if any(v > 0 for v in masking_values.values()) or training_cfg.get("use_held_out_codes", False):
        logger.warning("Masking or held-out code filtering is enabled: %s", masking_values)

    collate_fn = lambda batch: export_collate_fn(
        batch=batch,
        code_embedder=embedders_cpu.categorical,
        text_embedder=embedders_cpu.text,
        config=config,
        include_columns=include_columns,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        persistent_workers=True if num_workers > 0 else False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None

    total_exams = 0
    unique_persons = set()
    batches_exported = 0
    start_time = datetime.now()

    for batch_idx, batch_data in enumerate(loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break

        export_cols = batch_data.pop("export_columns", {})
        batch_data = move_batch_to_device(batch_data, device)

        with torch.no_grad():
            outputs = ssl_model(
                batch_data,
                embedders_device.categorical,
                embedders_device.numerical,
                embedders_device.text,
            )
            post_emb = outputs.post_causal_emb

        segment_lengths = batch_data["segment_lengths"]
        exam_emb = unpack_individual_sequences(post_emb, segment_lengths)
        t_index = build_t_index(segment_lengths)

        exam_emb = exam_emb.to(dtype=dtype, device="cpu").contiguous()
        emb_flat = exam_emb.view(-1).numpy()
        emb_array = pa.FixedSizeListArray.from_arrays(
            pa.array(emb_flat, type=pa.float16() if dtype == torch.float16 else pa.float32()),
            exam_emb.size(-1),
        )

        columns: Dict[str, Any] = {}
        columns["t_index"] = t_index
        columns["post_causal_emb"] = emb_array
        for col, values in export_cols.items():
            columns[col] = values

        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        total_exams += len(t_index)
        if "person_id" in export_cols:
            unique_persons.update(export_cols["person_id"])
        batches_exported += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                "Exported %d batches | exams=%d | unique_persons=%d",
                batches_exported,
                total_exams,
                len(unique_persons),
            )

    if writer is not None:
        writer.close()

    end_time = datetime.now()
    audit = {
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "manifest_path": str(manifest_path),
        "checkpoint_path": checkpoint_path,
        "output_path": str(out_path),
        "split": args.split,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "total_exams": total_exams,
        "total_persons": len(unique_persons),
        "include_columns": include_columns,
        "config_path": str(config_path),
    }
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    logger.info("Export complete. Exams=%d Persons=%d", total_exams, len(unique_persons))
    logger.info("Output: %s", out_path)
    logger.info("Audit: %s", audit_path)


if __name__ == "__main__":
    main()

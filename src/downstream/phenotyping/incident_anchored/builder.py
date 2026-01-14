# src/downstream/phenotyping/incident_anchored/builder.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyarrow.parquet as pq

from .collate_fn import incident_anchored_collate_fn
from .dataset import IncidentAnchoredDataset
from .sampler import IncidentAnchoredPersonBatchSampler


DEFAULT_MANIFEST_EXTRA_COLS = [
    "exam_id",
    "person_id",
    "ExamDate",
    "cohort",
    "anchor_code",
    "anchor_name",
    "K_washout",
    "W_years",
    "window_start",
    "index_date",
    "is_index",
    "t_rel",
    "Gender",
    "age_at_index",
    "age_bin",
    "BirthYear",
    "pre_total_obs",
    "pre_in_5y_obs",
    "case_person_id",
    "match_key",
    "min_pre_in_W",
    "n_candidates_in_stratum",
]


DEFAULT_MANIFEST_META_TENSOR_KEYS = [
    "t_rel",
    "is_index",
    "age_at_index",
    "pre_total_obs",
    "pre_in_5y_obs",
    "min_pre_in_W",
    "n_candidates_in_stratum",
    "K_washout",
    "W_years",
]


def select_manifest_extra_cols(
    manifest_path: Union[str, Path],
    candidate_cols: Optional[List[str]] = None,
) -> List[str]:
    manifest_path = Path(manifest_path)
    candidates = candidate_cols or DEFAULT_MANIFEST_EXTRA_COLS
    schema = pq.read_schema(manifest_path)
    available = set(schema.names)
    return [c for c in candidates if c in available]


def build_incident_anchored_dataset(
    manifest_path: Union[str, Path],
    manifest_extra_cols: Optional[List[str]] = None,
    manifest_meta_key: str = "manifest_meta",
    **kwargs: Any,
) -> IncidentAnchoredDataset:
    if manifest_extra_cols is None:
        manifest_extra_cols = select_manifest_extra_cols(manifest_path)
    return IncidentAnchoredDataset(
        manifest_path=manifest_path,
        manifest_extra_cols=manifest_extra_cols,
        manifest_meta_key=manifest_meta_key,
        **kwargs,
    )


def build_incident_anchored_sampler(
    manifest_path: Union[str, Path],
    batch_size: int,
    shuffle: bool = False,
    index_col: str = "is_index",
) -> IncidentAnchoredPersonBatchSampler:
    return IncidentAnchoredPersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=shuffle,
        index_col=index_col,
    )


def build_incident_anchored_collate_fn(
    config: Dict[str, Any],
    manifest_meta_tensor_keys: Optional[List[str]] = None,
    manifest_meta_key: str = "manifest_meta",
) -> Any:
    tensor_keys = manifest_meta_tensor_keys or DEFAULT_MANIFEST_META_TENSOR_KEYS

    def _collate(
        batch: List[Dict[str, Any]],
        code_embedder,
        text_embedder,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        return incident_anchored_collate_fn(
            batch=batch,
            code_embedder=code_embedder,
            text_embedder=text_embedder,
            config=config,
            device=device,
            manifest_meta_key=manifest_meta_key,
            manifest_meta_tensor_keys=tensor_keys,
        )

    return _collate


__all__ = [
    "IncidentAnchoredDataset",
    "IncidentAnchoredPersonBatchSampler",
    "incident_anchored_collate_fn",
    "DEFAULT_MANIFEST_EXTRA_COLS",
    "DEFAULT_MANIFEST_META_TENSOR_KEYS",
    "select_manifest_extra_cols",
    "build_incident_anchored_dataset",
    "build_incident_anchored_sampler",
    "build_incident_anchored_collate_fn",
]

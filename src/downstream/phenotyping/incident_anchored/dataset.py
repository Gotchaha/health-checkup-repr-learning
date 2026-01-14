# src/downstream/phenotyping/incident_anchored/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.models import HealthExamDataset


class IncidentAnchoredDataset(HealthExamDataset):
    """
    Dataset wrapper for incident-anchored phenotyping manifests.

    Extends HealthExamDataset by optionally passing through selected
    manifest columns for downstream analysis and representation export.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        manifest_extra_cols: Optional[List[str]] = None,
        manifest_meta_key: str = "manifest_meta",
        **kwargs: Any,
    ) -> None:
        manifest_path = Path(manifest_path)
        super().__init__(
            split_name="incident_anchored",
            manifest_path=str(manifest_path),
            **kwargs,
        )
        self.manifest_extra_cols = manifest_extra_cols or []
        self.manifest_meta_key = manifest_meta_key
        self._manifest_extra_validated = False

    def _validate_manifest_extra_cols(self) -> None:
        if self._manifest_extra_validated:
            return

        required_cols = ["person_id", "exam_id", "ExamDate", "is_index", "t_rel"]
        missing_required = [c for c in required_cols if c not in self.manifest.column_names]
        if missing_required:
            raise ValueError(
                f"Manifest missing required columns: {missing_required}"
            )

        if self.manifest_extra_cols:
            missing_extra = [
                c for c in self.manifest_extra_cols if c not in self.manifest.column_names
            ]
            if missing_extra:
                raise ValueError(
                    f"Manifest missing requested extra columns: {missing_extra}"
                )

        self._manifest_extra_validated = True

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = super().__getitem__(idx)
        self._validate_manifest_extra_cols()

        if self.manifest_extra_cols:
            manifest_row = self.manifest.slice(idx, 1).to_pydict()
            meta = {col: manifest_row[col][0] for col in self.manifest_extra_cols}
        else:
            meta = {}

        data[self.manifest_meta_key] = meta
        return data

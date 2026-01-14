# tests/optimization/test_dataset_opt.py
# -*- coding: utf-8 -*-
"""
Purpose: smoke test that the materialized path works end-to-end,
         and a sampled equivalence test vs. the legacy (hive) path.
Notes:
- Uses *real* data paths from the project.
- Keeps tests lightweight: no performance assertions, small random sample.
"""

import os
import sys
import json
import random
from pathlib import Path

import pytest

# Make "src" importable when running from project root
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.dataset import HealthExamDataset  # noqa: E402


# ---- Config (adjust split/materialized path if needed) ----
SPLIT_NAME = "train_ssl"
MAT_PATH = ROOT / "data" / "processed" / "mcinfo_materialized" / "train.parquet"
PRETOKEN_PATH = ROOT / "cache" / "pretokenized" / \
    "result_jmedrobertabasesentencepiece_tok512_trunc_phi.parquet"
MANIFEST_PATH = ROOT / "data" / "splits" / "core" / "sorted" / f"{SPLIT_NAME}.parquet"
MCINFO_HIVE_DIR = ROOT / "data" / "processed" / "mcinfo" / "exam_level"


def _assert_sample_schema(sample: dict):
    """Check presence and basic types of required keys (quick smoke)."""
    for k in ["person_id", "ExamDate", "tests", "birth_year", "gender"]:
        assert k in sample, f"missing key: {k}"

    for k in ["result_input_ids", "result_attention_mask"]:
        assert k in sample, f"missing key: {k}"
        assert isinstance(sample[k], (list, tuple)), f"{k} should be a sequence"

    assert "exam_id" in sample, "missing key: exam_id"
    assert sample["person_id"] is not None
    assert sample["tests"] is not None


def _canonical_default(o):
    """Best-effort serializer for non-JSON-native objects."""
    # Try ISO if it's a date/datetime-like
    iso = getattr(o, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass
    # Fallback to string
    return str(o)


def _canonicalize(obj):
    """Stable string form for deep-equality on nested structures."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_canonical_default)


def _assert_same_type_structure(a: dict, b: dict) -> None:
    """
    Strict type equality for core fields; keep scope minimal to mcinfo path.
    We intentionally ignore extra fields like 'exam_id'.
    """
    for k in ("person_id", "ExamDate", "birth_year", "gender"):
        ta, tb = type(a[k]), type(b[k])
        assert ta is tb, f"type mismatch for {k}: {ta.__name__} vs {tb.__name__}"
    _assert_type_tree(a["tests"], b["tests"], where="tests")


def _assert_type_tree(x, y, where: str = "root") -> None:
    """Recursive strict type/shape checker for the `tests` payload."""
    tx, ty = type(x), type(y)
    assert tx is ty, f"type mismatch at {where}: {tx.__name__} vs {ty.__name__}"

    if isinstance(x, dict):
        kx, ky = set(x.keys()), set(y.keys())
        assert kx == ky, f"keys mismatch at {where}: only_in_left={sorted(kx-ky)}, only_in_right={sorted(ky-kx)}"
        for k in x:
            _assert_type_tree(x[k], y[k], f"{where}.{k}")
    elif isinstance(x, list):
        assert len(x) == len(y), f"length mismatch at {where}: {len(x)} vs {len(y)}"
        for i, (xi, yi) in enumerate(zip(x, y)):
            _assert_type_tree(xi, yi, f"{where}[{i}]")
    else:
        return  # scalar: strict type already enforced


def _assert_equal_samples(a: dict, b: dict):
    """Strict value equality for a set of keys; 'tests' compared structurally."""
    # Basic identifiers & demographics
    for k in ["person_id", "ExamDate", "exam_id", "birth_year", "gender"]:
        assert a[k] == b[k], f"value mismatch at key={k}"

    # Pretokenized result arrays
    for k in ["result_input_ids", "result_attention_mask"]:
        assert a[k] == b[k], f"value mismatch at key={k}"

    # tests: nested structure, compare via canonical JSON
    assert _canonicalize(a["tests"]) == _canonicalize(b["tests"]), "tests mismatch"


# ---- Smoke test: materialized path works and yields expected schema ----
@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="manifest parquet not found")
@pytest.mark.skipif(not MAT_PATH.exists(), reason="materialized mcinfo parquet not found")
@pytest.mark.skipif(not PRETOKEN_PATH.exists(), reason="pretokenized result parquet not found")
def test_dataset_materialized_smoke():
    ds = HealthExamDataset(
        split_name=SPLIT_NAME,
        mcinfo_materialized_path=str(MAT_PATH),
        use_result=True,
        use_interview=False,
        use_pretokenized_result=True,
        result_tokenized_path=str(PRETOKEN_PATH),
    )

    n = len(ds)
    assert n > 0, "Empty dataset length"

    # Probe a few indices to exercise row-group locality slightly
    idxs = {0, min(10, n - 1), n - 1}
    if n > 1:
        idxs.add(1)

    for i in sorted(idxs):
        sample = ds[i]
        _assert_sample_schema(sample)

    # Idempotence + tiny LRU (no assertions beyond schema)
    _ = ds[0]
    if n > 1:
        _ = ds[1]

    ds.close()


# ---- Sampled equivalence test: materialized vs legacy outputs match ----
@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="manifest parquet not found")
@pytest.mark.skipif(not MAT_PATH.exists(), reason="materialized mcinfo parquet not found")
@pytest.mark.skipif(not PRETOKEN_PATH.exists(), reason="pretokenized result parquet not found")
@pytest.mark.skipif(not MCINFO_HIVE_DIR.exists(), reason="hive-partitioned mcinfo dir not found")
def test_dataset_materialized_equivalence_sample():
    # New path (materialized)
    ds_new = HealthExamDataset(
        split_name=SPLIT_NAME,
        mcinfo_materialized_path=str(MAT_PATH),
        use_result=True,
        use_interview=False,
        use_pretokenized_result=True,
        result_tokenized_path=str(PRETOKEN_PATH),
    )

    # Legacy path (hive, per-sample filter)
    ds_old = HealthExamDataset(
        split_name=SPLIT_NAME,
        use_result=True,
        use_interview=False,
        use_pretokenized_result=True,
        result_tokenized_path=str(PRETOKEN_PATH),
    )

    # n = min(len(ds_new), len(ds_old))
    # assert n > 0, "Empty dataset length"
    
    # Stronger invariant: the two datasets must have exactly the same length (and be non-empty)
    assert len(ds_new) == len(ds_old) > 0, "materialized and legacy lengths differ (or empty)"
    n = len(ds_new)

    # Small, reproducible sample to avoid heavy legacy scans
    random.seed(0)
    base = {0, min(1, n - 1), n - 1}
    extra = set(random.sample(range(n), k=min(64, n)))
    idxs = sorted(base | extra)

    for i in idxs:
        a = ds_new[i]
        b = ds_old[i]
        _assert_sample_schema(a)
        _assert_sample_schema(b)
        _assert_same_type_structure(a, b)
        _assert_equal_samples(a, b)

    ds_new.close()
    ds_old.close()

"""
Profiled version of HealthExamDataset.
Goal: keep semantics IDENTICAL to src/models/dataset.py while adding minimal instrumentation.
This module is profiling-only and should not be imported by training code.
"""

import os
import time
import json
import gc
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# -------------------------
# Small helpers
# -------------------------

def _nested_list_dd():
    return defaultdict(list)


class ProfiledHealthExamDataset(Dataset):
    """
    Exact-behavior copy of HealthExamDataset with step-wise profiling.
    - File paths and options are identical.
    - Lazy init and access patterns mirror the original implementation.
    - Added: per-worker JSONL timers and counters (multi-worker safe).
    """

    def __init__(
        self,
        split_name: str,
        mcinfo_dir: str = 'data/processed/mcinfo/exam_level/',
        demographics_path: str = 'data/processed/person.parquet',
        use_result: bool = True,
        result_path: str = 'data/processed/result.parquet',
        use_interview: bool = False,
        interview_path: str = 'data/processed/interview.parquet',
        use_pretokenized_result: bool = False,
        result_tokenized_path: Optional[str] = None,
        # profiling flags
        enable_profiling: bool = True,
        detailed_mcinfo: bool = True,
        start_logging_after_items: int = 0,
    ):
        # ---- same config & paths as original dataset ----
        self.split_name = split_name
        self.manifest_path = f"data/splits/core/sorted/{split_name}.parquet"
        self.mcinfo_dir = mcinfo_dir
        self.demographics_path = demographics_path
        self.use_result = use_result
        self.result_path = result_path
        self.use_interview = use_interview
        self.interview_path = interview_path
        self.use_pretokenized_result = use_pretokenized_result
        self.result_tokenized_path = result_tokenized_path

        # ---- lazy state & cached metadata (same shape as original) ----
        self._n_rows: Optional[int] = None
        self._manifest = None
        self._mcinfo_ds = None
        self._demographics = None
        self._demographics_dict: Optional[Dict[str, Dict[str, Any]]] = None
        self._result_data = None
        self._result_lookup: Optional[Dict[tuple, Any]] = None
        self._interview_data = None
        self._interview_lookup: Optional[Dict[tuple, str]] = None
        self._initialized = False

        # ---- constants (kept as in original) ----
        self.EMPTY_RESULT_DATA = {'input_ids': [0, 1], 'attention_mask': [1, 1]}

        # ---- profiling state ----
        self.enable_profiling = bool(enable_profiling)
        self.detailed_mcinfo = bool(detailed_mcinfo)
        self.start_after_items = int(start_logging_after_items or 0)

        self.simple_metrics = defaultdict(list)          # seconds
        self.detailed_metrics = defaultdict(_nested_list_dd)
        self.simple_counters = defaultdict(list)         # dimensionless
        self.detailed_counters = defaultdict(_nested_list_dd)

        self._items_seen = 0
        self._calls = 0

        self.worker_id = -1
        self.proc_id = os.getpid()
        self.run_id = os.environ.get("DL_PROFILE_RUN", "run")
        self.log_dir = os.environ.get("DL_PROFILE_DIR", "./dl_profile_logs")
        self.flush_every = int(os.environ.get("DL_PROFILE_FLUSH_EVERY", "200"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"dataset-{self.run_id}-w{self.worker_id}-p{self.proc_id}-{self.split_name}.jsonl")
        self._profiled_calls = 0  # only increments when logging is enabled and past warmup


    
    def _ensure_worker_identity(self) -> None:
        wi = get_worker_info()
        wid = wi.id if wi else -1
        pid = os.getpid()
        if (wid != self.worker_id) or (pid != self.proc_id) or (not getattr(self, "log_path", None)):
            self.worker_id = wid
            self.proc_id = pid
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = os.path.join(
                self.log_dir,
                f"dataset-{self.run_id}-w{wid}-p{pid}-{self.split_name}.jsonl"
            )

    # -----------------------------
    # Lazy init
    # -----------------------------

    def _lazy_init(self) -> None:
        """Load all Arrow objects like the original dataset; one-time per worker."""
        if self._initialized:
            return

        # load manifest
        t0 = time.perf_counter()
        self._manifest = pq.read_table(self.manifest_path)
        if self.enable_profiling and (self._items_seen >= self.start_after_items):
            self.simple_metrics['manifest_load'].append(time.perf_counter() - t0)

        # load mcinfo dataset
        t0 = time.perf_counter()
        self._mcinfo_ds = ds.dataset(self.mcinfo_dir, format="parquet", partitioning="hive")
        if self.enable_profiling and (self._items_seen >= self.start_after_items):
            self.simple_metrics['mcinfo_open'].append(time.perf_counter() - t0)

        # load demographics
        t0 = time.perf_counter()
        self._demographics = pq.read_table(self.demographics_path)
        self._demographics_dict = self._create_demographics_dict(self._demographics)
        if self.enable_profiling and (self._items_seen >= self.start_after_items):
            self.simple_metrics['demographics_dict_build'].append(time.perf_counter() - t0)

        # result
        if self.use_result:
            if self.use_pretokenized_result:
                t0 = time.perf_counter()
                # NEW PATH: Load pre-tokenized result data
                self._result_data = pq.read_table(
                    self.result_tokenized_path,
                    columns=['AnonymousID', 'McExamDt', 'input_ids', 'attention_mask'],
                    memory_map=True
                )
                # Build lookup dictionary for O(1) access with string keys for type safety
                key_cols = ['AnonymousID', 'McExamDt']
                rows = self._result_data.select(key_cols + ['input_ids', 'attention_mask']).to_pylist()
                self._result_lookup = {
                    (str(row['AnonymousID']), row['McExamDt'].strftime('%Y-%m-%d')): {
                        'input_ids': row['input_ids'],
                        'attention_mask': row['attention_mask']
                    }
                    for row in rows
                }
                if self.enable_profiling and (self._items_seen >= self.start_after_items):
                    self.simple_counters['result_rows'].append(float(len(rows)))
                    self.simple_metrics['result_data_load'].append(time.perf_counter() - t0)
                self._result_data = None
            else:
                t0 = time.perf_counter()
                self._result_data = pq.read_table(
                    self.result_path,
                    columns=['AnonymousID', 'McExamDt', 'ResultText'],
                    memory_map=True,
                )
                rows = self._result_data.select(['AnonymousID', 'McExamDt', 'ResultText']).to_pylist()
                self._result_lookup = {
                    (str(row['AnonymousID']), row['McExamDt'].strftime('%Y-%m-%d')): row['ResultText']
                    for row in rows
                }
                if self.enable_profiling and (self._items_seen >= self.start_after_items):
                    self.simple_counters['result_rows'].append(float(len(rows)))
                    self.simple_metrics['result_data_load'].append(time.perf_counter() - t0)
                self._result_data = None

        # interview
        if self.use_interview:
            t0 = time.perf_counter()
            self._interview_data = pq.read_table(
                self.interview_path,
                columns=['AnonymousID', 'McExamDt', 'Interview'],
                memory_map=True,
            )
            key_cols = ['AnonymousID', 'McExamDt']
            self._interview_lookup = {
                (str(row['AnonymousID']), row['McExamDt'].strftime('%Y-%m-%d')): row['Interview']
                for row in self._interview_data.select(key_cols + ['Interview']).to_pylist()
            }
            if self.enable_profiling and (self._items_seen >= self.start_after_items):
                self.simple_metrics['interview_data_load'].append(time.perf_counter() - t0)
            self._interview_data = None

        self._initialized = True

    # -----------------------------
    # Properties
    # -----------------------------

    @property
    def manifest(self):
        if self._manifest is None:
            self._lazy_init()
        return self._manifest

    @property
    def mcinfo_ds(self):
        if self._mcinfo_ds is None:
            self._lazy_init()
        return self._mcinfo_ds

    @property
    def demographics(self):
        if self._demographics is None:
            self._lazy_init()
        return self._demographics

    @property
    def demographics_dict(self):
        if self._demographics_dict is None:
            self._lazy_init()
        return self._demographics_dict

    @property
    def result_data(self):
        if self.use_result and self._result_data is None:
            self._lazy_init()
        return self._result_data
    
    @property
    def interview_data(self):
        if self.use_interview and self._interview_data is None:
            self._lazy_init()
        return self._interview_data

    def _create_demographics_dict(self, table: pa.Table) -> Dict[str, Dict[str, Any]]:
        """
        Create a dictionary for fast lookup of demographics by person_id using PyArrow.
        
        Args:
            table: PyArrow table with demographics data
            
        Returns:
            Dictionary mapping person_id to demographics data
        """
        pydict = table.to_pydict()
        demographics_dict = {}
        
        for person_id, birth_year, gender in zip(
            pydict['person_id'], 
            pydict['BirthYear'], 
            pydict['Gender']
        ):
            demographics_dict[person_id] = {
                'birth_year': birth_year,
                'gender': gender
            }
        
        return demographics_dict

    # -----------------------------
    # Length & indexing
    # -----------------------------

    def __len__(self) -> int:
        if self._n_rows is None:
            self._n_rows = pq.read_metadata(self.manifest_path).num_rows
        return self._n_rows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_worker_identity()
        self._calls += 1
        total_t0 = time.perf_counter()

        # manifest slice
        t0 = time.perf_counter()
        row = self.manifest.slice(idx, 1).to_pydict()
        if self.enable_profiling and (self._items_seen >= self.start_after_items):
            self.simple_metrics['manifest_access'].append(time.perf_counter() - t0)

        person_id = row['person_id'][0]
        exam_date = row['ExamDate'][0]
        year = exam_date.year
        exam_id = row['exam_id'][0]

        # mcinfo
        if self.detailed_mcinfo:
            t0 = time.perf_counter()
            mc_filter = (ds.field('year') == year) & (ds.field('exam_id') == exam_id)
            if self.enable_profiling and (self._items_seen >= self.start_after_items):
                self.detailed_metrics['mcinfo']['filter_construction'].append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            mc_row = self.mcinfo_ds.filter(mc_filter).to_table()
            if self.enable_profiling and (self._items_seen >= self.start_after_items):
                self.detailed_metrics['mcinfo']['dataset_filter'].append(time.perf_counter() - t1)

            if mc_row.num_rows == 0:
                raise KeyError(f"No exam found with id {exam_id} in year {year}")

            t2 = time.perf_counter()
            mcinfo_dict = {col: mc_row.column(col)[0].as_py() for col in mc_row.column_names}
            if self.enable_profiling and (self._items_seen >= self.start_after_items):
                self.detailed_metrics['mcinfo']['to_dict'].append(time.perf_counter() - t2)
        else:
            mc_filter = (ds.field('year') == year) & (ds.field('exam_id') == exam_id)
            mc_row = self.mcinfo_ds.filter(mc_filter).to_table()
            if mc_row.num_rows == 0:
                raise KeyError(f"No exam found with id {exam_id} in year {year}")
            mcinfo_dict = {col: mc_row.column(col)[0].as_py() for col in mc_row.column_names}

        # demographics
        t0 = time.perf_counter()
        demographics = self.demographics_dict.get(person_id, {})
        if self.enable_profiling and (self._items_seen >= self.start_after_items):
            self.simple_metrics['demographics_lookup'].append(time.perf_counter() - t0)

        result = {**mcinfo_dict, 'birth_year': demographics.get('birth_year'), 'gender': demographics.get('gender')}

        # result
        if self.use_result:
            t0 = time.perf_counter()
            key = (str(person_id), exam_date.strftime('%Y-%m-%d'))
            if self.use_pretokenized_result:
                tokenized = self._result_lookup.get(key)
                if tokenized:
                    result['result_input_ids'] = tokenized['input_ids']
                    result['result_attention_mask'] = tokenized['attention_mask']
                else:
                    result['result_input_ids'] = self.EMPTY_RESULT_DATA['input_ids']
                    result['result_attention_mask'] = self.EMPTY_RESULT_DATA['attention_mask']
            else:
                result['result_text'] = self._result_lookup.get(key, "") or ""
            if self.enable_profiling and (self._items_seen >= self.start_after_items):
                self.simple_metrics['result_lookup'].append(time.perf_counter() - t0)

        # interview
        if self.use_interview:
            t0 = time.perf_counter()
            key = (str(person_id), exam_date.strftime('%Y-%m-%d'))
            result['interview'] = self._interview_lookup.get(key, "")
            if self.enable_profiling and (self._items_seen >= self.start_after_items):
                self.simple_metrics['interview_lookup'].append(time.perf_counter() - t0)

        # total & flush
        if self.enable_profiling and (self._items_seen >= self.start_after_items):
            # record total wall time for this item
            self.simple_metrics['total_getitem'].append(time.perf_counter() - total_t0)
        
            # flush cadence is based ONLY on profiled (logged) items
            self._profiled_calls += 1
            if (self._profiled_calls % self.flush_every) == 0:
                self._flush_metrics_to_file()
        
        self._items_seen += 1
        return result


    # -----------------------------
    # JSONL flush & in-process summary
    # -----------------------------

    def _flush_metrics_to_file(self):
        self._ensure_worker_identity()
        path = self.log_path
        if path is None:
            return
        payload = {
            "ts": time.time(),
            "wid": self.worker_id,
            "pid": self.proc_id,
            "simple": {k: list(v) for k, v in self.simple_metrics.items()},
            "detailed": {cat: {m: list(vals) for m, vals in d.items()} for cat, d in self.detailed_metrics.items()},
            "simple_counters": {k: list(v) for k, v in self.simple_counters.items()},
            "detailed_counters": {cat: {m: list(vals) for m, vals in d.items()} for cat, d in self.detailed_counters.items()},
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        self.simple_metrics.clear()
        self.detailed_metrics.clear()
        self.simple_counters.clear()
        self.detailed_counters.clear()

    def get_profiling_summary(self) -> Dict[str, Dict[str, float]]:
        """Single-worker fallback timing summary (keys: simple_*, detailed_cat_metric)."""
        import numpy as np
        out = {}
        for k, vals in self.simple_metrics.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[f"simple_{k}"] = {
                "mean_ms": float(arr.mean() * 1000),
                "std_ms": float(arr.std() * 1000),
                "min_ms": float(arr.min() * 1000),
                "max_ms": float(arr.max() * 1000),
                "p50_ms": float(np.percentile(arr, 50) * 1000),
                "p95_ms": float(np.percentile(arr, 95) * 1000),
                "p99_ms": float(np.percentile(arr, 99) * 1000),
                "count": int(arr.size),
            }
        for cat, d in self.detailed_metrics.items():
            for m, vals in d.items():
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                out[f"detailed_{cat}_{m}"] = {
                    "mean_ms": float(arr.mean() * 1000),
                    "std_ms": float(arr.std() * 1000),
                    "min_ms": float(arr.min() * 1000),
                    "max_ms": float(arr.max() * 1000),
                    "p50_ms": float(np.percentile(arr, 50) * 1000),
                    "p95_ms": float(np.percentile(arr, 95) * 1000),
                    "p99_ms": float(np.percentile(arr, 99) * 1000),
                    "count": int(arr.size),
                }
        return out

    def get_counter_summary(self) -> Dict[str, Dict[str, float]]:
        """Single-worker fallback counters summary (keys: simple_*_count, detailed_cat_m_count)."""
        import numpy as np
        out = {}
        for k, vals in self.simple_counters.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            out[f"simple_{k}_count"] = {
                "sum": float(arr.sum()),
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
                "count": int(arr.size),
            }
        for cat, d in self.detailed_counters.items():
            for m, vals in d.items():
                arr = np.asarray(vals, dtype=float)
                if arr.size == 0:
                    continue
                out[f"detailed_{cat}_{m}_count"] = {
                    "sum": float(arr.sum()),
                    "mean": float(arr.mean()),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(arr.max()),
                    "count": int(arr.size),
                }
        return out

    # -----------------------------
    # Cleanup & mp pickling
    # -----------------------------

    def close(self):
        try:
            if self.enable_profiling:
                self._flush_metrics_to_file()
        except Exception:
            pass
        self._manifest = None
        self._mcinfo_ds = None
        self._demographics = None
        self._demographics_dict = None
        self._result_lookup = None
        self._interview_lookup = None
        gc.collect()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop Arrow objects
        for name in ('_manifest','_mcinfo_ds','_demographics','_demographics_dict','_result_data','_interview_data','_result_lookup','_interview_lookup'):
            state[name] = None
        # Reset run-time counters; keep config & paths
        state['_initialized'] = False
        state['_n_rows'] = None
        state['_calls'] = 0
        state['_items_seen'] = 0
        # Clear buffers (each worker gets fresh)
        state['simple_metrics'] = defaultdict(list)
        state['detailed_metrics'] = defaultdict(_nested_list_dd)
        state['simple_counters'] = defaultdict(list)
        state['detailed_counters'] = defaultdict(_nested_list_dd)
        # worker ids replaced on child side
        state['worker_id'] = -1
        state['proc_id'] = os.getpid()
        # log_path will be recomputed with same run_id
        state['log_path'] = None
        state['_profiled_calls'] = 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

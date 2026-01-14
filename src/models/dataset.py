# src/models/dataset.py

import os
import pandas as pd
import torch
import gc
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import OrderedDict
from bisect import bisect_right


class HealthExamDataset(Dataset):
    """
    Dataset for health examination records with flexible modality support.
    
    This dataset loads exam data from a specified split and supports optional 
    inclusion of result and interview text data.
    
    Note: All PyArrow objects are lazy-loaded to support multiprocessing with num_workers > 0.
    """
    
    def __init__(
        self, 
        split_name: str,
        manifest_path: Optional[str] = None,
        mcinfo_dir: str = 'data/processed/mcinfo/exam_level/',
        demographics_path: str = 'data/processed/person.parquet',
        use_result: bool = True,
        result_path: str = 'data/processed/result.parquet',
        use_interview: bool = False,
        interview_path: str = 'data/processed/interview.parquet',
        use_pretokenized_result: bool = False,
        result_tokenized_path: Optional[str] = None,
        # Materialized mcinfo
        mcinfo_materialized_path: Optional[str] = None,
        mcinfo_rg_cache_size: int = 2,
        mcinfo_validate_footer: bool = True,

    ):
        """
        Initialize the dataset with the specified split and data sources.
        
        Args:
            split_name: Name of the split to use (train_ssl, val_ssl, test_future)
            manifest_path: Path to split manifest file (defaults to legacy core split)
            mcinfo_dir: Directory containing year-partitioned mcinfo data
            demographics_path: Path to demographics Parquet file
            use_result: Whether to load result data
            result_path: Path to result data
            use_interview: Whether to load interview data
            interview_path: Path to interview data
            use_pretokenized_result: Whether to use pre-tokenized result data
            result_tokenized_path: Path to pre-tokenized result data
            mcinfo_materialized_path: Path to the materialized mcinfo Parquet (tests-only, aligned to manifest)
            mcinfo_rg_cache_size: LRU cache size in row-groups for materialized reads (e.g., 2–4)
            mcinfo_validate_footer: Verify footer hashes against the current manifest at init (fail-fast on mismatch)

        """
        # Store only file paths and configuration (NO PyArrow objects)
        self.split_name = split_name
        self.manifest_path = manifest_path or f"data/splits/core/sorted/{split_name}.parquet"
        self.mcinfo_dir = mcinfo_dir
        self.demographics_path = demographics_path
        self.use_result = use_result
        self.result_path = result_path
        self.use_interview = use_interview
        self.interview_path = interview_path
        self.use_pretokenized_result = use_pretokenized_result
        self.result_tokenized_path = result_tokenized_path
        
        # Lazy metadata reading (no PyArrow calls in __init__)
        self._n_rows = None
        
        # Initialize all PyArrow objects as None (lazy loading)
        self._manifest = None
        self._mcinfo_ds = None
        self._demographics = None
        self._demographics_dict = None
        self._result_data = None
        self._result_lookup = None
        self._interview_data = None
        self._interview_lookup = None
        
        # Flag to track if lazy initialization has been done
        self._initialized = False

        # Constants for missing result data (true empty sequence, no special tokens)
        self.EMPTY_RESULT_DATA = {
            'input_ids': [],
            'attention_mask': []
        }

        # --- Materialized mcinfo (ParquetFile + row-group index + tiny LRU cache) ---
        self.mcinfo_materialized_path = mcinfo_materialized_path
        self.mcinfo_rg_cache_size = mcinfo_rg_cache_size
        self.mcinfo_validate_footer = mcinfo_validate_footer
        self._pf = None
        self._rg_sizes = None
        self._rg_offsets = None
        self._rg_cache = None
        self._mat_footer = None

    
    def _lazy_init(self):
        """
        Lazy initialization of all PyArrow objects.
        Called automatically when any data is first accessed.
        This runs independently in each worker process.
        """
        if self._initialized:
            return
            
        # Load manifest
        self._manifest = pq.read_table(self.manifest_path)

        # Load mcinfo (NEW: prefer materialized ParquetFile if provided; else fallback to hive dataset)
        if self.mcinfo_materialized_path:
            # Open ParquetFile and build row-group index
            self._pf = pq.ParquetFile(self.mcinfo_materialized_path)

            # Quick structural checks
            # - Only one column named 'tests' is expected in the materialized file
            schema_names = self._pf.schema_arrow.names
            if schema_names != ["tests"]:
                raise RuntimeError(f"Materialized mcinfo must have only `tests` column, got {schema_names}")

            # - Row count must match manifest
            if self._pf.metadata.num_rows != self._manifest.num_rows:
                raise RuntimeError(
                    f"Row count mismatch: materialized={self._pf.metadata.num_rows} vs "
                    f"manifest={self._manifest.num_rows}"
                )

            # Footer KV metadata for runtime O(1) verification
            meta = self._pf.metadata.metadata or {}
            # Decode bytes -> str safely
            self._mat_footer = { (k.decode() if isinstance(k, bytes) else str(k)) :
                                 (v.decode() if isinstance(v, bytes) else str(v))
                                 for k, v in meta.items() }

            if self.mcinfo_validate_footer:
                # Recompute current manifest hashes and compare
                cur_manifest_sha = self._sha256_file(self.manifest_path)
                cur_order_sha = self._sha256_exam_id_order(self._manifest["exam_id"])
                exp_manifest_sha = self._mat_footer.get("input_manifest_sha256")
                exp_order_sha = self._mat_footer.get("exam_id_order_sha256")
                if exp_manifest_sha and cur_manifest_sha != exp_manifest_sha:
                    raise RuntimeError(
                        f"Manifest SHA mismatch: current={cur_manifest_sha}, expected={exp_manifest_sha}"
                    )
                if exp_order_sha and cur_order_sha != exp_order_sha:
                    raise RuntimeError(
                        f"Exam-id order hash mismatch: current={cur_order_sha}, expected={exp_order_sha}"
                    )

            # Build row-group sizes and offsets (prefix sums)
            n_rg = self._pf.metadata.num_row_groups
            self._rg_sizes = [self._pf.metadata.row_group(i).num_rows for i in range(n_rg)]
            self._rg_offsets = [0]
            for s in self._rg_sizes[:-1]:
                self._rg_offsets.append(self._rg_offsets[-1] + s)

            # Tiny LRU cache for recently used row-groups
            self._rg_cache = OrderedDict()

            # Disable hive dataset to avoid accidental use
            self._mcinfo_ds = None
        else:
            # OLD path: hive-partitioned dataset (year=YYYY)
            self._mcinfo_ds = ds.dataset(self.mcinfo_dir, format="parquet", partitioning="hive")

        
        # Load demographics
        self._demographics = pq.read_table(self.demographics_path)
        self._demographics_dict = self._create_demographics_dict(self._demographics)
        
        # Load optional modalities with memory mapping and build lookup dictionaries
        if self.use_result:
            if self.use_pretokenized_result:
                # NEW PATH: Load pre-tokenized result data
                self._result_data = pq.read_table(self.result_tokenized_path, 
                                                columns=['AnonymousID', 'McExamDt', 'input_ids', 'attention_mask'],
                                                memory_map=True)
                # Build lookup dictionary for O(1) access with string keys for type safety
                key_cols = ['AnonymousID', 'McExamDt']
                self._result_lookup = {
                    (str(row['AnonymousID']), row['McExamDt'].strftime('%Y-%m-%d')): {
                        'input_ids': row['input_ids'],
                        'attention_mask': row['attention_mask']
                    }
                    for row in self._result_data.select(key_cols + ['input_ids', 'attention_mask']).to_pylist()
                }
                self._result_data = None      # Free mmap handle
            else:
                # EXISTING PATH: Load raw text result data (unchanged)
                self._result_data = pq.read_table(self.result_path, 
                                                columns=['AnonymousID', 'McExamDt', 'ResultText'],
                                                memory_map=True)
                # Build lookup dictionary for O(1) access with string keys for type safety
                key_cols = ['AnonymousID', 'McExamDt']
                self._result_lookup = {
                    (str(row['AnonymousID']), row['McExamDt'].strftime('%Y-%m-%d')): row['ResultText']
                    for row in self._result_data.select(key_cols + ['ResultText']).to_pylist()
                }
                self._result_data = None      # Free mmap handle
        
        if self.use_interview:
            # Memory-mapped Arrow table with zero-copy across processes
            self._interview_data = pq.read_table(self.interview_path,
                                               columns=['AnonymousID', 'McExamDt', 'Interview'],
                                               memory_map=True)
            # Build lookup dictionary for O(1) access with string keys for type safety
            key_cols = ['AnonymousID', 'McExamDt']
            self._interview_lookup = {
                (str(row['AnonymousID']), row['McExamDt'].strftime('%Y-%m-%d')): row['Interview']
                for row in self._interview_data.select(key_cols + ['Interview']).to_pylist()
            }
            self._interview_data = None   # Free mmap handle
        
        self._initialized = True
    
    # Properties for transparent access (maintains existing interface)
    
    @property
    def manifest(self):
        """Lazy loading property for manifest data."""
        if self._manifest is None:
            self._lazy_init()
        return self._manifest
    
    @property
    def mcinfo_ds(self):
        """Lazy loading property for mcinfo dataset."""
        if self._mcinfo_ds is None:
            self._lazy_init()
        return self._mcinfo_ds
    
    @property
    def demographics(self):
        """Lazy loading property for demographics data."""
        if self._demographics is None:
            self._lazy_init()
        return self._demographics
    
    @property
    def demographics_dict(self):
        """Lazy loading property for demographics lookup dictionary."""
        if self._demographics_dict is None:
            self._lazy_init()
        return self._demographics_dict
    
    @property
    def result_data(self):
        """
        Lazy loading property for result data.
        Note: Returns None after optimization - raw data is freed after dictionary building.
        """
        if self.use_result and self._result_data is None:
            self._lazy_init()
        return self._result_data
    
    @property
    def interview_data(self):
        """
        Lazy loading property for interview data.
        Note: Returns None after optimization - raw data is freed after dictionary building.
        """
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


    # --- Footer verification helpers ---

    def _sha256_file(self, path: str, bufsize: int = 1 << 20) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(bufsize)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def _sha256_exam_id_order(self, exam_id_array: pa.ChunkedArray, chunk: int = 1_000_000) -> str:
        """
        Hash the manifest's exam_id sequence (order-sensitive), without materializing everything at once.
        """
        import hashlib
        # Normalize to a single chunk for predictable slicing
        arr = pc.cast(exam_id_array, pa.large_string())
        if len(arr.chunks) > 1:
            arr = pa.chunked_array([pa.concat_arrays(arr.chunks)], type=arr.type)
        h = hashlib.sha256()
        n = len(arr)
        i = 0
        while i < n:
            j = min(i + chunk, n)
            lst = arr.slice(i, j - i).to_pylist()
            h.update(("\n".join(lst)).encode("utf-8"))
            i = j
        return h.hexdigest()


    
    def __len__(self) -> int:
        """Return the number of exams in this split (lazy metadata reading)."""
        if self._n_rows is None:
            import pyarrow.parquet as pq
            self._n_rows = pq.read_metadata(self.manifest_path).num_rows
        return self._n_rows


    # --- Materialized mcinfo helpers ---

    def _get_tests_by_idx(self, idx: int) -> pa.Scalar:
        """
        Random-access the `tests` value for a given positional index, using a tiny LRU of row-groups.
        """
        if self._pf is None or self._rg_offsets is None or self._rg_sizes is None:
            raise RuntimeError("Materialized mcinfo not initialized. Did you pass mcinfo_materialized_path?")

        # Map idx -> (row-group index, offset inside the group)
        rg_idx = bisect_right(self._rg_offsets, idx) - 1
        if rg_idx < 0 or rg_idx >= len(self._rg_sizes):
            raise IndexError(f"Index {idx} out of range for materialized mcinfo.")
        offset = idx - self._rg_offsets[rg_idx]
        if offset < 0 or offset >= self._rg_sizes[rg_idx]:
            raise IndexError(f"Offset {offset} out of range in row-group {rg_idx} (size={self._rg_sizes[rg_idx]}).")

        tbl = self._load_row_group(rg_idx)  # Table with only 'tests' column
        return tbl.column('tests')[offset]

    def _load_row_group(self, rg_idx: int) -> pa.Table:
        """
        Load a single row-group (tests column only) with a tiny LRU cache.
        """
        # LRU lookup
        if rg_idx in self._rg_cache:
            tbl = self._rg_cache.pop(rg_idx)
            self._rg_cache[rg_idx] = tbl  # move to end (most-recent)
            return tbl

        # Miss: read from ParquetFile (tests column only)
        tbl = self._pf.read_row_groups([rg_idx], columns=['tests'])

        # Insert and evict if needed
        self._rg_cache[rg_idx] = tbl
        if len(self._rg_cache) > self.mcinfo_rg_cache_size:
            self._rg_cache.popitem(last=False)  # evict LRU

        return tbl

    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get the exam at the specified index with all required data.
        
        Args:
            idx: Index of the exam to retrieve
            
        Returns:
            Dictionary containing exam data with optional modalities
        """
        # Get manifest row
        manifest_row = self.manifest.slice(idx, 1).to_pydict()
        person_id = manifest_row['person_id'][0]
        exam_date = manifest_row['ExamDate'][0]
        
        # --- mcinfo (tests) retrieval: materialized path (preferred) vs old hive dataset ---
        exam_id = manifest_row['exam_id'][0]

        if self.mcinfo_materialized_path:
            # NEW: row-group random access from materialized Parquet (tests only)
            tests_scalar = self._get_tests_by_idx(idx)  # pa.Scalar
            tests_value = tests_scalar.as_py()
        else:
            # OLD: year + exam_id filter on hive-partitioned dataset (project tests only)
            year = exam_date.year
            mcinfo_filter = (ds.field('year') == year) & (ds.field('exam_id') == exam_id)
            mcinfo_row = self.mcinfo_ds.to_table(filter=mcinfo_filter, columns=['tests'])
            if mcinfo_row.num_rows == 0:
                raise KeyError(f"No exam found with id {exam_id} in year {year}")
            tests_value = mcinfo_row.column('tests')[0].as_py()

        # Look up demographics
        demographics = self.demographics_dict.get(person_id, {})

        # Build result dictionary (transparent to downstream):
        # - Since materialized file has only `tests`, we source identifiers from manifest.
        result = {
            'person_id': person_id,
            'ExamDate': exam_date,
            'exam_id': exam_id,                     # keep for debugging/compat; safe to include
            'tests': tests_value,
            'birth_year': demographics.get('birth_year'),
            'gender': demographics.get('gender'),
        }

        
        # Add optional data using O(1) dictionary lookups with string keys for type safety
        if self.use_result:
            lookup_key = (str(person_id), exam_date.strftime('%Y-%m-%d'))
            if self.use_pretokenized_result:
                # NEW PATH: Add pre-tokenized data as separate fields
                tokenized_data = self._result_lookup.get(lookup_key)
                if tokenized_data:
                    result['result_input_ids'] = tokenized_data['input_ids']
                    result['result_attention_mask'] = tokenized_data['attention_mask']
                else:
                    # FALLBACK: Use empty tokenized data for missing results
                    result['result_input_ids'] = self.EMPTY_RESULT_DATA['input_ids']
                    result['result_attention_mask'] = self.EMPTY_RESULT_DATA['attention_mask']
            else:
                # EXISTING PATH: Add raw text (unchanged)
                result['result_text'] = self._result_lookup.get(lookup_key, "") or ""

        
        if self.use_interview:
            lookup_key = (str(person_id), exam_date.strftime('%Y-%m-%d'))
            result['interview'] = self._interview_lookup.get(lookup_key, "")
        
        return result
    
    def close(self) -> None:
        """Release PyArrow objects and (optionally) shut down Arrow runtime."""
        for name in (
            "_manifest", "_mcinfo_ds",
            "_demographics", "_demographics_dict",
            "_result_lookup", "_interview_lookup",
            # Materialized-mcinfo internals
            "_pf", "_rg_cache", "_rg_sizes", "_rg_offsets", "_mat_footer",
        ):
            setattr(self, name, None)


        # # Only finalize in DataLoader worker processes
        # if torch.utils.data.get_worker_info() is not None:
        #     try:
        #         if hasattr(pa, "runtime") and hasattr(pa.runtime, "finalize"):
        #             pa.runtime.finalize()
        #     except Exception:
        #         pass
        gc.collect()
    
    # Multiprocessing support methods
    
    def __getstate__(self):
        """
        Custom pickling to handle PyArrow objects for multiprocessing.
        Only serializes file paths and configuration, not the actual data.
        """
        state = self.__dict__.copy()
        # Remove all PyArrow objects and computed data
        pyarrow_keys = [
            '_manifest', '_mcinfo_ds', '_demographics', '_demographics_dict',
            '_result_lookup', '_interview_lookup',
            # Materialized mcinfo internals
            '_pf', '_rg_sizes', '_rg_offsets', '_rg_cache', '_mat_footer'
        ]
        for key in pyarrow_keys:
            state[key] = None
        
        # _result_data and _interview_data are already None after _lazy_init optimization
        
        # Reset initialization flag
        state['_initialized'] = False
        
        return state
    
    def __setstate__(self, state):
        """
        Custom unpickling to restore dataset state.
        PyArrow objects will be lazy-loaded when first accessed.
        """
        self.__dict__.update(state)

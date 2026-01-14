# src/downstream/lab_test/model/datamodule.py

import torch
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Union, Iterator, Tuple

# Import base classes from SSL models (relative path)
from src.models import (
    PersonBatchSampler,
    HealthExamDataset, 
    collate_exams
)

from src.models.embedders import CategoricalEmbedder, TextEmbedder


class LabTestPersonBatchSampler(PersonBatchSampler):
    """
    Complete individual sampling for downstream lab test tasks with temporal split filtering.
    
    Unlike SSL training which splits individuals across batches for exact batch sizes,
    this sampler keeps complete individuals together and accepts variable batch sizes.
    
    Applies mode-specific temporal filtering to prevent data leakage:
    - TRAIN: Include exams until first non-TRAIN split
    - VAL: Include TRAIN as history + VAL for loss, stop at TESTF  
    - TEST: Include all exams (TRAIN+VAL as history, TESTF for loss)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        batch_size: int,
        mode: str = 'train',
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize lab test sampler with temporal split filtering.
        
        Args:
            manifest_path: Path to unified manifest parquet
            batch_size: Target batch size (may be smaller for complete individuals)
            mode: Training mode ('train', 'val', 'test') for split filtering
            shuffle: Whether to shuffle persons each epoch
            drop_last: Whether to drop incomplete final batch
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{mode}'")
        
        self.mode = mode
        super().__init__(manifest_path, batch_size, shuffle, drop_last)
        self._skip_summary_logged = False
    
    def _build_person_index_map(self) -> List[Tuple[str, List[int]]]:
        """
        Build person -> exam_indices mapping with mode-specific temporal filtering.
        
        Uses efficient groupby approach with proper safety checks from original implementation.
        
        Returns:
            List of (person_id, filtered_exam_indices) tuples where indices are positional (0-based)
        """
        # Safety checks from original implementation
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        # Read manifest file
        df = pd.read_parquet(self.manifest_path)
        
        # CRITICAL: Reset index to ensure idx corresponds to positional row numbers
        # Manifest files may have custom indices like __index_level_0__ which don't
        # match the expected 0-based positional indices for Dataset.__getitem__()
        df = df.reset_index(drop=True)
        
        # Safety checks from original implementation  
        if 'person_id' not in df.columns:
            raise ValueError(f"Manifest file must contain 'person_id' column: {self.manifest_path}")
        
        if 'split' not in df.columns:
            raise ValueError(f"Manifest file must contain 'split' column for filtering: {self.manifest_path}")
        
        # Efficient groupby approach with filtering
        person_groups = []
        
        for person_id, person_group in df.groupby('person_id'):
            # Get exam indices and splits (already chronologically ordered)
            exam_indices = person_group.index.tolist()
            splits = person_group['split'].tolist()
            
            # Apply mode-specific filtering
            filtered_indices = self._apply_mode_filter(exam_indices, splits)
            
            # Only include individuals with meaningful sequences
            if len(filtered_indices) >= 2:  # Need at least 1 history + 1 target
                person_groups.append((person_id, filtered_indices))
        
        return person_groups
    
    def _apply_mode_filter(self, exam_indices: List[int], splits: List[str]) -> List[int]:
        """
        Apply mode-specific filtering to a person's exam sequence.
        
        Args:
            exam_indices: List of exam indices for this person
            splits: List of split names corresponding to each exam
            
        Returns:
            List of filtered exam indices
        """
        if self.mode == 'train':
            # Include exams until first non-TRAIN split
            for i, split in enumerate(splits):
                if split != 'TRAIN':
                    return exam_indices[:i]
            # All exams are TRAIN - this is fine for train mode
            return exam_indices
        
        elif self.mode == 'val':
            # Must have VAL exams to be valid for val mode
            if 'VAL' not in splits:
                return []  # Skip person - no VAL exams to compute loss on
            
            # Include TRAIN as history + VAL for loss, stop at TESTF
            for i, split in enumerate(splits):
                if split == 'TESTF':
                    return exam_indices[:i]
            
            # No TESTF found, include all exams (TRAIN + VAL)
            return exam_indices
        
        elif self.mode == 'test':
            # Must have TESTF exams to be valid for test mode  
            if 'TESTF' not in splits:
                return []  # Skip person - no TESTF exams to compute loss on
            
            # Include all exams (full history, loss only on TESTF positions)
            return exam_indices
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches with complete individuals only.
        
        Yields:
            List of exam indices for each batch (variable sizes allowed)
        """
        # Shuffle persons if requested
        persons = self.persons.copy()
        if self.shuffle:
            import random
            random.shuffle(persons)
        
        person_idx = 0
        
        # Generate batches until all persons are processed
        skipped_persons = 0
        skipped_total_exams = 0
        skipped_max_exams = 0

        while person_idx < len(persons):
            current_batch = []
            
            # Fill current batch with complete individuals
            while person_idx < len(persons):
                person_id, exam_indices = persons[person_idx]
                slots_available = self.batch_size - len(current_batch)
                
                if len(exam_indices) <= slots_available:
                    # Include entire individual
                    current_batch.extend(exam_indices)
                    person_idx += 1
                else:
                    # Can't fit this individual, end current batch
                    break
            
            # Handle oversized individuals that prevent any progress
            if len(current_batch) == 0:
                if not self._skip_summary_logged:
                    exams = len(persons[person_idx][1])
                    skipped_persons += 1
                    skipped_total_exams += exams
                    if exams > skipped_max_exams:
                        skipped_max_exams = exams
                person_idx += 1
                continue
            
            # Yield batch if valid
            if self.drop_last and len(current_batch) < self.batch_size:
                # Skip incomplete batch if drop_last=True
                break
            else:
                yield current_batch

        if (not self._skip_summary_logged) and skipped_persons > 0:
            print(
                f"Skipping {skipped_persons} persons in mode='{self.mode}' "
                f"(total_exams={skipped_total_exams}, max_exams={skipped_max_exams}, "
                f"batch_size={self.batch_size})"
            )
            self._skip_summary_logged = True
    
    def __repr__(self) -> str:
        """String representation with downstream-specific info."""
        return (f"LabTestPersonBatchSampler(mode='{self.mode}', persons={len(self.persons)}, "
                f"total_exams={self.total_exams}, target_batch_size={self.batch_size}, "
                f"complete_individuals_only=True)")


class LabTestDataset(HealthExamDataset):
    """
    Dataset for downstream lab test prediction tasks.
    
    Preloads all labels into a NumPy array for efficient access.
    Uses index-based label access for scalability and multi-worker compatibility.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        labels_path: Union[str, Path],
        label_order: List[str],
        **kwargs
    ):
        """
        Initialize lab test dataset with NumPy array label preloading.
        
        Args:
            manifest_path: Path to unified manifest parquet  
            labels_path: Path to cleaned labels parquet
            label_order: List of 28 label column names in desired order (from config)
            **kwargs: Additional arguments passed to HealthExamDataset
        """
        manifest_path = Path(manifest_path)
        
        # Initialize parent with explicit manifest path support (v1 behavior)
        super().__init__(
            split_name="lab_test",
            manifest_path=str(manifest_path),
            **kwargs
        )
        
        self.labels_path = Path(labels_path)
        self.label_order = label_order
        
        # Preload all labels into NumPy array
        self._load_labels_array()
    
    def _load_labels_array(self):
        """
        Load labels parquet into ordered float32 NumPy array.
        
        All columns are already cleaned float64, just need to select and convert to float32.
        """
        # Read labels parquet
        labels_df = pd.read_parquet(self.labels_path)
        
        # Verify we have the expected number of rows using Arrow metadata
        manifest_rows = len(self)
        if len(labels_df) != manifest_rows:
            raise ValueError(f"Labels ({len(labels_df)}) and manifest ({manifest_rows}) row counts don't match")
        
        # Lightweight exam_id alignment check (if both sides provide exam_id)
        if manifest_rows > 0 and "exam_id" in labels_df.columns:
            manifest_table = self.manifest  # Trigger lazy load once
            if "exam_id" in manifest_table.column_names:
                sample_indices = {0, manifest_rows - 1}
                if manifest_rows > 2:
                    sample_indices.add(manifest_rows // 2)
                
                for idx in sample_indices:
                    manifest_row = manifest_table.slice(idx, 1).to_pydict()
                    manifest_exam_id = manifest_row["exam_id"][0]
                    labels_exam_id = labels_df.at[idx, "exam_id"]
                    if manifest_exam_id != labels_exam_id:
                        raise ValueError(
                            f"exam_id mismatch at index {idx}: manifest={manifest_exam_id}, labels={labels_exam_id}"
                        )
        
        # Verify all label columns exist
        missing_columns = [col for col in self.label_order if col not in labels_df.columns]
        if missing_columns:
            raise ValueError(f"Missing label columns: {missing_columns}")
        
        # Select label columns in exact order specified by label_order
        # Exclude metadata columns: exam_id, person_id, ExamDate, split
        ordered_df = labels_df[self.label_order]
        
        # Convert to float32 NumPy array (already cleaned float64 data)
        self.labels_arr = ordered_df.astype('float32').to_numpy()
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get single exam with lab test labels as NumPy array.
        
        Returns:
            Dictionary with SSL data + lab_labels as float32 NumPy array
        """
        # Get base SSL data from parent
        ssl_data = super().__getitem__(idx)
        
        # Add lab test labels as NumPy array - direct indexing
        ssl_data['lab_labels'] = self.labels_arr[idx]  # Shape: (28,) float32

        # Add split information
        manifest_row = self.manifest.slice(idx, 1).to_pydict()
        ssl_data['split'] = manifest_row['split'][0]
        
        return ssl_data
    
    def get_label_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded labels for debugging.
        
        Returns:
            Dictionary with label statistics
        """
        return {
            'num_labels': len(self.label_order),
            'label_order': self.label_order,
            'array_shape': self.labels_arr.shape,
            'memory_mb': self.labels_arr.nbytes / 1024**2,
            'dtype': self.labels_arr.dtype,
            'total_nan_count': np.isnan(self.labels_arr).sum(),
            'nan_rate_per_column': np.isnan(self.labels_arr).mean(axis=0).round(3).tolist()
        }
    
    def close(self) -> None:
        """Release PyArrow objects and NumPy array."""
        # Call parent close method
        super().close()
        
        # Clear labels array
        if hasattr(self, 'labels_arr'):
            del self.labels_arr
    
    def __getstate__(self):
        """Custom pickling - exclude labels array (will be reloaded)."""
        state = super().__getstate__()
        # Don't pickle the large NumPy array - will be reloaded on workers
        if 'labels_arr' in state:
            del state['labels_arr']
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - reload labels array."""
        super().__setstate__(state)
        # Reload labels array in worker process
        self._load_labels_array()


def _pad_and_stack_tokens(
    token_lists: List[List[int]],
    attention_lists: List[List[int]],
    pad_id: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length token sequences and stack into tensors.
    Replicates tokenizer's padding behavior for pre-tokenized data.
    
    Args:
        token_lists: List of token ID lists for each sample
        attention_lists: List of attention mask lists for each sample  
        pad_id: Token ID to use for padding
        device: Device for tensor creation
        
    Returns:
        Tuple of (input_ids, attention_mask) tensors
    """
    max_len = max(map(len, token_lists))

    # Pre-allocated Python lists, avoid multiple appends
    padded_tok = [seq + [pad_id] * (max_len - len(seq)) for seq in token_lists]
    padded_att = [seq + [0] * (max_len - len(seq)) for seq in attention_lists]

    # One-time conversion with memory-efficient dtypes
    input_ids = torch.as_tensor(padded_tok, dtype=torch.int32, device=device)
    attn_mask = torch.as_tensor(padded_att, device=device)

    return input_ids, attn_mask

def lab_test_collate_fn(
    batch: List[Dict[str, Any]], 
    code_embedder: CategoricalEmbedder, 
    text_embedder: TextEmbedder, 
    config: Dict[str, Any], 
    mode: str = 'train',
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Collate function for downstream lab test dense target training.
    
    Args:
        batch: List of dictionaries from LabTestDataset
        code_embedder: Embedder for test codes and categorical values
        text_embedder: Embedder for text values and result text
        config: Training configuration dictionary containing task settings
        device: Device for tensors
        
    Returns:
        Dictionary with TabEmbedder inputs + lab test labels for dense training
    """
    
    # Read config for downstream-specific settings
    datamodule_config = config.get('datamodule', {})
    label_processing_config = datamodule_config.get('label_processing', {})
    horizon = label_processing_config.get('horizon', 1)  # Prediction horizon (default: next exam)
    
    # ================================================================
    # CORE DATA PROCESSING (copied from collate_fn.py)
    # ================================================================
    if not batch:
        return {}

    # Probe batch to determine result data type (ONCE, at the top)
    has_result_data = False
    has_pretokenized_result = False

    if batch:  # Safety check
        sample = batch[0]  # Check first sample as representative
        if "result_input_ids" in sample and "result_attention_mask" in sample:
            has_result_data = True
            has_pretokenized_result = True
        elif "result_text" in sample:
            has_result_data = True
            has_pretokenized_result = False

    B = len(batch)
    T_max = max(len(sample["tests"]) for sample in batch)
    
    # Constants
    PAD_CODE = '<PAD>'
    PAD_CAT = '<PAD>'
    PAD_TEXT = '[PAD]'
    
    UNK_CODE = '<UNK>'
    UNK_CAT = '<UNK>'
    
    MASK_CAT_TOKEN = '<MASK>'
    
    # Gender mapping (starting at 1, 0 for missing/unknown)
    gender_to_id = {"M": 1, "F": 2}
    
    # Initialize flat lists for test values
    flat_codes = [PAD_CODE] * (B * T_max)
    flat_cat_values = [PAD_CAT] * (B * T_max)
    flat_num_values = [0.0] * (B * T_max)
    flat_type_ids = [0] * (B * T_max)

    # New optimized text handling
    nonempty_text_values = []  # Only actual text values
    text_indices = []  # (i, j) positions of text values
    
    # Initialize masking tensors directly on the target device
    mask_code = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_num = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_cat = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_text = torch.zeros(B, T_max, dtype=torch.bool, device=device)

    # Type conversion map
    type_to_id = {"PQ": 1, "CD": 2, "CO": 2, "ST": 3}
    
    # Initialize metadata storage
    exam_dates = []
    exam_ages = []  # Exam-level age bins
    exam_genders = []  # Exam-level gender IDs
    
    # Result text storage initialization based on probing
    if has_result_data:
        if has_pretokenized_result:
            result_input_ids = []
            result_attention_masks = []
        else:
            result_texts = [""] * B
    
    # Track segment information (consecutive exams from same person)
    current_person = None
    segment_lengths = []  # Keep as List[int]
    segment_count = 0
    
    # Helper function for age binning
    def age_to_bin_id(age: int) -> int:
        """Convert age to discrete bin ID. 0=missing/invalid, 1=<20, 2=[20,30), ..., 8=>=80"""
        if age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        elif age < 60:
            return 5
        elif age < 70:
            return 6
        elif age < 80:
            return 7
        else:  # age >= 80
            return 8
    
    # Process each sample in the batch
    for i, sample in enumerate(batch):
        person_id = sample["person_id"]
        
        # Update segment information
        if current_person != person_id:
            if segment_count > 0:
                segment_lengths.append(segment_count)
            current_person = person_id
            segment_count = 1
        else:
            segment_count += 1
        
        # Store exam date
        exam_date = sample["ExamDate"]
        if isinstance(exam_date, date):
            # Convert to days since epoch for numerical processing
            epoch = date(1970, 1, 1)
            days_since_epoch = (exam_date - epoch).days
            exam_dates.append(days_since_epoch)
            exam_year = exam_date.year
        else:
            # Handle unexpected format
            exam_dates.append(0)
            exam_year = 0
        
        # Compute exam-level age
        birth_year = sample.get("birth_year", 0)
        if birth_year > 0 and exam_year > birth_year:
            exam_age = exam_year - birth_year
            age_bin_id = age_to_bin_id(exam_age)
        else:
            age_bin_id = 0  # Missing/invalid
        exam_ages.append(age_bin_id)
        
        # Store exam-level gender
        gender = sample.get("gender", "")
        gender_id = gender_to_id.get(gender, 0)
        exam_genders.append(gender_id)

        # Store result data based on data type
        if has_result_data:
            if has_pretokenized_result:
                # NEW PATH: Collect pre-tokenized data
                result_input_ids.append(sample["result_input_ids"])
                result_attention_masks.append(sample["result_attention_mask"])

            else:
                # EXISTING PATH: Store raw text (unchanged)
                result_texts[i] = sample["result_text"]
        
        # Process tests
        tests = sample["tests"]
        for j, test in enumerate(tests):
            if j >= T_max:
                break  # Safety check
            
            # Check for held-out codes and skip if necessary
            code = test.get("code", "")
                
            idx = i * T_max + j
            
            # Process code
            if not code:
                code = UNK_CODE
            flat_codes[idx] = code
            mask_code[i, j] = True
            
            # Process type and corresponding value
            test_type = test.get("type", "")
            type_id = type_to_id.get(test_type, 0)
            flat_type_ids[idx] = type_id
            
            if test_type == "PQ":
                value_num = test.get("value_num")
                if value_num is not None and not np.isnan(value_num):
                    flat_num_values[idx] = value_num
                    mask_num[i, j] = True
            
            elif test_type in ["CD", "CO"]:
                value_cat = test.get("value_cat", "")
                if value_cat:
                    # Combine code and value for categorical embedder
                    flat_cat_values[idx] = f"{code}={value_cat}"
                    mask_cat[i, j] = True
                else:
                    flat_cat_values[idx] = UNK_CAT
            
            elif test_type == "ST":
                value_text = test.get("value_text", "")
                if value_text:
                    nonempty_text_values.append(value_text)
                    text_indices.append((i, j))
                    mask_text[i, j] = True

    # Add the last segment length
    if segment_count > 0:
        segment_lengths.append(segment_count)

    # Process codes
    code_tokens = flat_codes
    code_token_ids = code_embedder.map(code_tokens, device=device)
    code_token_ids = code_token_ids.reshape(B, T_max)
    
    # Process categorical values
    cat_value_tokens = flat_cat_values
    cat_value_ids = code_embedder.map(cat_value_tokens, device=device)
    cat_value_ids = cat_value_ids.reshape(B, T_max)

    # Process numerical values
    num_values = torch.tensor(flat_num_values, dtype=torch.float, device=device).reshape(B, T_max)

    # Process text test values (optimized)
    if nonempty_text_values:
        text_value_encodings = text_embedder.tokenize(nonempty_text_values)
        # Keep original shape: [N_seq, L_text_max]
        text_token_ids = text_value_encodings["input_ids"].to(device)
        text_attention_mask = text_value_encodings["attention_mask"].to(device)
        # Convert text indices to tensor
        text_locations = torch.tensor(text_indices, dtype=torch.long, device=device)
    else:
        # Handle case with no text values
        text_token_ids = torch.empty(0, 0, dtype=torch.long, device=device)
        text_attention_mask = torch.empty(0, 0, dtype=torch.long, device=device)
        text_locations = torch.empty(0, 2, dtype=torch.long, device=device)

    # Process result texts - BRANCHING POINT
    if has_result_data:
        if has_pretokenized_result:
            # NEW PATH: Use pre-tokenized data
            input_ids, attention_mask = _pad_and_stack_tokens(
                result_input_ids, 
                result_attention_masks,
                text_embedder.tokenizer.pad_token_id, 
                device
            )
        else:
            # UPDATED PATH: Tokenize raw text, keep missing samples empty for padding
            pad_token_id = text_embedder.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Text tokenizer must define a pad_token_id for batching.")

            token_lists: List[List[int]] = [[] for _ in range(B)]
            attention_lists: List[List[int]] = [[] for _ in range(B)]
            nonempty_indices: List[int] = []
            nonempty_texts: List[str] = []

            for idx, text in enumerate(result_texts):
                normalized_text = text or ""
                if normalized_text.strip():
                    nonempty_indices.append(idx)
                    nonempty_texts.append(normalized_text)

            if nonempty_indices:
                encoded = text_embedder.tokenize(nonempty_texts)
                ids_batch = encoded["input_ids"]
                mask_batch = encoded["attention_mask"]

                for j, sample_idx in enumerate(nonempty_indices):
                    seq_ids = ids_batch[j].cpu().tolist()
                    seq_mask = mask_batch[j].cpu().tolist()
                    token_lists[sample_idx] = [int(v) for v in seq_ids]
                    attention_lists[sample_idx] = [int(v) for v in seq_mask]

            input_ids, attention_mask = _pad_and_stack_tokens(
                token_lists,
                attention_lists,
                pad_token_id,
                device
            )

    # Type IDs and mask
    type_ids = torch.tensor(flat_type_ids, dtype=torch.long, device=device).reshape(B, T_max)
    mask_type = (type_ids != 0)  # Already on device because type_ids is
    
    # Prepare masks for each modality (all already on device)
    modality_masks = {
        "mask_code": mask_code,
        "mask_num": mask_num,
        "mask_cat": mask_cat,
        "mask_text": mask_text,
        "mask_type": mask_type
    }
    
    # Prepare metadata
    metadata = {
        "exam_dates": torch.tensor(exam_dates, dtype=torch.long, device=device),
        "exam_ages": torch.tensor(exam_ages, dtype=torch.long, device=device),
        "exam_genders": torch.tensor(exam_genders, dtype=torch.long, device=device),
        "segment_lengths": segment_lengths,  # Kept as List[int]
        "B": B,  # Batch size
        "T_max": T_max  # Maximum number of tests per exam in this batch
    }

    # ================================================================
    # SSL TASK FIELDS (disabled states)
    # ================================================================

    # MCM: no masking applied
    mcm_inputs = cat_value_ids.clone()  # Use original categorical values
    mcm_labels = torch.full_like(cat_value_ids, -100)  # Ignore all positions
    
    # CVR: no candidates
    cvr_mask = torch.full_like(cat_value_ids, -100)  # Ignore all positions
    cvr_labels = torch.empty(0, dtype=torch.long, device=device)
    cvr_true_ids = torch.empty(0, 0, dtype=torch.long, device=device)
    cvr_true_attention_masks = torch.empty(0, 0, dtype=torch.bool, device=device)
    
    # MCC: no candidates
    mcc_mask = torch.zeros_like(mask_num, dtype=torch.bool)  # No positions selected
    opts_raw = torch.empty(0, 0, dtype=torch.float32, device=device)
    mcc_labels = torch.empty(0, dtype=torch.long, device=device)
    
    # MLM: no masking
    result_mlm_labels = torch.full_like(input_ids, -100)  # Ignore all tokens
    
    # ================================================================
    # LAB TEST LABEL PROCESSING (new for downstream)
    # ================================================================

    # Extract split information from batch
    splits_batch = [sample['split'] for sample in batch]
    
    # Extract lab_labels from batch (NumPy arrays)
    lab_labels_batch = [sample['lab_labels'] for sample in batch]
    
    # Process lab labels with horizon shifting and mode-specific masking
    processed_lab_labels = process_lab_labels(
        lab_labels_batch, 
        splits_batch,
        segment_lengths, 
        mode, 
        horizon, 
        device
    )

    # ================================================================
    # ASSEMBLE OUTPUT
    # ================================================================

    return {
        # Core TabEmbedder inputs
        "code_ids": code_token_ids,
        "cat_value_ids": cat_value_ids,
        "num_values": num_values,
        "text_token_ids": text_token_ids,
        "text_attention_mask": text_attention_mask,
        "text_locations": text_locations,
        "result_input_ids": input_ids,
        "result_attention_mask": attention_mask,
        "type_ids": type_ids,
        
        # SSL task fields (disabled)
        "mcm_inputs": mcm_inputs,
        "mcm_labels": mcm_labels,
        "cvr_mask": cvr_mask,
        "cvr_labels": cvr_labels,
        "cvr_true_ids": cvr_true_ids,
        "cvr_true_attention_masks": cvr_true_attention_masks,
        "mcc_mask": mcc_mask,
        "opts_raw": opts_raw,
        "mcc_labels": mcc_labels,
        "result_mlm_labels": result_mlm_labels,
        
        # Modality masks and metadata
        **modality_masks,
        **metadata,
        
        # Lab test labels for dense training
        **processed_lab_labels
    }

def process_lab_labels(
    lab_labels_batch: List[np.ndarray],
    splits_batch: List[str],  # Split information per exam
    segment_lengths: List[int], 
    mode: str,
    horizon: int,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Process lab test labels with horizon shifting and mode-specific masking.
    
    Args:
        lab_labels_batch: List of (28,) NumPy arrays from dataset
        splits_batch: List of split names per exam ('TRAIN', 'VAL', 'TESTF') 
        segment_lengths: Number of exams per person
        mode: 'train', 'val', or 'test' - affects which positions to predict
        horizon: Prediction horizon (t+horizon), default 1 for next exam
        device: Device for tensors
        
    Returns:
        Dictionary with lab_targets and lab_masks tensors
    """
    
    # Stack into (B_total, 28) tensor
    y = torch.from_numpy(np.stack(lab_labels_batch)).float().to(device)
    B_total = y.shape[0]
    
    # Apply horizon shifting per individual
    y_shift = y.clone()
    start = 0
    
    for L in segment_lengths:
        end = start + L
        
        if L > horizon:
            # Shift labels by horizon positions
            y_shift[start:end-horizon] = y[start+horizon:end]
            y_shift[end-horizon:end] = float('nan')  # Last h positions become NaN
        else:
            # Not enough exams for horizon prediction
            y_shift[start:end] = float('nan')
        
        start = end
    
    # Apply mode-specific masking
    if mode == 'train':
        # Standard training: use all valid shifted positions
        mask = ~torch.isnan(y_shift)
        
    elif mode == 'val':
        # Validation: only compute loss for predictions targeting VAL exams
        # We input history (TRAIN) + VAL exams, but only compute loss on VAL targets
        val_mask = torch.zeros_like(y_shift, dtype=torch.bool, device=device)
        start = 0
        
        for L in segment_lengths:
            end = start + L
            
            # Check each position's target exam split
            for t in range(L):
                target_pos = t + horizon
                if target_pos < L:  # Valid prediction exists
                    target_split = splits_batch[start + target_pos]
                    if target_split == 'VAL':
                        # Only compute loss for predictions targeting VAL exams
                        val_mask[start + t] = ~torch.isnan(y_shift[start + t])
            
            start = end
        mask = val_mask
        
    elif mode == 'test':
        # Test: only predict the last position corresponding to TESTF split per individual
        test_mask = torch.zeros_like(y_shift, dtype=torch.bool, device=device)
        start = 0
        
        for L in segment_lengths:
            end = start + L
            
            # Step 1: Find the last position that targets a TESTF exam (temporal filtering)
            last_testf_pos = None
            for t in range(L):
                target_pos = t + horizon
                if target_pos < L and splits_batch[start + target_pos] == 'TESTF':
                    last_testf_pos = t  # Keep updating to get the LAST TESTF position
            
            # Step 2: Apply per-task NaN filtering for the identified position (content filtering) 
            if last_testf_pos is not None:
                position_idx = start + last_testf_pos
                # CRITICAL: Only set mask=True for non-NaN lab values at this position
                # This prevents NaN values from reaching metrics computation
                test_mask[position_idx] = ~torch.isnan(y_shift[position_idx])
            
            start = end
        mask = test_mask
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train', 'val', or 'test'")
    
    # Ensure mask is on correct device (explicit for clarity)
    mask = mask.to(device)
    
    return {
        'lab_targets': y_shift,  # (B_total, 28)
        'lab_masks': mask,       # (B_total, 28) - True where we compute loss
    }

"""
Profiled collate_exams: strictly behavior-identical to collate_fn.collate_exams,
with side-channel timing & counting for research profiling.

- No changes to inputs, outputs, shapes, or dtypes.
- Uses exactly the same encoding & masking logic as the original.
- Profiling can be toggled via enable_profiling and warmup gating.

JSONL payload (per worker):
  {
    "ts": ...,
    "wid": ...,
    "pid": ...,
    "simple": {...timers...},              # seconds
    "detailed": {"text": {...}, "mcc": {...}, ...},   # seconds
    "simple_counters": {...},              # dimensionless
    "detailed_counters": {"text": {...}, "mcc": {...}}
  }
"""

import os
import json
import time
from datetime import date
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import get_worker_info

# Import helpers from the original implementation
from src.models.collate_fn import (
    _load_held_out_codes,
    _pad_and_stack_tokens,
    _sample_mcc_options,
)
# -----------------------------------------------------------------------------
# Helpers (directly copied from )
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Per-worker profiler
# -----------------------------------------------------------------------------

class CollateProfiler:
    def __init__(self):
        # Timers (seconds)
        self.simple_metrics = defaultdict(list)                      # e.g., total_collate, category_mapping
        self.detailed_metrics = defaultdict(lambda: defaultdict(list))  # e.g., text/tokenization, mcc/total

        # Counters (dimensionless)
        self.simple_counters = defaultdict(list)
        self.detailed_counters = defaultdict(lambda: defaultdict(list))

        # Global text stats for duplication signals (single-worker only)
        self.text_stats = {
            "unique_texts": set(),
            "text_occurrences": Counter(),
            "batch_duplicates": []
        }

        # MCC per-batch stats
        self.mcc_stats = {
            "cells_masked": [],
            "retry_counts": [],
            "cells_with_retries": []
        }

        # Warmup gate: only record when _batch_seen >= start_after_batches
        self.start_after_batches = 0
        self._batch_seen = 0

    def should_log(self) -> bool:
        return self._batch_seen >= self.start_after_batches


collate_profiler = CollateProfiler()


def _profile_path(prefix: str) -> Tuple[str, int, int, int]:
    wi = get_worker_info()
    wid = wi.id if wi else -1
    pid = os.getpid()
    outdir = os.environ.get("DL_PROFILE_DIR", "./dl_profile_logs")
    run = os.environ.get("DL_PROFILE_RUN", "run")
    flush_every = max(1, int(os.environ.get("DL_PROFILE_FLUSH_EVERY", "1")))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{prefix}-{run}-w{wid}-p{pid}.jsonl")
    return path, wid, pid, flush_every


def _flush_collate_metrics_to_file(force: bool = False) -> None:
    path, wid, pid, flush_every = _profile_path(prefix="collate")
    if (collate_profiler._batch_seen % flush_every) != 0 and not force:
        return

    payload = {
        "ts": time.time(),
        "wid": wid,
        "pid": pid,
        "simple": {k: list(v) for k, v in collate_profiler.simple_metrics.items()},
        "detailed": {
            cat: {m: list(vals) for m, vals in d.items()}
            for cat, d in collate_profiler.detailed_metrics.items()
        },
        "simple_counters": {k: list(v) for k, v in collate_profiler.simple_counters.items()},
        "detailed_counters": {
            cat: {m: list(vals) for m, vals in d.items()}
            for cat, d in collate_profiler.detailed_counters.items()
        },
        "text_stats": {
            "unique_texts_total": len(collate_profiler.text_stats["unique_texts"]),
            "last_batch_duplicates": (
                collate_profiler.text_stats["batch_duplicates"][-1]
                if collate_profiler.text_stats["batch_duplicates"] else 0
            )
        },
        "mcc_stats": {
            "cells_masked": list(collate_profiler.mcc_stats['cells_masked']),
            "retry_counts": list(collate_profiler.mcc_stats['retry_counts']),
            "cells_with_retries": list(collate_profiler.mcc_stats['cells_with_retries']),
        }
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    # reset small buffers to avoid unbounded growth
    collate_profiler.simple_metrics.clear()
    collate_profiler.detailed_metrics.clear()
    collate_profiler.simple_counters.clear()
    collate_profiler.detailed_counters.clear()
    collate_profiler.text_stats["batch_duplicates"].clear()
    collate_profiler.mcc_stats['cells_masked'].clear()
    collate_profiler.mcc_stats['retry_counts'].clear()
    collate_profiler.mcc_stats['cells_with_retries'].clear()


# -----------------------------------------------------------------------------
# Main (behavior-identical) collate with profiling
# -----------------------------------------------------------------------------

def profiled_collate_exams(
    batch: List[Dict],
    code_embedder: Any,
    text_embedder: Any,
    config: dict,
    device: str = "cpu",
    enable_profiling: bool = True,
    detailed_mcc: bool = True,      # kept for parity; not altering behavior
    detailed_text: bool = True,     # kept for parity; not altering behavior
    start_logging_after_batches: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Exactly mirrors src.models.collate_fn.collate_exams, while recording timers & counters.
    """

    total_start = time.perf_counter()
    if not batch:
        return {}

    # Warmup gate for this worker
    collate_profiler.start_after_batches = int(start_logging_after_batches or 0)

    training_config = config.get("training", {})
    # MCC enabled flag decides p_mcc (skip if disabled)
    mcc_config = training_config.get("mcc", {})
    mcc_enabled = mcc_config.get("enabled", True)

    p_cvr = float(training_config.get("p_cvr", 0.20))
    p_mcm = float(training_config.get("p_mcm", 0.15))
    p_mlm = float(training_config.get("p_mlm", 0.15))
    p_mcc = float(training_config.get("p_mcc", 0.20)) if mcc_enabled else 0.0

    # Probe result data type (exactly as original)
    has_result_data = False
    has_pretokenized_result = False
    if batch:
        sample = batch[0]
        if ("result_input_ids" in sample) and ("result_attention_mask" in sample):
            has_result_data = True
            has_pretokenized_result = True
        elif "result_text" in sample:
            has_result_data = True
            has_pretokenized_result = False

    # Held-out codes
    use_held_out_codes = training_config.get("use_held_out_codes", False)
    held_out_codes_path = training_config.get("held_out_codes_path", None)
    held_out_codes_set: set = _load_held_out_codes(held_out_codes_path) if use_held_out_codes else set()
    held_out_stats = Counter()
    total_held_out_cells = 0

    # Batch shapes: T_max = max tests length in this batch
    B = len(batch)
    T_max = max(len(sample["tests"]) for sample in batch)

    # Constants
    PAD_CODE = "<PAD>"
    PAD_CAT = "<PAD>"
    PAD_TEXT = "[PAD]"
    UNK_CODE = "<UNK>"
    UNK_CAT = "<UNK>"
    MASK_CAT_TOKEN = "<MASK>"

    gender_to_id = {"M": 1, "F": 2}

    # Flat storage
    flat_codes = [PAD_CODE] * (B * T_max)
    flat_cat_values = [PAD_CAT] * (B * T_max)
    flat_num_values = [0.0] * (B * T_max)
    flat_type_ids = [0] * (B * T_max)

    # Text cells
    nonempty_text_values = []
    text_indices = []

    # Masks on device
    mask_code = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_num  = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_cat  = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_text = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    
    # Type conversion map
    type_to_id = {"PQ": 1, "CD": 2, "CO": 2, "ST": 3}
    
    # Metadata
    exam_dates = []
    exam_ages = []
    exam_genders = []

    # Result storage (by probing)
    if has_result_data:
        if has_pretokenized_result:
            result_input_ids = []
            result_attention_masks = []
        else:
            result_texts = [""] * B
    
    current_person = None
    segment_lengths = []
    segment_count = 0

    # Helper: age->bin id
    def age_to_bin_id(age: int) -> int:
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

    # -------------------------
    # Flatten loop
    # -------------------------
    t_flat = time.perf_counter()

    for i, sample in enumerate(batch):
        person_id = sample["person_id"]
        
        if current_person != person_id:
            if segment_count > 0:
                segment_lengths.append(segment_count)
            current_person = person_id
            segment_count = 1
        else:
            segment_count += 1

        # exam date -> days since epoch + year
        exam_date = sample["ExamDate"]
        if isinstance(exam_date, date):
            epoch = date(1970, 1, 1)
            days_since_epoch = (exam_date - epoch).days
            exam_dates.append(days_since_epoch)
            exam_year = exam_date.year
        else:
            exam_dates.append(0)
            exam_year = 0

        # exam age bin
        birth_year = sample.get("birth_year", 0)
        if birth_year > 0 and exam_year > birth_year:
            exam_age = exam_year - birth_year
            age_bin_id = age_to_bin_id(exam_age)
        else:
            age_bin_id = 0
        exam_ages.append(age_bin_id)

        # gender id
        gender = sample.get("gender", "")
        gender_id = gender_to_id.get(gender, 0)
        exam_genders.append(gender_id)

        # result storage
        if has_result_data:
            if has_pretokenized_result:
                result_input_ids.append(sample["result_input_ids"])
                result_attention_masks.append(sample["result_attention_mask"])
            else:
                result_texts[i] = sample["result_text"]

        # tests
        tests = sample["tests"]
        for j, test in enumerate(tests):
            if j >= T_max:
                break
                
            code = test.get("code", "")
            if use_held_out_codes and code in held_out_codes_set:
                held_out_stats[code] += 1
                total_held_out_cells += 1
                continue

            idx = i * T_max + j
            
            if not code:
                code = UNK_CODE
            flat_codes[idx] = code
            mask_code[i, j] = True

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

    if segment_count > 0:
        segment_lengths.append(segment_count)

    if enable_profiling and collate_profiler.should_log():
        collate_profiler.detailed_metrics["tests"]["flattening"].append(time.perf_counter() - t_flat)
        collate_profiler.detailed_counters["text"]["num_texts"].append(len(nonempty_text_values))
        collate_profiler.detailed_counters["text"]["num_unique"].append(len(set(nonempty_text_values)))

    # -------------------------
    # Map codes & categorical values
    # -------------------------
    t0 = time.perf_counter()
    code_tokens = flat_codes
    code_token_ids = code_embedder.map(code_tokens, device=device)
    code_token_ids = code_token_ids.reshape(B, T_max)
    
    cat_value_tokens = flat_cat_values
    cat_value_ids = code_embedder.map(cat_value_tokens, device=device)
    cat_value_ids = cat_value_ids.reshape(B, T_max)
    if enable_profiling and collate_profiler.should_log():
        collate_profiler.simple_metrics["category_mapping"].append(time.perf_counter() - t0)

    # -------------------------
    # MCM
    # -------------------------
    t0 = time.perf_counter()
    mask_token_id = code_embedder.get_token_id(MASK_CAT_TOKEN)
    
    mcm_probs = torch.full((B, T_max), p_mcm, device=device)
    mcm_probs = mcm_probs * mask_cat
    
    is_mcm = torch.bernoulli(mcm_probs).bool()
    
    mcm_inputs = cat_value_ids.clone()
    mcm_labels = torch.full_like(cat_value_ids, -100)
    
    mcm_labels[is_mcm] = cat_value_ids[is_mcm]
    mcm_inputs[is_mcm] = mask_token_id
    if enable_profiling and collate_profiler.should_log():
        collate_profiler.simple_metrics["mcm_masking"].append(time.perf_counter() - t0)

    # -------------------------
    # Numerical values
    # -------------------------
    num_values = torch.tensor(flat_num_values, dtype=torch.float32, device=device).reshape(B, T_max)

    # -------------------------
    # MCC
    # -------------------------
    if mcc_enabled and mask_num.any():
        t_total = time.perf_counter()
        
        mcc_probs = torch.full((B, T_max), p_mcc, device=device)
        mcc_probs = mcc_probs * mask_num
        
        mcc_mask = torch.bernoulli(mcc_probs).bool()

        if mcc_mask.any():
            t_samp = time.perf_counter()
            
            opts_raw_list = []
            mcc_labels_list = []
            
            masked_positions = torch.nonzero(mcc_mask)  # [n,2]
            
            for batch_idx, cell_idx in masked_positions:
                true_value = num_values[batch_idx, cell_idx].clone()
                opts_raw, mcc_lab = _sample_mcc_options(true_value, mcc_config)
                opts_raw_list.append(opts_raw)
                mcc_labels_list.append(mcc_lab)
                
            opts_raw = torch.stack(opts_raw_list)
            mcc_labels = torch.tensor(mcc_labels_list, dtype=torch.long, device=device)
            
            num_values[mcc_mask] = -999.0
            
            if enable_profiling and collate_profiler.should_log():
                collate_profiler.detailed_metrics["mcc"]["sampling_loop"].append(time.perf_counter() - t_samp)
                collate_profiler.detailed_counters["mcc"]["num_cells"].append(int(masked_positions.size(0)))
        else:
            K = mcc_config.get("K", 5)
            opts_raw = torch.empty(0, K, dtype=torch.float32, device=device)
            mcc_labels = torch.empty(0, dtype=torch.long, device=device)

        if enable_profiling and collate_profiler.should_log():
            collate_profiler.detailed_metrics["mcc"]["total"].append(time.perf_counter() - t_total)
    else:
        K = mcc_config.get("K", 5)
        mcc_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        opts_raw = torch.empty(0, K, dtype=torch.float32, device=device)
        mcc_labels = torch.empty(0, dtype=torch.long, device=device)

    # -------------------------
    # Text test values
    # -------------------------
    if nonempty_text_values:
        t0 = time.perf_counter()
        
        text_value_encodings = text_embedder.tokenize(nonempty_text_values)
        text_token_ids = text_value_encodings["input_ids"].to(device)
        text_attention_mask = text_value_encodings["attention_mask"].to(device)
        text_locations = torch.tensor(text_indices, dtype=torch.long, device=device)
        
        if enable_profiling and collate_profiler.should_log():
            collate_profiler.detailed_metrics["text"]["tokenization"].append(time.perf_counter() - t0)
            # global duplication stats (single-worker hint)
            for s in nonempty_text_values:
                collate_profiler.text_stats["text_occurrences"][s] += 1
            collate_profiler.text_stats["unique_texts"].update(nonempty_text_values)
            collate_profiler.text_stats["batch_duplicates"].append(
                len(nonempty_text_values) - len(set(nonempty_text_values))
            )
    else:
        text_token_ids = torch.empty(0, 0, dtype=torch.long, device=device)
        text_attention_mask = torch.empty(0, 0, dtype=torch.long, device=device)
        text_locations = torch.empty(0, 2, dtype=torch.long, device=device)

    # -------------------------
    # CVR (on text cells)
    # -------------------------
    cvr_probs = torch.full((B, T_max), p_cvr, device=device)
    cvr_probs = cvr_probs * mask_text
    
    is_cvr = torch.bernoulli(cvr_probs).bool()

    cvr_mask = torch.full((B, T_max), -100, dtype=torch.long, device=device)
    cvr_mask[mask_text] = 0
    cvr_mask[is_cvr] = 1

    cvr_true_ids = []
    cvr_true_attention_masks = []
    cvr_labels = []
    

    cvr_mask_in_text_seqs = torch.zeros(len(text_locations), dtype=torch.bool, device=device)
    for i, (batch_idx, cell_idx) in enumerate(text_locations):
        if is_cvr[batch_idx, cell_idx]:
            cvr_mask_in_text_seqs[i] = True
            cvr_true_ids.append(text_token_ids[i].clone())
            cvr_true_attention_masks.append(text_attention_mask[i].clone())

    for i in torch.nonzero(cvr_mask_in_text_seqs).squeeze(-1):
        valid_len = int(text_attention_mask[i].sum())
        text_token_ids[i, :valid_len].fill_(text_embedder.tokenizer.mask_token_id)

    # permute candidates and labels
    n_cvr = len(cvr_true_ids)
    if n_cvr > 0:
        perm_indices = torch.randperm(n_cvr).tolist()
        
        cvr_true_ids = [cvr_true_ids[i] for i in perm_indices]
        cvr_true_attention_masks = [cvr_true_attention_masks[i] for i in perm_indices]
        
        cvr_labels_list = [0] * n_cvr
        for new_pos, old_pos in enumerate(perm_indices):
            cvr_labels_list[old_pos] = new_pos
            
        cvr_labels = torch.tensor(cvr_labels_list, dtype=torch.long, device=device)
    else:
        cvr_labels = torch.empty(0, dtype=torch.long, device=device)

    if n_cvr > 0:
        cvr_true_ids = torch.stack(cvr_true_ids)  # [n_cvr, L_cvr]
        cvr_true_attention_masks = torch.stack(cvr_true_attention_masks)  # [n_cvr, L_cvr]
    else:
        cvr_true_ids = torch.empty(0, 0, dtype=torch.long, device=device)
        cvr_true_attention_masks = torch.empty(0, 0, device=device)
        

    # -------------------------
    # Result text (pretokenized or raw + MLM)
    # -------------------------
    if has_result_data:
        if has_pretokenized_result:
            input_ids, attention_mask = _pad_and_stack_tokens(
                result_input_ids,
                result_attention_masks,
                text_embedder.tokenizer.pad_token_id,
                device
            )
        else:
            result_encodings = text_embedder.tokenize(result_texts)
            input_ids = result_encodings["input_ids"].to(device, dtype=torch.int32)
            attention_mask = result_encodings["attention_mask"].to(device)

    # MLM
    mlm_labels = input_ids.clone().to(dtype=torch.long)
    probability_matrix = torch.full(input_ids.shape, p_mlm, device=device)

    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, seq in enumerate(input_ids.tolist()):
        seq_mask = text_embedder.tokenizer.get_special_tokens_mask(
            seq, already_has_special_tokens=True
        )
        special_tokens_mask[i, :len(seq_mask)] = torch.tensor(
            seq_mask, dtype=torch.bool, device=device
        )[:special_tokens_mask.size(1)]

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    mlm_labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = text_embedder.tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(text_embedder.tokenizer), input_ids.shape, dtype=torch.int32, device=device)
    input_ids[indices_random] = random_words[indices_random]

    # -------------------------
    # Type ids & mask
    # -------------------------
    type_ids = torch.tensor(flat_type_ids, dtype=torch.long, device=device).reshape(B, T_max)
    mask_type = (type_ids != 0)

    # -------------------------
    # Modality masks & metadata
    # -------------------------
    modality_masks = {
        "mask_code": mask_code,
        "mask_num": mask_num,
        "mask_cat": mask_cat,
        "mask_text": mask_text,
        "mask_type": mask_type,
    }
    
    metadata = {
        "exam_dates": torch.tensor(exam_dates, dtype=torch.long, device=device),
        "exam_ages": torch.tensor(exam_ages, dtype=torch.long, device=device),
        "exam_genders": torch.tensor(exam_genders, dtype=torch.long, device=device),
        "segment_lengths": segment_lengths,
        "B": B,
        "T_max": T_max,
    }
    
    if use_held_out_codes:
        metadata["held_out_cells_count"] = total_held_out_cells
        metadata["held_out_codes_in_batch"] = list(held_out_stats.keys())

    # -------------------------
    # Outputs
    # -------------------------
    outputs = {
        # categorical
        "code_ids": code_token_ids,
        "cat_value_ids": cat_value_ids,
        "mcm_inputs": mcm_inputs,
        "mcm_labels": mcm_labels,

        # numerical
        "num_values": num_values,

        # MCC
        "mcc_mask": mcc_mask,
        "opts_raw": opts_raw,
        "mcc_labels": mcc_labels,

        # sparse text (cell-level)
        "text_token_ids": text_token_ids,
        "text_attention_mask": text_attention_mask,
        "text_locations": text_locations,

        # result text + MLM
        "result_input_ids": input_ids,
        "result_attention_mask": attention_mask,
        "result_mlm_labels": mlm_labels,

        # Type information
        "type_ids": type_ids,

        # modality masks
        **modality_masks,

        # CVR
        "cvr_mask": cvr_mask,
        "cvr_labels": cvr_labels,
        "cvr_true_ids": cvr_true_ids,
        "cvr_true_attention_masks": cvr_true_attention_masks,

        # metadata
        **metadata,
    }

    # -------------------------
    # Profiling: total + flush
    # -------------------------
    if enable_profiling and collate_profiler.should_log():
        collate_profiler.simple_metrics["total_collate"].append(time.perf_counter() - total_start)
    
    collate_profiler._batch_seen += 1
    
    if enable_profiling and collate_profiler.should_log():
        _flush_collate_metrics_to_file(force=False)


    return outputs


# -----------------------------------------------------------------------------
# In-process summaries (single-worker convenience; does not affect behavior)
# -----------------------------------------------------------------------------

def get_collate_profiling_summary() -> Dict[str, Dict[str, float]]:
    """
    Return timers summary (namespaced) for single-worker runs.
    """
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in collate_profiler.simple_metrics.items():
        if not vals: continue
        arr = np.asarray(vals, dtype=float)
        out[f"collate_{k}"] = {
            "mean_ms": float(arr.mean() * 1000),
            "std_ms": float(arr.std() * 1000),
            "min_ms": float(arr.min() * 1000),
            "max_ms": float(arr.max() * 1000),
            "p50_ms": float(np.percentile(arr, 50) * 1000),
            "p95_ms": float(np.percentile(arr, 95) * 1000),
            "p99_ms": float(np.percentile(arr, 99) * 1000),
            "count": int(arr.size),
        }
    for cat, d in collate_profiler.detailed_metrics.items():
        for m, vals in d.items():
            if not vals: continue
            arr = np.asarray(vals, dtype=float)
            out[f"collate_{cat}_{m}"] = {
                "mean_ms": float(arr.mean() * 1000),
                "std_ms": float(arr.std() * 1000),
                "min_ms": float(arr.min() * 1000),
                "max_ms": float(arr.max() * 1000),
                "p50_ms": float(np.percentile(arr, 50) * 1000),
                "p95_ms": float(np.percentile(arr, 95) * 1000),
                "p99_ms": float(np.percentile(arr, 99) * 1000),
                "count": int(arr.size),
            }
    # optional duplication hints (single-worker only)
    if collate_profiler.text_stats["unique_texts"]:
        total_occ = sum(collate_profiler.text_stats["text_occurrences"].values())
        uniq = len(collate_profiler.text_stats["unique_texts"])
        out["text_unique_count"] = uniq
        out["text_total_occurrences"] = total_occ
        out["text_duplication_rate"] = 1.0 - (uniq / max(1, total_occ))
    return out


def get_collate_counter_summary() -> Dict[str, Dict[str, float]]:
    """
    Return counters summary (namespaced) for single-worker runs.
    """
    out: Dict[str, Dict[str, float]] = {}
    def stat(arrf: np.ndarray) -> Dict[str, float]:
        return {
            "sum": float(arrf.sum()),
            "mean": float(arrf.mean()),
            "p50": float(np.percentile(arrf, 50)),
            "p95": float(np.percentile(arrf, 95)),
            "p99": float(np.percentile(arrf, 99)),
            "max": float(arrf.max()),
            "count": int(arrf.size),
        }
    for k, vals in collate_profiler.simple_counters.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            out[f"collate_{k}_count"] = stat(arr)
    for cat, d in collate_profiler.detailed_counters.items():
        for m, vals in d.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size:
                out[f"collate_{cat}_{m}_count"] = stat(arr)
    return out

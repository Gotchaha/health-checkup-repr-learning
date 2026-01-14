# src/models/collate_fn.py

import torch
import warnings
import random
import yaml
import logging
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Union, Any, Set
import numpy as np
from datetime import date
from collections import Counter
from src.models.embedders.CategoricalEmbedder import CategoricalEmbedder
from src.models.embedders.TextEmbedder import TextEmbedder

# Set up logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# MCC Utility Functions
# -----------------------------------------------------------


def _raw_to_norm(x: torch.Tensor) -> torch.Tensor:
    """Map real numbers to (-1,1); safe for fp16/bf16."""
    return x / (1.0 + x.abs())

def _norm_to_raw(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse mapping; clamp to avoid division by zero."""
    return v / (1.0 - v.abs().clamp_max(1.0 - eps))


def _sample_mcc_options_batched(
    true_values: torch.Tensor,  # shape: [n_mcc], dtype=float32 on CPU
    mcc_config: dict,
    rng: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized MCC candidate generator.
    Semantics: same as per-cell version (mixture of small/medium Gaussian + uniform-with-gap in normalized space),
    but without Python loops/retries; strictly reproducible if 'rng' is given.

    Returns:
        opts_raw:   Tensor[n_mcc, K]   (float32, raw domain)
        gold_idx:   Tensor[n_mcc]      (long, index of the true value per row after shuffle)
    """
    device = true_values.device
    dtype  = true_values.dtype
    n = true_values.numel()
    K = int(mcc_config.get('K', 5))
    if n == 0:
        return torch.empty(0, K, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.long, device=device)
    if K < 2:
        raise ValueError("mcc.K must be >= 2")

    noise_cfg   = mcc_config.get('noise', {})
    scales      = noise_cfg.get('gaussian_scales', [0.05, 0.20])
    mix_probs   = noise_cfg.get('mix_probs', [0.5, 0.35, 0.15])
    min_dist    = float(noise_cfg.get('large_min_norm_dist', 0.3))
    tau_norm    = float(mcc_config.get('dedup_tau_norm', 1e-6))  # tiny jitter to avoid exact ties with the true

    # Normalize configs
    mix_probs_t = torch.as_tensor(mix_probs, dtype=dtype, device=device)
    mix_probs_t = mix_probs_t / mix_probs_t.sum()
    cum_mix     = torch.cumsum(mix_probs_t, dim=0)  # [3]
    s_small     = float(scales[0])
    s_medium    = float(scales[1])

    # True values in normalized space
    v_true = _raw_to_norm(true_values).clamp(-0.999, 0.999)        # [n]
    v      = v_true.unsqueeze(1).expand(n, K - 1)                  # [n, K-1]

    # Effective min distance per row to guarantee non-empty support
    # d_eff <= 1 - |v_true| - eps
    eps_border = 1e-6
    d_scalar   = torch.full((n, 1), min_dist, dtype=dtype, device=device)
    max_d      = (1.0 - v_true.abs()).unsqueeze(1) - eps_border
    d_eff      = torch.minimum(d_scalar, torch.clamp(max_d, min=0.0))  # [n,1], >=0

    # Sample noise category per candidate: 0=small, 1=medium, 2=uniform-with-gap
    U = torch.rand(n, K - 1, generator=rng, device=device)
    cats = torch.bucketize(U, cum_mix)  # [n, K-1] in {0,1,2}
    is_small  = (cats == 0)
    is_medium = (cats == 1)
    is_uniform= (cats == 2)

    # Small/medium Gaussian eps (vectorized)
    eps_small  = torch.randn(n, K - 1, generator=rng, device=device, dtype=dtype) * s_small
    eps_medium = torch.randn(n, K - 1, generator=rng, device=device, dtype=dtype) * s_medium

    # Uniform-with-gap (exact, no retries): sample on [-1, v-d] ∪ [v+d, 1]
    L_left  = (v - d_eff + 1.0).clamp_min(0.0)     # [n, K-1]
    L_right = (1.0 - (v + d_eff)).clamp_min(0.0)   # [n, K-1]
    L_sum   = (L_left + L_right).clamp_min(1e-12)
    p_left  = L_left / L_sum

    R1 = torch.rand(n, K - 1, generator=rng, device=device)
    R2 = torch.rand(n, K - 1, generator=rng, device=device)
    choose_left = (R1 < p_left)

    u_left  = -1.0 + R2 * L_left
    u_right = (v + d_eff) + R2 * L_right
    u = torch.where(choose_left, u_left, u_right)  # [n, K-1]
    eps_uniform = (u - v)

    # Merge three branches
    eps = torch.where(is_small, eps_small, torch.where(is_medium, eps_medium, eps_uniform))
    v_cand = (v + eps).clamp(-0.999, 0.999)  # [n, K-1]

    # Avoid exact ties with the true value in normalized space (extremely rare but safe)
    if tau_norm > 0.0:
        near_true = (v_cand - v).abs() < tau_norm
        if near_true.any():
            sign = torch.sign(torch.rand(v_cand.shape, generator=rng, device=v_cand.device, dtype=v_cand.dtype) - 0.5)
            v_cand = torch.where(near_true, (v + sign * tau_norm).clamp(-0.999, 0.999), v_cand)

    # Assemble candidates with the true value at column 0 (normalized → raw)
    opts_norm = torch.cat([v_true.unsqueeze(1), v_cand], dim=1)     # [n, K]
    opts_raw  = _norm_to_raw(opts_norm).to(dtype=torch.float32)     # keep model-friendly dtype

    # Row-wise shuffle and gold index
    keys   = torch.rand(n, K, generator=rng, device=device)
    perm   = keys.argsort(dim=1)                 # [n, K], values are original column indices
    opts_raw = opts_raw.gather(1, perm)          # [n, K]
    # Cast bool -> long to support older CPU builds where argmax on bool is not implemented
    gold_idx = (perm == 0).to(torch.int64).argmax(dim=1)  # [n]

    return opts_raw, gold_idx.long()


def _sample_mcc_options(
    true_value_tensor: torch.Tensor,
    mcc_config: dict,
    rng: torch.Generator = None,
) -> tuple[torch.Tensor, int]:
    """
    Generate K candidates for MCC task (1 true + K-1 distractors).
    
    Args:
        true_value_tensor: scalar tensor (the true numerical value)
        mcc_config: dict with 'K', 'noise' sub-config
        rng: optional torch.Generator for reproducibility
    
    Returns:
        opts_raw: Tensor[K] - All K candidates including true value (shuffled)
        mcc_label: int - Position of true value in opts_raw (for InfoNCE loss)
    """
    device = true_value_tensor.device
    noise_cfg = mcc_config.get('noise', {})
    K = int(mcc_config.get('K', 5))
    scales = noise_cfg.get('gaussian_scales', [0.05, 0.20])
    mix_probs = torch.tensor(noise_cfg.get('mix_probs', [0.5, 0.35, 0.15]), device=device)
    min_dist = float(noise_cfg.get('large_min_norm_dist', 0.3))

    v_true = _raw_to_norm(true_value_tensor)
    candidates = [true_value_tensor.unsqueeze(0)]  # Fix: consistent shape [1]
    existing = candidates[0]  # Initialize dedup cache
    
    max_attempts = K * 10  # Scale with difficulty
    attempts = 0
    
    while len(candidates) < K and attempts < max_attempts:
        attempts += 1
        
        # Sample noise type  
        noise_type = torch.multinomial(mix_probs, 1, generator=rng).item()
        
        if noise_type == 0:        # small
            eps = torch.randn(1, device=device, generator=rng) * scales[0]
        elif noise_type == 1:      # medium
            eps = torch.randn(1, device=device, generator=rng) * scales[1]
        else:                      # large uniform - preserve distribution
            for retry in range(20):  # More retries for proper uniform sampling
                u = torch.rand(1, device=device, generator=rng) * 2.0 - 1.0
                if (u - v_true).abs().item() >= min_dist:
                    break
            else:
                # Fallback: use minimum distance (rare case)
                sign = torch.sign(torch.rand(1, device=device) - 0.5)
                u = v_true + sign * min_dist
                # Clamp to avoid boundary issues
                u = u.clamp(-0.99, 0.99)
            eps = u - v_true
            
        v_cand = (v_true + eps).clamp(-0.999, 0.999)
        x_cand = _norm_to_raw(v_cand).to(true_value_tensor.dtype)
        
        # Efficient deduplication with cache (Fix 1: explicit rtol=0.0)
        if not torch.any(torch.isclose(existing, x_cand, atol=1e-8, rtol=0.0)):
            candidates.append(x_cand)
            existing = torch.cat(candidates)  # Update cache only when adding
    
    if attempts == max_attempts:
        warnings.warn("MCC: reached max_attempts while sampling unique distractors")
    
    # Handle edge case: not enough unique candidates
    while len(candidates) < K:
        base = candidates[-1]
        perturbation = torch.randn_like(base) * 0.01
        new_candidate = base + perturbation
        
        # Check for duplicates before adding
        if not torch.any(torch.isclose(existing, new_candidate, atol=1e-8, rtol=0.0)):
            candidates.append(new_candidate)
            existing = torch.cat(candidates)
        # If duplicate, while loop continues with new random perturbation
    
    opts_raw = torch.cat(candidates[:K], dim=0)
    
    # Simple CPU shuffle with generator support
    perm = torch.randperm(K, generator=rng)
    opts_raw = opts_raw[perm]
    gold_idx = (perm == 0).nonzero(as_tuple=True)[0].item()

    return opts_raw.float(), gold_idx


@lru_cache(maxsize=1)
def _load_held_out_codes(path: str) -> set[str]:
    """Load held-out codes from YAML file with caching."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return set(cfg.get("held_out_codes", []))


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
    

def collate_exams(
    batch: List[Dict],
    code_embedder: CategoricalEmbedder,
    text_embedder: TextEmbedder,
    config: dict,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Decomposes nested test structures and prepares inputs for multi-modal learning
    with appropriate masking for different training objectives.
    
    Args:
        batch: List of dictionaries, each representing an exam
        code_embedder: Embedder for test codes and categorical values
        text_embedder: Embedder for text values and result text
        config: Training configuration dictionary containing masking probabilities and task settings
        device: Device for tensors
        
    Returns:
        Dictionary with model inputs and training labels:
        - exam_dates: Tensor[B] - Exam dates as days since epoch
        - exam_ages: Tensor[B] - Age bins for each exam (0=missing, 1=<20, 2=[20,30), ..., 8=>=80)
        - exam_genders: Tensor[B] - Gender IDs for each exam (0=missing, 1=M, 2=F)
        - segment_lengths: List[int] - Number of exams per person (sum=B)
        - B: int - Batch size
        - T_max: int - Maximum number of tests per exam in this batch
    """
    # Extract training configuration parameters
    training_config = config.get('training', {})
    
    # Existing task parameters
    p_cvr = training_config.get('p_cvr', 0.20)  # Cell Value Retrieval probability
    p_mcm = training_config.get('p_mcm', 0.15)  # Masked Category Modeling probability
    p_mlm = training_config.get('p_mlm', 0.15)  # Masked Language Modeling probability
    use_held_out_codes = training_config.get('use_held_out_codes', True)  # Whether to mask held-out codes
    held_out_codes_path = training_config.get('held_out_codes_path', "config/splitting/held_out_codes.yaml")  # Path to held-out codes config
    
    # MCC parameters - check if enabled first
    mcc_config = training_config.get('mcc', {})
    mcc_enabled = mcc_config.get('enabled', True)  # Default to enabled
    p_mcc = training_config.get('p_mcc', 0.20) if mcc_enabled else 0.0  # Skip masking if disabled
    
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

    # Load held-out codes if needed
    held_out_codes_set: set[str] = set()
    if use_held_out_codes:
        try:
            held_out_codes_set = _load_held_out_codes(held_out_codes_path)
            logger.debug(f"Loaded {len(held_out_codes_set)} held-out codes from {held_out_codes_path}")
        except FileNotFoundError:
            logger.warning(f"Held-out codes file not found at {held_out_codes_path}, proceeding without masking")
            use_held_out_codes = False
        except Exception as e:
            logger.warning(f"Error loading held-out codes: {e}, proceeding without masking")
            use_held_out_codes = False
    
    # Statistics for held-out code tracking
    held_out_stats = Counter()  # Count occurrences of each held-out code
    total_held_out_cells = 0
    
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
            if use_held_out_codes and code in held_out_codes_set:
                # Skip this test entirely - leave as padding
                held_out_stats[code] += 1
                total_held_out_cells += 1
                continue
                
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

    # Avoid logging mess in multiprocessing situation
    # # Log held-out code statistics for this batch
    # if use_held_out_codes and total_held_out_cells > 0:
    #     held_out_codes_list = list(held_out_stats.keys())
    #     logger.info(
    #         f"This batch masked out {total_held_out_cells} held-out cells "
    #         f"(codes: {held_out_codes_list[:5]}{'...' if len(held_out_codes_list) > 5 else ''})"
    #     )
    # elif use_held_out_codes:
    #     logger.debug("No held-out codes found in this batch")
    
    # Prepare categorical inputs (codes and categorical values)
    
    # Process codes
    code_tokens = flat_codes
    code_token_ids = code_embedder.map(code_tokens, device=device)
    code_token_ids = code_token_ids.reshape(B, T_max)
    
    # Process categorical values
    cat_value_tokens = flat_cat_values
    cat_value_ids = code_embedder.map(cat_value_tokens, device=device)
    cat_value_ids = cat_value_ids.reshape(B, T_max)
    
    # MCM preprocessing - mask some categorical values (vectorized)
    mask_token_id = code_embedder.get_token_id(MASK_CAT_TOKEN)
    
    # Create probability matrix for MCM
    mcm_probs = torch.full((B, T_max), p_mcm, device=device)
    mcm_probs = mcm_probs * mask_cat  # No need for .to(device) since mask_cat is already on device
    
    # Generate mask
    is_mcm = torch.bernoulli(mcm_probs).bool()
    
    # Create inputs and labels
    mcm_inputs = cat_value_ids.clone()
    mcm_labels = torch.full_like(cat_value_ids, -100)  # -100 is ignored in loss calculation
    
    # Apply masking
    mcm_labels[is_mcm] = cat_value_ids[is_mcm]
    mcm_inputs[is_mcm] = mask_token_id
    
    # Process numerical values
    num_values = torch.tensor(flat_num_values, dtype=torch.float, device=device).reshape(B, T_max)

    # MCC preprocessing for numerical values
    if mcc_enabled and mask_num.any():
        # Create probability matrix for MCC
        mcc_probs = torch.full((B, T_max), p_mcc, device=device)
        mcc_probs = mcc_probs * mask_num  # Only mask numerical positions
        
        # Generate mask
        mcc_mask = torch.bernoulli(mcc_probs).bool()
        
        # # Generate candidates for each masked cell
        # # OLD version
        # if mcc_mask.any():
        #     opts_raw_list = []
        #     mcc_labels_list = []
            
        #     # Get masked positions
        #     masked_positions = torch.nonzero(mcc_mask)  # [n_mcc, 2]
            
        #     for batch_idx, cell_idx in masked_positions:
        #         true_value = num_values[batch_idx, cell_idx].clone()
        #         opts_raw, mcc_label = _sample_mcc_options(true_value, mcc_config)
        #         opts_raw_list.append(opts_raw)
        #         mcc_labels_list.append(mcc_label)
            
        #     # Convert to tensors
        #     opts_raw = torch.stack(opts_raw_list)  # [n_mcc, K]
        #     mcc_labels = torch.tensor(mcc_labels_list, dtype=torch.long, device=device)  # [n_mcc]
            
        #     # Apply masking to num_values (use special value)
        #     num_values[mcc_mask] = -999.0

        # Generate candidates for each masked cell (vectorized)
        if mcc_mask.any():
            # Flatten true values in the masked positions
            true_values_flat = num_values[mcc_mask]  # [n_mcc]
        
            # Optional: reproducible CPU RNG (use training seed if provided)
            gen_cpu = None
            base_seed = int(config.get('seed', 0))
            if base_seed != 0:
                gen_cpu = torch.Generator(device='cpu').manual_seed(base_seed)
        
            # Vectorized candidate generation (no Python loops)
            opts_raw, mcc_labels = _sample_mcc_options_batched(true_values_flat, mcc_config, gen_cpu)
        
            # Apply masking to num_values (use special sentinel value)
            num_values[mcc_mask] = -999.0
        else:
            # No cells selected
            K = mcc_config.get('K', 5)
            opts_raw = torch.empty(0, K, dtype=torch.float32, device=device)
            mcc_labels = torch.empty(0, dtype=torch.long, device=device)
    else:
        # MCC disabled or no numerical cells
        K = mcc_config.get('K', 5)
        mcc_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        opts_raw = torch.empty(0, K, dtype=torch.float32, device=device)
        mcc_labels = torch.empty(0, dtype=torch.long, device=device)
    
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
    
    # CVR preprocessing - select text cells for value retrieval
    # Create probability matrix for CVR
    cvr_probs = torch.full((B, T_max), p_cvr, device=device)
    cvr_probs = cvr_probs * mask_text
    
    # Generate mask
    is_cvr = torch.bernoulli(cvr_probs).bool()
    
    # Create position mask
    cvr_mask = torch.full((B, T_max), -100, dtype=torch.long, device=device)
    cvr_mask[mask_text] = 0  # Not selected by default
    cvr_mask[is_cvr] = 1     # Selected for CVR
    
    # Extract original content and apply masking
    cvr_true_ids = []
    cvr_true_attention_masks = []
    cvr_labels = []
    
    # Extract content before masking (use text_locations for mapping)
    cvr_mask_in_text_seqs = torch.zeros(len(text_locations), dtype=torch.bool, device=device)
    for i, (batch_idx, cell_idx) in enumerate(text_locations):
        if is_cvr[batch_idx, cell_idx]:
            cvr_mask_in_text_seqs[i] = True
            # Clone before saving to prevent corruption from later masking
            cvr_true_ids.append(text_token_ids[i].clone())
            cvr_true_attention_masks.append(text_attention_mask[i].clone())
    
    # Apply masking to selected cells (only valid tokens)
    for i in torch.nonzero(cvr_mask_in_text_seqs).squeeze(-1):
        valid_len = int(text_attention_mask[i].sum())
        text_token_ids[i, :valid_len].fill_(text_embedder.tokenizer.mask_token_id)
    
    # Permute candidates and create labels
    n_cvr = len(cvr_true_ids)
    if n_cvr > 0:
        # Random permutation
        perm_indices = torch.randperm(n_cvr).tolist()
        
        # Shuffle candidates
        cvr_true_ids = [cvr_true_ids[i] for i in perm_indices]
        cvr_true_attention_masks = [cvr_true_attention_masks[i] for i in perm_indices]
        
        # Create corresponding labels (inverse permutation for InfoNCE)
        cvr_labels_list = [0] * n_cvr
        for new_pos, old_pos in enumerate(perm_indices):
            cvr_labels_list[old_pos] = new_pos
        
        # Convert to tensor
        cvr_labels = torch.tensor(cvr_labels_list, dtype=torch.long, device=device)
    else:
        # Handle zero-sample case
        cvr_labels = torch.empty(0, dtype=torch.long, device=device)

    # Convert CVR lists to tensors
    if n_cvr > 0:
        cvr_true_ids = torch.stack(cvr_true_ids)  # [n_cvr, L_cvr]
        cvr_true_attention_masks = torch.stack(cvr_true_attention_masks)  # [n_cvr, L_cvr]
    else:
        cvr_true_ids = torch.empty(0, 0, dtype=torch.long, device=device)
        cvr_true_attention_masks = torch.empty(0, 0, device=device)
    
    # Process result texts with MLM - BRANCHING POINT
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
            # EXISTING PATH: Tokenize raw text, but keep missing samples empty
            pad_token_id = text_embedder.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Text tokenizer must define a pad_token_id for batching.")

            token_lists: List[List[int]] = [[] for _ in range(B)]
            attention_lists: List[List[int]] = [[] for _ in range(B)]
            nonempty_indices = []
            nonempty_texts = []

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
    
    # MLM preprocessing for result texts
    mlm_labels = input_ids.clone().to(dtype=torch.long)
    
    # Only consider tokens that are:
    # 1. Not special tokens
    # 2. Not padding
    # 3. Covered by the attention mask
    probability_matrix = torch.full(input_ids.shape, p_mlm, device=device)

    # Generate special tokens mask with proper handling for batch
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, seq in enumerate(input_ids.tolist()):
        # Get special token mask for this sequence
        seq_mask = text_embedder.tokenizer.get_special_tokens_mask(
            seq, already_has_special_tokens=True
        )
        # Convert to tensor and store
        special_tokens_mask[i, :len(seq_mask)] = torch.tensor(
            seq_mask, dtype=torch.bool, device=device
        )[:special_tokens_mask.size(1)]    

    # Apply masks
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
    
    # Select tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    mlm_labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # Apply masking strategy: 80% MASK, 10% random, 10% unchanged
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = text_embedder.tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(text_embedder.tokenizer), input_ids.shape, dtype=torch.int32, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
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
    
    # Add held-out statistics to metadata for tracking
    if use_held_out_codes:
        metadata["held_out_cells_count"] = total_held_out_cells
        metadata["held_out_codes_in_batch"] = list(held_out_stats.keys())
    
    # Assemble all outputs
    outputs = {
        # Categorical inputs and labels
        "code_ids": code_token_ids,
        "cat_value_ids": cat_value_ids,
        "mcm_inputs": mcm_inputs,
        "mcm_labels": mcm_labels,
        
        # Numerical inputs
        "num_values": num_values,

        # MCC data
        "mcc_mask": mcc_mask,
        "opts_raw": opts_raw,
        "mcc_labels": mcc_labels,
        
        # Text test value inputs (optimized)
        "text_token_ids": text_token_ids,
        "text_attention_mask": text_attention_mask,
        "text_locations": text_locations,
        
        # Result text with MLM
        "result_input_ids": input_ids,
        "result_attention_mask": attention_mask,
        "result_mlm_labels": mlm_labels,
        
        # Type information
        "type_ids": type_ids,
        
        # Masks for each modality
        **modality_masks,
        
        # CVR data
        "cvr_mask": cvr_mask,
        "cvr_labels": cvr_labels,
        "cvr_true_ids": cvr_true_ids,
        "cvr_true_attention_masks": cvr_true_attention_masks,
        
        # Metadata
        **metadata
    }
    
    return outputs

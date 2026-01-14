# src/downstream/phenotyping/incident_anchored/collate_fn.py

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.embedders import CategoricalEmbedder, TextEmbedder


def _pad_and_stack_tokens(
    token_lists: List[List[int]],
    attention_lists: List[List[int]],
    pad_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(map(len, token_lists)) if token_lists else 0
    padded_tok = [seq + [pad_id] * (max_len - len(seq)) for seq in token_lists]
    padded_att = [seq + [0] * (max_len - len(seq)) for seq in attention_lists]
    input_ids = torch.as_tensor(padded_tok, dtype=torch.int32, device=device)
    attn_mask = torch.as_tensor(padded_att, device=device)
    return input_ids, attn_mask


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return 0
        return int(value)
    return 0


def _stack_manifest_meta(
    meta_list: List[Dict[str, Any]],
    tensor_keys: set[str],
    device: str,
) -> Dict[str, Any]:
    keys: List[str] = []
    for meta in meta_list:
        for key in meta.keys():
            if key not in keys:
                keys.append(key)

    stacked: Dict[str, Any] = {}
    for key in keys:
        values = [meta.get(key) for meta in meta_list]
        if key in tensor_keys:
            stacked[key] = torch.tensor([_coerce_int(v) for v in values], dtype=torch.long, device=device)
        else:
            stacked[key] = values
    return stacked


def incident_anchored_collate_fn(
    batch: List[Dict[str, Any]],
    code_embedder: CategoricalEmbedder,
    text_embedder: TextEmbedder,
    config: Dict[str, Any],
    device: str = "cpu",
    manifest_meta_key: str = "manifest_meta",
    manifest_meta_tensor_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Collate function for incident-anchored representation extraction.

    Uses the core SSL input preparation but disables all SSL masking tasks.
    """
    if not batch:
        return {}

    has_result_data = False
    has_pretokenized_result = False
    sample = batch[0]
    if "result_input_ids" in sample and "result_attention_mask" in sample:
        has_result_data = True
        has_pretokenized_result = True
    elif "result_text" in sample:
        has_result_data = True
        has_pretokenized_result = False

    B = len(batch)
    T_max = max(len(sample["tests"]) for sample in batch)

    PAD_CODE = "<PAD>"
    PAD_CAT = "<PAD>"
    UNK_CODE = "<UNK>"
    UNK_CAT = "<UNK>"
    MASK_CAT_TOKEN = "<MASK>"

    gender_to_id = {"M": 1, "F": 2}

    flat_codes = [PAD_CODE] * (B * T_max)
    flat_cat_values = [PAD_CAT] * (B * T_max)
    flat_num_values = [0.0] * (B * T_max)
    flat_type_ids = [0] * (B * T_max)

    nonempty_text_values: List[str] = []
    text_indices: List[Tuple[int, int]] = []

    mask_code = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_num = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_cat = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    mask_text = torch.zeros(B, T_max, dtype=torch.bool, device=device)

    type_to_id = {"PQ": 1, "CD": 2, "CO": 2, "ST": 3}

    exam_dates: List[int] = []
    exam_ages: List[int] = []
    exam_genders: List[int] = []

    if has_result_data:
        if has_pretokenized_result:
            result_input_ids: List[List[int]] = []
            result_attention_masks: List[List[int]] = []
        else:
            result_texts = [""] * B

    current_person = None
    segment_lengths: List[int] = []
    segment_count = 0

    def age_to_bin_id(age: int) -> int:
        if age < 20:
            return 1
        if age < 30:
            return 2
        if age < 40:
            return 3
        if age < 50:
            return 4
        if age < 60:
            return 5
        if age < 70:
            return 6
        if age < 80:
            return 7
        return 8

    for i, sample in enumerate(batch):
        person_id = sample["person_id"]
        if current_person != person_id:
            if segment_count > 0:
                segment_lengths.append(segment_count)
            current_person = person_id
            segment_count = 1
        else:
            segment_count += 1

        exam_date = sample["ExamDate"]
        if isinstance(exam_date, date):
            epoch = date(1970, 1, 1)
            days_since_epoch = (exam_date - epoch).days
            exam_dates.append(days_since_epoch)
            exam_year = exam_date.year
        else:
            exam_dates.append(0)
            exam_year = 0

        birth_year = sample.get("birth_year", 0)
        if birth_year > 0 and exam_year > birth_year:
            exam_age = exam_year - birth_year
            age_bin_id = age_to_bin_id(exam_age)
        else:
            age_bin_id = 0
        exam_ages.append(age_bin_id)

        gender = sample.get("gender", "")
        exam_genders.append(gender_to_id.get(gender, 0))

        if has_result_data:
            if has_pretokenized_result:
                result_input_ids.append(sample["result_input_ids"])
                result_attention_masks.append(sample["result_attention_mask"])
            else:
                result_texts[i] = sample["result_text"]

        tests = sample["tests"]
        for j, test in enumerate(tests):
            if j >= T_max:
                break
            code = test.get("code", "")
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

    code_token_ids = code_embedder.map(flat_codes, device=device).reshape(B, T_max)
    cat_value_ids = code_embedder.map(flat_cat_values, device=device).reshape(B, T_max)
    num_values = torch.tensor(flat_num_values, dtype=torch.float, device=device).reshape(B, T_max)

    if nonempty_text_values:
        text_value_encodings = text_embedder.tokenize(nonempty_text_values)
        text_token_ids = text_value_encodings["input_ids"].to(device)
        text_attention_mask = text_value_encodings["attention_mask"].to(device)
        text_locations = torch.tensor(text_indices, dtype=torch.long, device=device)
    else:
        text_token_ids = torch.empty(0, 0, dtype=torch.long, device=device)
        text_attention_mask = torch.empty(0, 0, dtype=torch.long, device=device)
        text_locations = torch.empty(0, 2, dtype=torch.long, device=device)

    if has_result_data:
        if has_pretokenized_result:
            input_ids, attention_mask = _pad_and_stack_tokens(
                result_input_ids,
                result_attention_masks,
                text_embedder.tokenizer.pad_token_id,
                device,
            )
        else:
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
                device,
            )
    else:
        input_ids = torch.empty(B, 0, dtype=torch.long, device=device)
        attention_mask = torch.empty(B, 0, dtype=torch.long, device=device)

    type_ids = torch.tensor(flat_type_ids, dtype=torch.long, device=device).reshape(B, T_max)
    mask_type = type_ids != 0

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

    mcm_inputs = cat_value_ids.clone()
    mcm_labels = torch.full_like(cat_value_ids, -100)
    cvr_mask = torch.full_like(cat_value_ids, -100)
    cvr_labels = torch.empty(0, dtype=torch.long, device=device)
    cvr_true_ids = torch.empty(0, 0, dtype=torch.long, device=device)
    cvr_true_attention_masks = torch.empty(0, 0, dtype=torch.bool, device=device)
    mcc_mask = torch.zeros_like(mask_num, dtype=torch.bool)
    opts_raw = torch.empty(0, 0, dtype=torch.float32, device=device)
    mcc_labels = torch.empty(0, dtype=torch.long, device=device)
    result_mlm_labels = torch.full_like(input_ids, -100)

    manifest_meta_list = [sample.get(manifest_meta_key, {}) for sample in batch]
    default_tensor_keys = {
        "t_rel",
        "is_index",
        "age_at_index",
        "pre_total_obs",
        "pre_in_5y_obs",
        "min_pre_in_W",
        "n_candidates_in_stratum",
        "K_washout",
        "W_years",
    }
    tensor_keys = set(manifest_meta_tensor_keys or default_tensor_keys)
    manifest_meta = _stack_manifest_meta(manifest_meta_list, tensor_keys, device)

    return {
        "code_ids": code_token_ids,
        "cat_value_ids": cat_value_ids,
        "num_values": num_values,
        "text_token_ids": text_token_ids,
        "text_attention_mask": text_attention_mask,
        "text_locations": text_locations,
        "result_input_ids": input_ids,
        "result_attention_mask": attention_mask,
        "type_ids": type_ids,
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
        **modality_masks,
        **metadata,
        "manifest_meta": manifest_meta,
    }

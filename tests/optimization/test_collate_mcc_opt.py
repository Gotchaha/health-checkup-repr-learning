# tests/optimization/test_collate_mcc_opt.py
# -*- coding: utf-8 -*-
"""
Tests focused only on MCC sampling:
- Old per-cell sampler: _sample_mcc_options
- New batched sampler: _sample_mcc_options_batched

We assert API compatibility (shapes/dtypes) and invariants:
1) Exactly one true candidate per row; label points to it.
2) Uniform-with-gap distance constraint holds in normalized domain.
3) Determinism (for the new batched sampler with a fixed CPU seed).
4) Empty input behavior.

Note: We do NOT require value-wise equality between old and new samplers.
"""

import torch

from src.models.collate_fn import (
    _sample_mcc_options,               # old per-cell sampler
    _sample_mcc_options_batched,       # new batched sampler
    _raw_to_norm,
)


def _near_true_mask(norm_opts: torch.Tensor, norm_true: torch.Tensor, tol: float = 1e-6):
    """Boolean [n, K] mask marking near-true positions per row in normalized domain."""
    return (norm_opts - norm_true).abs() < tol


def _run_old_per_cell_loop(true_vals: torch.Tensor, mcc_config: dict, gen: torch.Generator | None):
    """Simulate a 'batched' call using the old per-cell function."""
    opts_list, lab_list = [], []
    for tv in true_vals:
        # tv is a 0-dim tensor; pass generator to keep determinism within this loop
        opts_raw, lab = _sample_mcc_options(tv, mcc_config, rng=gen)
        opts_list.append(opts_raw)
        lab_list.append(lab)
    opts_old = torch.stack(opts_list) if opts_list else torch.empty(0, mcc_config.get("K", 5))
    labels_old = torch.tensor(lab_list, dtype=torch.long, device=true_vals.device) if lab_list else torch.empty(0, dtype=torch.long)
    return opts_old, labels_old


def test_api_compat_shapes_and_dtypes_old_vs_new():
    """Compare formats (shape/dtype) between old per-cell loop and new batched sampler."""
    true_vals = torch.tensor([-1.5, -0.2, 0.0, 0.3, 0.7, 1.2], dtype=torch.float32)
    mcc_config = {
        "K": 5,
        "dedup_tau_norm": 1e-6,
        "noise": {
            "gaussian_scales": [0.05, 0.20],
            "mix_probs": [0.4, 0.4, 0.2],
            "large_min_norm_dist": 0.3,
        },
    }

    # Separate generators (no need to align RNG streams between different algorithms)
    gen_old = torch.Generator(device="cpu").manual_seed(123)
    gen_new = torch.Generator(device="cpu").manual_seed(123)

    opts_old, lab_old = _run_old_per_cell_loop(true_vals, mcc_config, gen_old)
    opts_new, lab_new = _sample_mcc_options_batched(true_vals, mcc_config, gen_new)

    n, K = true_vals.numel(), mcc_config["K"]

    # Shapes
    assert opts_old.shape == (n, K)
    assert opts_new.shape == (n, K)
    assert lab_old.shape == (n,)
    assert lab_new.shape == (n,)

    # Dtypes (downstream expects float32 for opts, long for labels)
    assert opts_old.dtype == torch.float32
    assert opts_new.dtype == torch.float32
    assert lab_old.dtype == torch.long
    assert lab_new.dtype == torch.long


def test_label_semantics_old_and_new():
    """For both samplers: exactly one near-true per row; label points to it."""
    true_vals = torch.tensor([-0.9, -0.25, 0.0, 0.33, 0.55], dtype=torch.float32)
    mcc_config = {
        "K": 6,
        "dedup_tau_norm": 1e-6,
        "noise": {
            "gaussian_scales": [0.05, 0.20],
            "mix_probs": [0.3, 0.4, 0.3],
            "large_min_norm_dist": 0.3,
        },
    }
    gen_old = torch.Generator(device="cpu").manual_seed(7)
    gen_new = torch.Generator(device="cpu").manual_seed(7)

    opts_old, lab_old = _run_old_per_cell_loop(true_vals, mcc_config, gen_old)
    opts_new, lab_new = _sample_mcc_options_batched(true_vals, mcc_config, gen_new)

    norm_true = _raw_to_norm(true_vals).unsqueeze(1)  # [n,1]
    norm_old = _raw_to_norm(opts_old)                 # [n,K]
    norm_new = _raw_to_norm(opts_new)                 # [n,K]

    near_old = _near_true_mask(norm_old, norm_true)
    near_new = _near_true_mask(norm_new, norm_true)

    # Exactly one true per row
    assert torch.all(near_old.sum(dim=1) == 1)
    assert torch.all(near_new.sum(dim=1) == 1)

    # Labels point to the true match
    rows_old = torch.arange(lab_old.numel(), dtype=torch.long)
    rows_new = torch.arange(lab_new.numel(), dtype=torch.long)
    assert torch.all(near_old[rows_old, lab_old])
    assert torch.all(near_new[rows_new, lab_new])


def test_uniform_with_gap_distance_constraint_both():
    """Force only uniform-with-gap and check min normalized distance ≥ d for both samplers."""
    true_vals = torch.tensor([-0.8, -0.3, 0.0, 0.25, 0.6], dtype=torch.float32)
    d = 0.35
    K = 7
    mcc_config = {
        "K": K,
        "dedup_tau_norm": 1e-6,
        "noise": {
            "gaussian_scales": [0.05, 0.20],
            "mix_probs": [0.0, 0.0, 1.0],  # only uniform-with-gap
            "large_min_norm_dist": d,
        },
    }
    gen_old = torch.Generator(device="cpu").manual_seed(2025)
    gen_new = torch.Generator(device="cpu").manual_seed(2025)

    opts_old, _ = _run_old_per_cell_loop(true_vals, mcc_config, gen_old)
    opts_new, _ = _sample_mcc_options_batched(true_vals, mcc_config, gen_new)

    for opts in (opts_old, opts_new):
        norm_true = _raw_to_norm(true_vals).unsqueeze(1)  # [n,1]
        norm_opts = _raw_to_norm(opts)                    # [n,K]
        near = (norm_opts - norm_true).abs() < 1e-6
        # distractors = all non-true columns per row
        distractors = norm_opts[~near].view(true_vals.numel(), K - 1)
        min_dist_obs = (distractors - norm_true.repeat(1, K - 1)).abs().min(dim=1).values
        # Allow tiny numeric slack
        assert torch.all(min_dist_obs >= d - 1e-6)


def test_batched_determinism_with_fixed_seed():
    """New batched sampler should be deterministic with a fixed CPU seed."""
    true_vals = torch.linspace(-1.0, 1.0, steps=33, dtype=torch.float32)
    mcc_config = {
        "K": 6,
        "dedup_tau_norm": 1e-6,
        "noise": {
            "gaussian_scales": [0.05, 0.20],
            "mix_probs": [0.3, 0.3, 0.4],
            "large_min_norm_dist": 0.3,
        },
    }
    gen1 = torch.Generator(device="cpu").manual_seed(777)
    gen2 = torch.Generator(device="cpu").manual_seed(777)
    gen3 = torch.Generator(device="cpu").manual_seed(778)

    opts1, lab1 = _sample_mcc_options_batched(true_vals, mcc_config, gen1)
    opts2, lab2 = _sample_mcc_options_batched(true_vals, mcc_config, gen2)
    opts3, lab3 = _sample_mcc_options_batched(true_vals, mcc_config, gen3)

    assert torch.equal(opts1, opts2)
    assert torch.equal(lab1, lab2)
    # Loose inequality to avoid pathological coincidences
    assert not torch.allclose(opts1, opts3)
    assert not torch.equal(lab1, lab3)


def test_empty_input_returns_empty():
    """n_mcc=0 returns empty tensors with correct dtypes and shapes."""
    true_vals = torch.empty(0, dtype=torch.float32)
    mcc_config = {"K": 5, "noise": {"gaussian_scales": [0.05, 0.20], "mix_probs": [0.3, 0.3, 0.4]}}
    opts_old, labels_old = _run_old_per_cell_loop(true_vals, mcc_config, gen=None)
    opts_new, labels_new = _sample_mcc_options_batched(true_vals, mcc_config, rng=None)
    assert opts_old.shape == (0, 5) and labels_old.shape == (0,)
    assert opts_new.shape == (0, 5) and labels_new.shape == (0,)
    assert opts_old.dtype == torch.float32 and labels_old.dtype == torch.long
    assert opts_new.dtype == torch.float32 and labels_new.dtype == torch.long

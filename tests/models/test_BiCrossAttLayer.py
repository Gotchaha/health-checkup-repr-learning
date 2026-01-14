# tests/models/test_BiCrossAttLayer.py

import math
import torch
import torch.nn as nn
from src.models.transformers import BiCrossAttLayer


def test_bicrossatt_basic():
    """Basic functionality and shape test."""
    # Realistic dimensions from our pipeline
    B, D = 4, 768
    L_tab = 20  # 4 + 2*8 (example with T_max=8)
    L_text = 32  # typical result text length
    
    # Create module
    layer = BiCrossAttLayer(d_model=D, n_heads=8, asymmetric_fusion=False)
    
    # Synthetic inputs
    tab_tokens = torch.randn(B, L_tab, D)
    text_tokens = torch.randn(B, L_text, D)
    tab_mask = torch.ones(B, L_tab, dtype=torch.bool)
    text_mask = torch.ones(B, L_text, dtype=torch.bool)
    
    # Forward pass
    tab_out, text_out = layer(tab_tokens, text_tokens, tab_mask, text_mask)
    
    # Shape verification
    assert tab_out.shape == (B, L_tab, D)
    assert text_out.shape == (B, L_text, D)
    print("✓ Shape test passed")


def test_gradient_flow():
    """Test that gradients flow through fusion parameters."""
    layer = BiCrossAttLayer(d_model=256, learnable_fusion=True, asymmetric_fusion=True)
    
    # Small synthetic data
    tab_tokens = torch.randn(2, 10, 256, requires_grad=True)
    text_tokens = torch.randn(2, 15, 256, requires_grad=True)
    
    # Forward + backward
    tab_out, text_out = layer(tab_tokens, text_tokens)
    loss = (tab_out.sum() + text_out.sum())
    loss.backward()
    
    # Check fusion gate parameters have gradients
    tab_gate_params = list(layer._tab_gate.parameters())
    text_gate_params = list(layer._text_gate.parameters())
    assert any(p.grad is not None for p in tab_gate_params)
    assert any(p.grad is not None for p in text_gate_params)
    print("✓ Gradient flow test passed")


def test_fusion_modes():
    """Test different fusion configurations."""
    configs = [
        {"learnable_fusion": True, "asymmetric_fusion": False},
        {"learnable_fusion": True, "asymmetric_fusion": True},
        {"learnable_fusion": False, "asymmetric_fusion": False},
    ]
    
    for config in configs:
        layer = BiCrossAttLayer(d_model=128, **config)
        tab_tokens = torch.randn(2, 8, 128)
        text_tokens = torch.randn(2, 12, 128)
        
        # Should not error
        tab_out, text_out = layer(tab_tokens, text_tokens)
        assert tab_out.shape == (2, 8, 128)
        assert text_out.shape == (2, 12, 128)
    
    print("✓ Fusion modes test passed")


def test_missing_text_forces_self_weight():
    """When text is missing, cross weights should drop to zero."""
    B, L_tab, L_text, D = 2, 6, 4, 128
    layer = BiCrossAttLayer(d_model=D, learnable_fusion=True, asymmetric_fusion=False)

    tab_tokens = torch.randn(B, L_tab, D)
    text_tokens = torch.zeros(B, L_text, D)
    tab_mask = torch.ones(B, L_tab, dtype=torch.bool)
    text_mask = torch.zeros(B, L_text, dtype=torch.bool)  # missing text

    layer(tab_tokens, text_tokens, tab_mask, text_mask)

    stats = layer._last_fusion_stats
    assert stats is not None
    assert math.isclose(stats["tab_cross_mean"], 0.0, abs_tol=1e-6)


if __name__ == "__main__":
    test_bicrossatt_basic()
    test_gradient_flow()
    test_fusion_modes()
    test_missing_text_forces_self_weight()
    print("All tests passed!")

# tests/models/test_fusion.py

import torch

from src.models.fusion import TextCompressor, ImportanceWeightedConcat


def test_text_compressor_presence_mask():
    torch.manual_seed(0)
    compressor = TextCompressor(d_model=16, n_out=4, n_heads=2, dropout=0.0)
    compressor.eval()

    text_tokens = torch.randn(2, 3, 16)
    text_mask = torch.tensor(
        [[1, 1, 1],
         [0, 0, 0]],
        dtype=torch.bool
    )

    comp, comp_mask = compressor(text_tokens, text_mask)

    assert comp_mask[0].all()
    assert not comp_mask[1].any()

    expected_null = compressor.null_comp.to(comp.dtype).expand(1, -1, -1)
    assert torch.allclose(comp[1:2], expected_null, atol=1e-6)


def test_text_compressor_empty_sequence():
    compressor = TextCompressor(d_model=8, n_out=2, n_heads=2, dropout=0.0)
    compressor.eval()

    text_tokens = torch.empty(1, 0, 8)
    text_mask = torch.empty(1, 0, dtype=torch.bool)

    comp, comp_mask = compressor(text_tokens, text_mask)

    assert comp.shape == (1, 2, 8)
    assert not comp_mask.any()

    expected_null = compressor.null_comp.to(comp.dtype).expand(1, -1, -1)
    assert torch.allclose(comp, expected_null, atol=1e-6)


def test_importance_weighted_concat_presence_gating():
    torch.manual_seed(1)
    concat = ImportanceWeightedConcat(d_model=8, use_layernorm=True, use_segment_emb=True)
    concat.eval()

    tab_tokens = torch.randn(2, 3, 8)
    text_tokens = torch.randn(2, 2, 8)
    tab_mask = torch.ones(2, 3, dtype=torch.bool)
    text_mask = torch.tensor(
        [[1, 0],
         [0, 0]],
        dtype=torch.bool
    )

    fused_tokens, fused_mask = concat(tab_tokens, text_tokens, tab_mask, text_mask)

    L_tab = tab_tokens.size(1)

    # Sample with text → verify masked position is zeroed
    assert fused_mask[0, L_tab]
    assert not fused_mask[0, L_tab + 1]
    assert torch.allclose(fused_tokens[0, L_tab + 1], torch.zeros_like(fused_tokens[0, L_tab + 1]))

    # Sample without text → entire text segment should be invalid and zero
    assert not fused_mask[1, L_tab:].any()
    assert torch.allclose(fused_tokens[1, L_tab:], torch.zeros_like(fused_tokens[1, L_tab:]))


if __name__ == "__main__":
    test_text_compressor_presence_mask()
    test_text_compressor_empty_sequence()
    test_importance_weighted_concat_presence_gating()
    print("All fusion tests passed!")

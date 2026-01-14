# src/models/fusion.py

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, Tuple


class TextCompressor(nn.Module):
    """
    Text compression module using cross-attention with learnable query tokens.
    
    Compresses variable-length text sequences to a fixed number of output tokens
    using cross-attention mechanism. Includes residual connection with mean-pooled
    input for preserving global information. Returns both compressed tokens and
    a presence-aware attention mask (only samples with valid text are marked True).
    """
    
    def __init__(
        self,
        d_model: int,
        n_out: int = 50,
        n_heads: int = 8,
        dropout: float = 0.1,
        residual_init: float = -2.2
    ):
        """
        Initialize TextCompressor.
        
        Args:
            d_model: Model dimension
            n_out: Number of output compressed tokens
            n_heads: Number of attention heads
            dropout: Dropout probability
            residual_init: Initial logit value for residual weight (sigmoid(residual_init) ≈ residual strength)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_out = n_out
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Learnable compression tokens (queries for cross-attention)
        self.compress_tokens = nn.Parameter(torch.randn(1, n_out, d_model))
        
        # Cross-attention module
        self.compress_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Post-attention layer norm
        self.post_ln = nn.LayerNorm(d_model)
        
        # Learnable residual weight in (0,1)
        self.residual_logit = nn.Parameter(torch.full((1,), residual_init))
        
        # Initialize compression tokens
        nn.init.xavier_uniform_(self.compress_tokens)

        # Learnable null representation for missing text
        self.null_comp = nn.Parameter(torch.zeros(1, n_out, d_model))
    
    def forward(
        self, 
        text_tokens: torch.Tensor, 
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress text tokens to fixed-length representation.
        
        Args:
            text_tokens: Input text tokens [B, L_text, D]
            text_mask: Attention mask [B, L_text] where True=valid, False=padding
            
        Returns:
            Tuple of:
            - comp: Compressed tokens [B, n_out, D]
            - comp_mask: Attention mask for compressed tokens [B, n_out] (all ones)
        """
        B, L, D = text_tokens.shape
        device = text_tokens.device

        if text_mask is not None:
            mask = text_mask.bool()
        else:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        presence = mask.any(dim=1) if L > 0 else torch.zeros(B, dtype=torch.bool, device=device)

        # Initialize outputs with null representation; mask reflects presence
        comp = self.null_comp.expand(B, -1, -1).to(text_tokens.dtype).clone()
        comp_mask = presence.unsqueeze(1).expand(B, self.n_out)

        # If no tokens at all (L == 0) or no present samples, return early
        if L == 0 or not presence.any():
            return comp, comp_mask
        
        # Expand compression tokens for batch
        pool = self.compress_tokens.expand(B, -1, -1)  # [B, n_out, D]
        
        # Convert mask to PyTorch multihead attention format (True = ignore)
        key_pad_mask = ~mask  # [B, L_text]
        
        # Run attention only for present samples to avoid NaNs from all-masked K/V
        present_idx = presence.nonzero(as_tuple=True)[0]
        pool_sub = pool.index_select(0, present_idx)                 # [B_present, n_out, D]
        toks_sub = text_tokens.index_select(0, present_idx)          # [B_present, L, D]
        mask_sub = mask.index_select(0, present_idx)                 # [B_present, L]
        key_pad_sub = ~mask_sub

        comp_sub, _ = self.compress_attn(
            pool_sub,
            toks_sub,
            toks_sub,
            key_padding_mask=key_pad_sub,
            need_weights=False,
        )
        comp_sub = self.post_ln(comp_sub)

        # Masked mean pooling and residual for present samples
        denom_sub = mask_sub.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        mean_sub = (toks_sub * mask_sub.unsqueeze(-1).float()).sum(dim=1, keepdim=True) / denom_sub
        alpha = torch.sigmoid(self.residual_logit)
        comp_sub = comp_sub + alpha * mean_sub

        # Write back only to present rows; missing rows remain as null_comp
        comp.index_copy_(0, present_idx, comp_sub)

        return comp, comp_mask
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'n_out': self.n_out,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'residual_init': float(self.residual_logit.item())
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"TextCompressor(d_model={self.d_model}, n_out={self.n_out}, "
                f"n_heads={self.n_heads}, dropout={self.dropout})")


class ImportanceWeightedConcat(nn.Module):
    """
    Importance-weighted concatenation for multi-modal fusion.
    
    Applies learnable scaling weights to different modalities before concatenation
    to account for differences in information density. Includes optional segment
    embeddings to distinguish modalities and layer normalization for stability.
    Handles concatenation of both tokens and their attention masks.
    """
    
    def __init__(
        self,
        d_model: int,
        initial_tab_weight: float = 2.0,
        initial_text_weight: float = 1.0,
        use_layernorm: bool = True,
        use_segment_emb: bool = True
    ):
        """
        Initialize ImportanceWeightedConcat.
        
        Args:
            d_model: Model dimension
            initial_tab_weight: Initial scaling weight for tabular tokens
            initial_text_weight: Initial scaling weight for text tokens
            use_layernorm: Whether to apply layer normalization after concatenation
            use_segment_emb: Whether to add segment embeddings to distinguish modalities
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_layernorm = use_layernorm
        self.use_segment_emb = use_segment_emb
        
        # Learnable importance weights
        # Using (1 + ReLU(α)) ensures positive scaling and allows adaptation
        self.alpha_tab = nn.Parameter(torch.tensor(initial_tab_weight))
        self.alpha_text = nn.Parameter(torch.tensor(initial_text_weight))
        
        # Optional segment embeddings for modality distinction
        if use_segment_emb:
            self.seg_emb = nn.Embedding(2, d_model)  # 0=tabular, 1=text
            nn.init.normal_(self.seg_emb.weight, std=0.02)  # BERT-style initialization
        
        # Optional post-concatenation layer normalization
        if use_layernorm:
            self.post_ln = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        tab_tokens: torch.Tensor, 
        text_tokens: torch.Tensor,
        tab_mask: torch.Tensor,
        text_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply importance weighting, concatenate tokens and masks, add segment embeddings.
        
        Args:
            tab_tokens: Tabular tokens [B, L_tab, D]
            text_tokens: Text tokens [B, L_text, D]
            tab_mask: Attention mask for tabular tokens [B, L_tab]
            text_mask: Attention mask for text tokens [B, L_text]
            
        Returns:
            Tuple of:
            - fused_tokens: Concatenated tokens [B, L_tab + L_text, D]
            - fused_mask: Concatenated attention mask [B, L_tab + L_text]
        """
        B = tab_tokens.size(0)
        L_tab = tab_tokens.size(1)
        L_text = text_tokens.size(1)
        
        tab_weight = 1 + torch.relu(self.alpha_tab)
        text_weight = 1 + torch.relu(self.alpha_text)

        tab_scaled = tab_tokens * tab_weight
        text_presence = text_mask.any(dim=1, keepdim=True).float() if L_text > 0 else torch.zeros(B, 1, device=tab_tokens.device)
        if L_text == 0:
            text_scaled = text_tokens * text_weight
        else:
            text_scaled = text_tokens * text_weight * text_presence.unsqueeze(-1)
        
        # Concatenate along sequence dimension
        fused_tokens = torch.cat([tab_scaled, text_scaled], dim=1)  # [B, L_tab + L_text, D]
        
        # Concatenate attention masks
        fused_mask = torch.cat([tab_mask, text_mask], dim=1)  # [B, L_tab + L_text]
        
        # Add segment embeddings if enabled
        if self.use_segment_emb:
            seg_ids = torch.cat([
                torch.zeros(L_tab, dtype=torch.long, device=fused_tokens.device),
                torch.ones(L_text, dtype=torch.long, device=fused_tokens.device)
            ])
            seg_ids = seg_ids.unsqueeze(0).expand(B, -1)
            seg_emb = self.seg_emb(seg_ids)
            if L_text > 0:
                text_seg = seg_emb[:, L_tab:, :] * text_presence.unsqueeze(-1)
                seg_emb = torch.cat([seg_emb[:, :L_tab, :], text_seg], dim=1)
            fused_tokens = fused_tokens + seg_emb
        
        # Optional layer normalization
        if self.use_layernorm:
            fused_tokens = self.post_ln(fused_tokens)

        fused_tokens = fused_tokens * fused_mask.unsqueeze(-1).float()

        return fused_tokens, fused_mask
    
    def get_importance_weights(self) -> Tuple[float, float]:
        """
        Get current importance weights for analysis.
        
        Returns:
            Tuple of (tab_weight, text_weight)
        """
        tab_weight = 1 + torch.relu(self.alpha_tab).item()
        text_weight = 1 + torch.relu(self.alpha_text).item()
        return tab_weight, text_weight
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'initial_tab_weight': float(self.alpha_tab.item()),
            'initial_text_weight': float(self.alpha_text.item()),
            'use_layernorm': self.use_layernorm,
            'use_segment_emb': self.use_segment_emb
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        tab_weight, text_weight = self.get_importance_weights()
        return (f"ImportanceWeightedConcat(d_model={self.d_model}, "
                f"tab_weight={tab_weight:.2f}, text_weight={text_weight:.2f}, "
                f"use_layernorm={self.use_layernorm}, use_segment_emb={self.use_segment_emb})")

# src/models/transformers.py

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TinyTextTransformer(nn.Module):
    """
    Lightweight transformer for processing short text sequences from medical test values.
    
    Designed for sequences of 5-20 tokens, this module applies contextual processing
    via transformer layers, mean-pools over the sequence dimension, and projects
    to the target embedding dimension.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 4,
        d_ff: int = 1536,
        n_layers: int = 2,
        D_out: int = 768,
        dropout: float = 0.1
    ):
        """
        Initialize TinyTextTransformer.
        
        Args:
            d_model: Transformer model dimension (should match input embeddings)
            nhead: Number of attention heads  
            d_ff: Feed-forward dimension
            n_layers: Number of transformer layers
            D_out: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.D_out = D_out
        self.dropout = dropout
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # MLP projection head
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, D_out)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input embeddings [N_seq, L, d_model]
            mask: Attention mask [N_seq, L] where 1=valid, 0=padding
            
        Returns:
            Output embeddings [N_seq, D_out]
        """
        # Apply transformer encoding
        # PyTorch expects src_key_padding_mask where True=ignore, so invert our mask
        h = self.encoder(x, src_key_padding_mask=~mask.bool())  # [N_seq, L, d_model]
        
        # Mean pool over sequence dimension
        pooled = self._mean_pool(h, mask)  # [N_seq, d_model]
        
        # Apply MLP projection
        output = self.mlp(pooled)  # [N_seq, D_out]
        
        return output
    
    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pool over sequence dimension with proper mask handling.
        
        Args:
            x: Input tensor [N_seq, L, d_model]
            mask: Attention mask [N_seq, L] where 1=valid, 0=padding
            
        Returns:
            Pooled tensor [N_seq, d_model]
        """
        # Apply mask to embeddings
        masked_x = x * mask.unsqueeze(-1)  # [N_seq, L, d_model]
        
        # Sum over sequence dimension
        sum_x = masked_x.sum(dim=1)  # [N_seq, d_model]
        
        # Count valid tokens per sequence
        valid_counts = mask.sum(dim=1, keepdim=True)  # [N_seq, 1]
        
        # Avoid division by zero (shouldn't happen in practice, but safety first)
        valid_counts = torch.clamp(valid_counts, min=1)
        
        # Compute mean
        mean_x = sum_x / valid_counts  # [N_seq, d_model]
        
        return mean_x
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'nhead': self.nhead,
            'd_ff': self.d_ff,
            'n_layers': self.n_layers,
            'D_out': self.D_out,
            'dropout': self.dropout
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"TinyTextTransformer(d_model={self.d_model}, nhead={self.nhead}, "
                f"d_ff={self.d_ff}, n_layers={self.n_layers}, D_out={self.D_out}, "
                f"dropout={self.dropout})")



class BiCrossAttLayer(nn.Module):
    """
    Bidirectional Cross Attention layer for medical exam data.
    
    Enables rich interaction between tabular exam data and result text through
    separate self-attention and cross-attention mechanisms with learnable fusion.
    
    Architecture:
    1. Self-attention within each modality
    2. Cross-attention between modalities (bidirectional)
    3. Learnable fusion of self and cross-attention outputs
    4. Feed-forward networks for each modality
    
    Always uses pre-norm for stability and separate parameters for each
    cross-attention direction to capture asymmetric relationships.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        learnable_fusion: bool = True,
        asymmetric_fusion: bool = False,
        fusion_d_gate: int = 64,
        fusion_tau: float = 1.0,
        presence_logit_mask: bool = True
    ):
        """
        Initialize BiCrossAttLayer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            attn_dropout: Attention dropout rate
            mlp_dropout: MLP dropout rate
            proj_dropout: Projection dropout rate
            learnable_fusion: Whether to use learnable fusion weights
            asymmetric_fusion: Whether to use different fusion weights per modality
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.learnable_fusion = learnable_fusion
        self.asymmetric_fusion = asymmetric_fusion
        self.fusion_tau = fusion_tau
        self.presence_logit_mask = presence_logit_mask
        self.fusion_d_gate = fusion_d_gate
        # Register large-magnitude constant for presence-aware logit masking
        # Avoid explicit dtype to prevent potential parsing issues in some environments
        self.register_buffer("_fusion_big_m", torch.tensor(1e4))
        
        # Self-attention modules for each modality
        self.self_attn_tab = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.self_attn_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        
        # Cross-attention modules (separate parameters for each direction)
        self.cross_attn_tab2text = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.cross_attn_text2tab = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        
        # Layer normalizations (separate for each modality and position)
        self.norm_tab_attn = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_text_attn = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_tab_ffn = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_text_ffn = nn.LayerNorm(d_model, eps=1e-6)
        
        # Projection dropout
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        # Feed-forward networks (separate for each modality)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp_tab = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(mlp_dropout)
        )
        self.mlp_text = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(mlp_dropout)
        )
        
        # Learnable fusion gates
        gate_input_dim = 2 * d_model + 1

        if learnable_fusion:
            self.text_null_cls = nn.Parameter(torch.zeros(1, d_model))
            self.tab_null_cls = nn.Parameter(torch.zeros(1, d_model))

            def _build_gate():
                linear1 = nn.Linear(gate_input_dim, fusion_d_gate)
                linear2 = nn.Linear(fusion_d_gate, 2)
                nn.init.zeros_(linear2.bias)
                return nn.Sequential(
                    linear1,
                    nn.GELU(),
                    linear2
                )

            if asymmetric_fusion:
                self._tab_gate = _build_gate()
                self._text_gate = _build_gate()
            else:
                shared_gate = _build_gate()
                self._tab_gate = shared_gate
                self._text_gate = shared_gate
        else:
            self.register_buffer("_dummy", torch.tensor(0.0))

        self._last_attn_weights = None
        self._last_fusion_stats = None
    
    def forward(
        self, 
        tab_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        tab_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through bidirectional cross-attention.
        
        Args:
            tab_tokens: [B, L_tab, D] - Tabular data tokens
            text_tokens: [B, L_text, D] - Result text tokens
            tab_mask: [B, L_tab] - Attention mask for tabular tokens (True=valid)
            text_mask: [B, L_text] - Attention mask for text tokens (True=valid)
            
        Returns:
            Tuple of (updated_tab_tokens, updated_text_tokens)
        """
        # Input validation
        assert tab_tokens.size(-1) == self.d_model, f"Expected d_model={self.d_model}, got {tab_tokens.size(-1)}"
        assert text_tokens.size(-1) == self.d_model, f"Expected d_model={self.d_model}, got {text_tokens.size(-1)}"
        
        # Ensure all tensors are on the same device
        device = tab_tokens.device
        if text_tokens.device != device:
            text_tokens = text_tokens.to(device)
        if tab_mask is not None and tab_mask.device != device:
            tab_mask = tab_mask.to(device)
        if text_mask is not None and text_mask.device != device:
            text_mask = text_mask.to(device)
        
        B = tab_tokens.size(0)
        D = tab_tokens.size(-1)

        # Presence indicators (True=valid sequence)
        dtype_f = tab_tokens.dtype

        if tab_mask is None:
            tab_presence = torch.ones(B, dtype=torch.bool, device=device)
        else:
            tab_presence = tab_mask.any(dim=1)

        if text_mask is None:
            text_presence = torch.ones(B, dtype=torch.bool, device=device) if text_tokens.size(1) > 0 else torch.zeros(B, dtype=torch.bool, device=device)
        else:
            text_presence = text_mask.any(dim=1)

        p_tab = tab_presence.to(dtype=dtype_f).unsqueeze(1)          # [B,1]
        p_text = text_presence.to(dtype=dtype_f).unsqueeze(1)        # [B,1]
        p_tab_broadcast = p_tab.unsqueeze(-1)              # [B,1,1]
        p_text_broadcast = p_text.unsqueeze(-1)

        # Store residuals for attention block
        residual_tab = tab_tokens
        residual_text = text_tokens
        
        # Pre-norm for attention
        tab_tokens_norm = self.norm_tab_attn(tab_tokens)
        text_tokens_norm = self.norm_text_attn(text_tokens)
        
        # Convert masks to PyTorch multihead attention format (True = ignore)
        tab_mask_mha = None if tab_mask is None else ~tab_mask.bool()
        text_mask_mha = None if text_mask is None else ~text_mask.bool()
        
        # Self-attention within each modality (skip samples without the modality)
        tab_self_attn = torch.zeros_like(tab_tokens_norm)
        if tab_presence.any():
            tab_idx = torch.nonzero(tab_presence, as_tuple=False).squeeze(-1)
            tab_inputs = tab_tokens_norm[tab_idx]
            tab_mask_subset = None if tab_mask_mha is None else tab_mask_mha[tab_idx]
            tab_self_attn_subset, _ = self.self_attn_tab(
                tab_inputs, tab_inputs, tab_inputs,
                key_padding_mask=tab_mask_subset,
                need_weights=False
            )
            tab_self_attn[tab_idx] = tab_self_attn_subset.to(tab_self_attn.dtype)
        
        text_self_attn = torch.zeros_like(text_tokens_norm)
        if text_presence.any() and text_tokens_norm.size(1) > 0:
            text_idx = torch.nonzero(text_presence, as_tuple=False).squeeze(-1)
            text_inputs = text_tokens_norm[text_idx]
            text_mask_subset = None if text_mask_mha is None else text_mask_mha[text_idx]
            text_self_attn_subset, _ = self.self_attn_text(
                text_inputs, text_inputs, text_inputs,
                key_padding_mask=text_mask_subset,
                need_weights=False
            )
            text_self_attn[text_idx] = text_self_attn_subset.to(text_self_attn.dtype)
        
        text_seq_len = text_tokens_norm.size(1)
        joint_presence = tab_presence & text_presence
        if text_seq_len == 0 or not joint_presence.any():
            tab_cross_attn = torch.zeros_like(tab_self_attn)
            text_cross_attn = torch.zeros_like(text_self_attn)
            self._last_attn_weights = None
        else:
            joint_idx = torch.nonzero(joint_presence, as_tuple=False).squeeze(-1)
            tab_cross_attn = torch.zeros_like(tab_self_attn)
            text_cross_attn = torch.zeros_like(text_self_attn)
            
            text_mask_subset = None if text_mask_mha is None else text_mask_mha[joint_idx]
            tab_mask_subset = None if tab_mask_mha is None else tab_mask_mha[joint_idx]
            
            # Cross-attention between modalities
            tab_cross_subset, tab_cross_weights = self.cross_attn_tab2text(
                tab_tokens_norm[joint_idx], text_tokens_norm[joint_idx], text_tokens_norm[joint_idx],
                key_padding_mask=text_mask_subset,
                need_weights=True,
                average_attn_weights=False  # For 4D tensor shape -> monitoring
            )
            tab_cross_attn[joint_idx] = tab_cross_subset.to(tab_cross_attn.dtype)
            
            # Initialize full attention storage with zeros for absent samples
            attn_shape = (B, self.n_heads, tab_tokens_norm.size(1), text_seq_len)
            full_attn_weights = torch.zeros(attn_shape, device=tab_cross_weights.device, dtype=tab_cross_weights.dtype)
            full_attn_weights[joint_idx] = tab_cross_weights
            self._last_attn_weights = full_attn_weights.detach()
            
            # Text queries attend to tabular keys/values
            text_cross_subset, _ = self.cross_attn_text2tab(
                text_tokens_norm[joint_idx], tab_tokens_norm[joint_idx], tab_tokens_norm[joint_idx],
                key_padding_mask=tab_mask_subset,
                need_weights=False
            )
            text_cross_attn[joint_idx] = text_cross_subset.to(text_cross_attn.dtype)
        
        # Fusion of self and cross-attention outputs
        if self.learnable_fusion:
            # Compute CLS summaries
            c_tab = tab_tokens_norm[:, 0, :]
            if text_seq_len == 0:
                c_text = self.text_null_cls.expand(B, -1)
            else:
                c_text = text_tokens_norm[:, 0, :]

            text_null = self.text_null_cls.expand(B, -1)
            tab_null = self.tab_null_cls.expand(B, -1)

            cond_text = p_text * c_text + (1 - p_text) * text_null
            cond_tab = p_tab * c_tab + (1 - p_tab) * tab_null

            tab_gate_input = torch.cat([c_tab, cond_text, p_text], dim=-1)
            text_gate_input = torch.cat([c_text, cond_tab, p_tab], dim=-1)

            tab_logits = self._tab_gate(tab_gate_input)
            text_logits = self._text_gate(text_gate_input)

            if self.presence_logit_mask:
                no_text = (1 - p_text).squeeze(1)
                no_tab = (1 - p_tab).squeeze(1)
                big_m = self._fusion_big_m.to(dtype=tab_logits.dtype, device=tab_logits.device)
                tab_logits[:, 1] = tab_logits[:, 1] - big_m * no_text
                text_logits[:, 1] = text_logits[:, 1] - big_m * no_tab

            tab_weights = torch.softmax(tab_logits / self.fusion_tau, dim=-1)
            text_weights = torch.softmax(text_logits / self.fusion_tau, dim=-1)

            w_self_tab = tab_weights[:, 0].view(B, 1, 1)
            w_cross_tab = tab_weights[:, 1].view(B, 1, 1)
            w_self_text = text_weights[:, 0].view(B, 1, 1)
            w_cross_text = text_weights[:, 1].view(B, 1, 1)

            tab_fused = w_self_tab * tab_self_attn + w_cross_tab * tab_cross_attn
            text_fused = w_self_text * text_self_attn + w_cross_text * text_cross_attn

            self._last_fusion_stats = {
                "tab_cross_mean": w_cross_tab.mean().detach().item(),
                "text_cross_mean": w_cross_text.mean().detach().item()
            }
        else:
            # Fixed 0.5 weight fusion, optionally presence-aware for stability
            weight_self = 0.5
            weight_cross = 0.5
            w_self_tab = torch.full((B, 1, 1), weight_self, dtype=tab_self_attn.dtype, device=tab_self_attn.device)
            w_cross_tab = torch.full((B, 1, 1), weight_cross, dtype=tab_self_attn.dtype, device=tab_self_attn.device)
            w_self_text = w_self_tab.clone()
            w_cross_text = w_cross_tab.clone()
            if self.presence_logit_mask:
                w_cross_tab = w_cross_tab * p_text_broadcast
                w_self_tab = 1.0 - w_cross_tab
                w_cross_text = w_cross_text * p_tab_broadcast
                w_self_text = 1.0 - w_cross_text
            tab_fused = w_self_tab * tab_self_attn + w_cross_tab * tab_cross_attn
            text_fused = w_self_text * text_self_attn + w_cross_text * text_cross_attn
            self._last_fusion_stats = {
                "tab_cross_mean": w_cross_tab.mean().detach().item(),
                "text_cross_mean": w_cross_text.mean().detach().item()
            }
        
        # Apply projection dropout
        tab_fused = self.proj_dropout(tab_fused)
        text_fused = self.proj_dropout(text_fused)
        
        # Residual connection for attention block
        tab_tokens = residual_tab + tab_fused
        text_tokens = residual_text + text_fused
        
        # Mask out invalid positions to avoid propagating garbage values
        if tab_mask is not None:
            tab_tokens = tab_tokens * tab_mask.unsqueeze(-1).to(dtype=tab_tokens.dtype)
        if text_mask is not None:
            text_tokens = text_tokens * text_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
        
        # Store residuals for FFN block
        residual_tab = tab_tokens
        residual_text = text_tokens
        
        # Pre-norm for FFN
        tab_tokens = self.norm_tab_ffn(tab_tokens)
        text_tokens = self.norm_text_ffn(text_tokens)
        
        # Feed-forward networks with residual connections
        tab_tokens = residual_tab + self.mlp_tab(tab_tokens)
        text_tokens = residual_text + self.mlp_text(text_tokens)
        
        if tab_mask is not None:
            tab_tokens = tab_tokens * tab_mask.unsqueeze(-1).to(dtype=tab_tokens.dtype)
        if text_mask is not None:
            text_tokens = text_tokens * text_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
        
        return tab_tokens, text_tokens
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'mlp_ratio': float(self.mlp_tab[0].out_features / self.d_model),
            'learnable_fusion': self.learnable_fusion,
            'asymmetric_fusion': self.asymmetric_fusion,
            'fusion_d_gate': self.fusion_d_gate,
            'fusion_tau': self.fusion_tau,
            'presence_logit_mask': self.presence_logit_mask
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"BiCrossAttLayer(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"learnable_fusion={self.learnable_fusion}, "
                f"asymmetric_fusion={self.asymmetric_fusion})")



class UniTransformerLayer(nn.Module):
    """
    Unified transformer layer using PyTorch's TransformerEncoderLayer.
    
    A clean wrapper around PyTorch's standard transformer encoder layer with
    pre-norm architecture for training stability. Provides consistent interface
    for the unified multi-modal sequence processing.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        """
        Initialize UniTransformerLayer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            mlp_ratio: FFN hidden dimension ratio (hidden_dim = d_model * mlp_ratio)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Calculate FFN hidden dimension
        dim_feedforward = int(d_model * mlp_ratio)
        
        # Core transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer layer.
        
        Args:
            x: Input tokens [B, S, d_model]
            attention_mask: Attention mask [B, S] where True=valid, False=padding
            
        Returns:
            Output tokens [B, S, d_model]
        """
        # Convert mask to PyTorch format (True = ignore for padding)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # Invert: True = ignore padding
        
        # Apply transformer layer
        output = self.transformer_layer(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"UniTransformerLayer(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"mlp_ratio={self.mlp_ratio}, dropout={self.dropout})")



class IndCausalTransformer(nn.Module):
    """
    Individual-level causal transformer for exam sequence modeling.
    
    Processes sequences of exam embeddings for each individual using causal
    self-attention to capture temporal dependencies in a patient's medical history.
    Uses smaller architecture appropriate for higher-level individual modeling.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize IndCausalTransformer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads (smaller than exam-level)
            mlp_ratio: FFN hidden dimension ratio (smaller than exam-level)
            n_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Calculate FFN hidden dimension
        dim_feedforward = int(d_model * mlp_ratio)
        
        # Stack of causal transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            )
            for _ in range(n_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the causal transformer.
        
        Args:
            x: Input exam sequence [B', E_max, d_model] where B'=individuals, E_max=max exams
            attention_mask: Attention mask [B', E_max] where True=valid, False=padding
            
        Returns:
            Output exam sequence [B', E_max, d_model] with causal modeling
        """
        B_prime, E_max, D = x.shape
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.ones(E_max, E_max, device=x.device, dtype=torch.bool), 
            diagonal=1
        )  # Upper triangular = True (mask out future), lower triangular = False (allow past)
        
        # Convert padding mask to PyTorch format (True = ignore for padding)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # Invert: True = ignore padding
        
        # Apply each transformer layer with causal attention
        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,  # Causal mask for autoregressive modeling
                src_key_padding_mask=src_key_padding_mask,  # Padding mask
                is_causal=True  # Enable causal attention optimization
            )
        
        return x
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'mlp_ratio': self.mlp_ratio,
            'n_layers': self.n_layers,
            'dropout': self.dropout
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"IndCausalTransformer(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"mlp_ratio={self.mlp_ratio}, n_layers={self.n_layers}, "
                f"dropout={self.dropout})")

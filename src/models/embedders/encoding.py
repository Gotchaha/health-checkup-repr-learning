# src/models/embedders/encoding.py

import math
import torch
import torch.nn as nn
from typing import Union


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for medical result text sequences.
    
    Adds learned position information to result text embeddings following
    the standard approach from "Attention Is All You Need" (Vaswani et al.).
    
    Used specifically for result_emb sequences after TabEmbedder processing.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length (conservative for medical text)
        dropout: Dropout probability after adding positional encoding
        scale: Whether to scale input embeddings by sqrt(d_model) before adding PE
        device: Device for computations
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 1024,
        dropout: float = 0.1,
        scale: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_p = dropout
        self.scale = scale
        self.device = device
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        
        # Frequency factor: 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (saved with model, not trainable)
        self.register_buffer('pe', pe)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        if scale:
            self.scale_factor = math.sqrt(d_model)
        else:
            self.scale_factor = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [B, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding added [B, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Ensure input is on same device
        if x.device != self.pe.device:
            x = x.to(self.pe.device)
        
        # Apply scaling if enabled
        if self.scale:
            x = x * self.scale_factor
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        # Apply dropout
        return self.dropout(x)
    
    def to(self, device: Union[str, torch.device]) -> "SinusoidalPositionalEncoding":
        """Move module to specified device."""
        self.device = str(device)
        return super().to(device)
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'max_len': self.max_len,
            'dropout': self.dropout_p,
            'scale': self.scale,
            'device': self.device
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"SinusoidalPositionalEncoding(d_model={self.d_model}, "
                f"max_len={self.max_len}, dropout={self.dropout_p}, "
                f"scale={self.scale})")


class TimeAwarePositionalEmbedding(nn.Module):
    """
    Time-aware positional embedding for individual exam sequences.
    
    Implements the approach from Zhang et al. FATA-Trans (CIKM '23):
        TP(i) = w_p * i + w_t * (t_i / time_scale) + b
        P(i, 2j)   = sin(TP(i) / 10000^(2j / d_model))
        P(i, 2j+1) = cos(TP(i) / 10000^(2j / d_model))
    
    Used before IndCausalTransformer to encode both exam order and time intervals
    for individual patient histories.
    
    Args:
        d_model: Embedding dimension
        time_scale: Time normalization factor (365.0 = yearly scale)
        dropout: Dropout probability after embedding
        device: Device for computations
    """
    
    def __init__(
        self,
        d_model: int,
        time_scale: float = 365.0,  # Normalize time by days (365 = 1 year)
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize TimeAwarePositionalEmbedding.
        
        Args:
            d_model: Embedding dimension
            time_scale: Time normalization factor (365.0 = yearly scale)
            dropout: Dropout probability after embedding
            device: Device for computations
        """
        super().__init__()
        
        self.d_model = d_model
        self.time_scale = time_scale
        self.dropout_p = dropout
        self.device = device
        
        # Learnable parameters from FATA-Trans
        self.w_p = nn.Parameter(torch.tensor(1.0))  # Position weight
        self.w_t = nn.Parameter(torch.tensor(1.0))  # Time weight  
        self.b = nn.Parameter(torch.tensor(0.0))    # Bias term
        
        # Precompute inverse frequencies for sinusoidal encoding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2, dtype=torch.float) / d_model))
        self.register_buffer("inv_freq", inv_freq.to(device))
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Move to device
        if device != "cpu":
            self.to(device)
    
    def forward(
        self, 
        positions: torch.Tensor,        # [B', E_max] position indices
        times: torch.Tensor,            # [B', E_max] time intervals in days
        attention_mask: torch.Tensor    # [B', E_max] valid exam mask
    ) -> torch.Tensor:
        """
        Generate time-aware positional embeddings.
        
        Args:
            positions: Position indices [B', E_max]
            times: Time intervals in days [B', E_max] 
            attention_mask: Valid exam mask [B', E_max]
            
        Returns:
            Time-aware positional embeddings [B', E_max, d_model]
        """
        # Ensure inputs are on same device
        device = next(self.parameters()).device
        if positions.device != device:
            positions = positions.to(device)
        if times.device != device:
            times = times.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        
        # Normalize time intervals by scale factor
        times_normalized = times.float() / self.time_scale
        
        # Compute TP(i) = w_p * i + w_t * t_i + b
        TP = positions.float() * self.w_p + times_normalized * self.w_t + self.b
        
        # Compute angles for sinusoidal encoding
        angles = TP.unsqueeze(-1) * self.inv_freq  # [B', E_max, d_model/2]
        
        # Apply sinusoidal encoding
        emb = torch.zeros(*angles.shape[:-1], self.d_model, device=device)
        emb[..., 0::2] = torch.sin(angles)  # Even dimensions
        emb[..., 1::2] = torch.cos(angles)  # Odd dimensions
        
        # Zero out embeddings for padded positions
        emb = emb * attention_mask.unsqueeze(-1).float()
        
        # Apply dropout
        return self.dropout(emb)
    
    def to(self, device: Union[str, torch.device]) -> "TimeAwarePositionalEmbedding":
        """Move module to specified device."""
        self.device = str(device)
        return super().to(device)
    
    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'd_model': self.d_model,
            'time_scale': self.time_scale,
            'dropout': self.dropout_p,
            'device': self.device
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"TimeAwarePositionalEmbedding(d_model={self.d_model}, "
                f"time_scale={self.time_scale}, dropout={self.dropout_p})")
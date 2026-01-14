# src/models/embedders/CategoricalEmbedder.py

"""
CategoricalEmbedder: A module for embedding categorical medical test codes and values.

This module provides an embedder for categorical features in medical test data,
with separate methods for mapping tokens to IDs and embedding those IDs.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any


class CategoricalEmbedder(nn.Module):
    """
    Embedder for categorical test codes and values.
    
    This class maps categorical test codes and their values to embeddings
    using a learned embedding matrix. It provides separate methods for
    the mapping (tokenization) and embedding steps.
    """
    
    def __init__(
        self,
        vocab_path: str,
        embedding_dim: int = 768,
        trainable: bool = True,
        padding_idx: int = 0,
        use_xavier_init: bool = True,
        output_dir: str = "outputs/embedders/categorical_embedder",
        device: str = "cpu"
    ):
        """
        Initialize the CategoricalEmbedder.
        
        Args:
            vocab_path: Path to vocabulary YAML file
            embedding_dim: Dimension of the embeddings
            trainable: Whether embeddings should be updated during training
            padding_idx: Index to use for padding (typically 0)
            use_xavier_init: Whether to use Xavier initialization for embeddings
            output_dir: Directory for saving embedder outputs
            device: Device for embedding computations ("cpu" or "cuda:x")
        """
        super().__init__()
        
        # Store parameters
        self.vocab_path = vocab_path
        self.embedding_dim = embedding_dim
        self.trainable = trainable
        self.padding_idx = padding_idx
        self.use_xavier_init = use_xavier_init
        self.output_dir = output_dir
        self.device = device
        
        # Load vocabulary
        self._load_vocabulary(vocab_path)
        
        # Initialize embedding layer
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # Apply Xavier initialization if requested
        if self.use_xavier_init:
            init.xavier_uniform_(self.embeddings.weight.data)
            
        # Set trainable state
        self.embeddings.weight.requires_grad = self.trainable
        
        # Move to device if specified
        if device != "cpu":
            self.to(device)
    
    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> "CategoricalEmbedder":
        """
        Create an embedder from a YAML config file.
        
        Args:
            config_path: Path to YAML config file
            **kwargs: Override config parameters
            
        Returns:
            Configured CategoricalEmbedder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with kwargs
        config.update(kwargs)
        
        return cls(**config)
    
    @classmethod
    def from_pretrained(cls, pretrained_dir: str) -> "CategoricalEmbedder":
        """
        Load a pretrained embedder from a directory.
        
        Args:
            pretrained_dir: Directory containing saved embedder
            
        Returns:
            Loaded CategoricalEmbedder instance
        """
        config_path = os.path.join(pretrained_dir, "config.yaml")
        
        # Initialize from saved config
        embedder = cls.from_config(config_path)
        
        # Load saved weights
        weights_path = os.path.join(pretrained_dir, "embedding_weights.pt")
        state_dict = torch.load(weights_path, map_location="cpu")
        embedder.load_state_dict(state_dict)
        
        return embedder
    
    def _load_vocabulary(self, vocab_path: str) -> None:
        """
        Load vocabulary from a YAML file.
        
        Args:
            vocab_path: Path to vocabulary YAML file
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = yaml.safe_load(f)
        
        self.token_to_idx = vocab_data["token_to_idx"]
        
        # Convert string keys back to integers for idx_to_token
        self.idx_to_token = {}
        for k, v in vocab_data["idx_to_token"].items():
            self.idx_to_token[int(k)] = v
            
        self.vocab_size = vocab_data["vocab_size"]
        self.metadata = vocab_data.get("metadata", {})
    
    def map(self, 
            tokens: Union[List[str], List[List[str]]],
            device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Map string tokens to token IDs.
        
        Args:
            tokens: List of tokens, or batch of token lists
            device: Device to place the resulting tensor on
            
        Returns:
            Tensor of token IDs
        """
        if not isinstance(tokens[0], list):
            # Single sequence
            return self._map_single(tokens, device)
        else:
            # Batch of sequences
            return self._map_batch(tokens, device)
    
    def _map_single(self, 
                    tokens: List[str],
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """Map a single sequence of tokens to token IDs."""
        token_ids = [self.token_to_idx.get(token, self.token_to_idx["<UNK>"]) for token in tokens]
        
        # Convert to tensor
        result = torch.tensor(token_ids, dtype=torch.long)
        if device is not None:
            result = result.to(device)
            
        return result
    
    def _map_batch(self, 
                   batch_tokens: List[List[str]],
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """Map a batch of token sequences to token IDs."""
        batch_token_ids = []
        
        for tokens in batch_tokens:
            token_ids = self._map_single(tokens, device=None)  # Don't move to device yet
            batch_token_ids.append(token_ids)
            
        # Pad to same length
        max_len = max(ids.size(0) for ids in batch_token_ids)
        padded_batch = []
        
        for token_ids in batch_token_ids:
            if token_ids.size(0) < max_len:
                padding = torch.full((max_len - token_ids.size(0),), 
                                    self.padding_idx, 
                                    dtype=torch.long)
                token_ids = torch.cat([token_ids, padding])
            padded_batch.append(token_ids)
            
        # Stack and move to device
        result = torch.stack(padded_batch)
        if device is not None:
            result = result.to(device)
            
        return result
    
    def decode(self, token_ids: torch.Tensor) -> Union[List[str], List[List[str]]]:
        """
        Convert token IDs back to test codes and values.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of tokens or list of lists of tokens
        """
        if token_ids.dim() == 1:
            # Single sequence
            return [self.idx_to_token.get(idx.item(), "<UNK>") for idx in token_ids]
        else:
            # Batch of sequences
            return [
                [self.idx_to_token.get(idx.item(), "<UNK>") for idx in seq if idx.item() != self.padding_idx]
                for seq in token_ids
            ]
    
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Tensor of embeddings
        """
        return self.embeddings(token_ids)
    
    def forward(self, 
                tokens: Union[List[str], List[List[str]]],
                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Map tokens to IDs and then to embeddings.
        
        Args:
            tokens: List of tokens, or batch of token lists
            device: Device to place tensors on
            
        Returns:
            Tensor of embeddings
        """
        if device is not None:
            self.to(device)
            
        token_ids = self.map(tokens, device)
        embeddings = self.embed(token_ids)
        
        return embeddings
    
    def to_device(self, device: Union[str, torch.device]) -> "CategoricalEmbedder":
        """Move embedder to specified device."""
        self.to(device)
        return self
    
    def get_token_id(self, token: str) -> int:
        """Get ID for a specific token."""
        return self.token_to_idx.get(token, self.token_to_idx["<UNK>"])
    
    def get_token(self, idx: int) -> str:
        """Get token for a specific ID."""
        return self.idx_to_token.get(idx, "<UNK>")
    
    def save_pretrained(self, output_dir: Optional[str] = None) -> str:
        """
        Save embedder weights and configuration.
        
        Args:
            output_dir: Directory to save to (uses self.output_dir if None)
            
        Returns:
            Path to saved directory
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embedding weights
        weights_path = os.path.join(output_dir, "embedding_weights.pt")
        torch.save(self.state_dict(), weights_path)
        
        # Save configuration
        config = {
            "vocab_path": self.vocab_path,
            "embedding_dim": self.embedding_dim,
            "trainable": self.trainable,
            "padding_idx": self.padding_idx,
            "use_xavier_init": self.use_xavier_init,
            "output_dir": output_dir,
            "device": self.device
        }
        
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        return output_dir
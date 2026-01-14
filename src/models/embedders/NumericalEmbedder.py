# src/models/embedders/NumericalEmbedder.py
import os
import yaml
import logging
import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class NumericalEmbedder(nn.Module):
    """
    Numerical feature embedder using Random Fourier Features (RFF).
    
    Based on Tancik et al. (NeurIPS 2020) "Fourier Features Let Networks Learn 
    High Frequency Functions in Low Dimensional Domains", this implementation
    uses fixed random frequencies sampled from a Gaussian distribution to
    transform numerical values into high-dimensional embeddings.
    
    Key features:
    - Zero-statistics normalization: x' = x / (1 + |x|)
    - Fixed random frequencies (no trainable frequency parameters)
    - Robust handling of medical test values with diverse ranges
    """
    
    def __init__(
        self,
        d_embedding: int = 768,
        n_bands: int = 16,
        sigma: float = 1.0,
        bias: bool = False,
        seed: Optional[int] = None,
        output_dir: str = "outputs/embedders/numerical_embedder",
        trainable: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the numerical embedder with Random Fourier Features.
        
        Args:
            d_embedding: Dimension of the output embeddings
            n_bands: Number of random frequency bands (m in paper)
            sigma: Standard deviation for frequency sampling (controls frequency scale)
            bias: Whether to include bias in the projection layer (default: False for theoretical purity)
            seed: Optional random seed for reproducible frequency sampling
            output_dir: Directory for saving model and metadata
            trainable: Whether projection weights should be trainable
            device: Device for embedding computations ("cpu" or "cuda:x")
        """
        super().__init__()
        self.d_embedding = d_embedding
        self.n_bands = n_bands
        self.sigma = sigma
        self.bias = bias
        self.seed = seed
        self.output_dir = output_dir
        self.trainable = trainable
        self.device = device
        
        # Sample fixed random frequencies from N(0, σ²)
        # Following Tancik et al.: "each entry in B is sampled from N(0, σ²)"
        # Optional seed for reproducible frequency sampling (research reproducibility)
        if seed is not None:
            # Save current random state for isolation
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            freq = torch.randn(n_bands) * sigma
            # Restore random state to avoid affecting other components
            torch.set_rng_state(rng_state)
        else:
            freq = torch.randn(n_bands) * sigma
        self.register_buffer('random_freq', freq)
        
        # Linear projection from 2m RFF features to d_embedding
        # bias=False by default for theoretical purity (following Tancik et al.)
        self.proj = nn.Linear(2 * n_bands, d_embedding, bias=bias)
        
        # Set trainability
        self._set_trainable_internal(trainable)
        
        # Move to device if specified
        if device != "cpu":
            self.to(device)
    
    def _set_trainable_internal(self, trainable: bool) -> None:
        """Internal method to set trainable state of weights."""
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = trainable
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform numerical features to embeddings using Random Fourier Features.
        
        Args:
            x: Tensor of numerical values, shape [..., n_features]
                
        Returns:
            Tensor of embeddings, shape [..., n_features, d_embedding]
        """
        # Move input to the same device as model parameters if needed
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        
        # Handle NaN and infinite values
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Zero-statistics normalization: x' = x / (1 + |x|)
        # Maps any real number to (-1, 1) smoothly, preserving sign
        x_norm = x / (1 + x.abs())
        
        # Apply Random Fourier Features transformation
        # Following Tancik et al.: γ(v) = [cos(2πBv), sin(2πBv)]ᵀ
        x_scaled = 2 * math.pi * x_norm.unsqueeze(-1) * self.random_freq  # [..., n_features, n_bands]
        
        # Concatenate cosine and sine features (block order)
        fourier_feats = torch.cat([
            torch.cos(x_scaled),  # [..., n_features, n_bands]
            torch.sin(x_scaled)   # [..., n_features, n_bands]
        ], dim=-1)  # [..., n_features, 2*n_bands]
        
        # Project to final embedding dimension
        emb = self.proj(fourier_feats)  # [..., n_features, d_embedding]
        
        return emb
    
    def get_output_shape(self) -> torch.Size:
        """Get the output shape without batch dimensions."""
        return torch.Size([self.d_embedding])  # Per feature embedding dimension
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            int: Embedding dimension
        """
        return self.d_embedding
    
    def get_config(self) -> Dict:
        """Get the configuration parameters."""
        return {
            "d_embedding": self.d_embedding,
            "n_bands": self.n_bands,
            "sigma": self.sigma,
            "bias": self.bias,
            "seed": self.seed,
            "trainable": self.trainable,
            "device": self.device,
            "output_dir": self.output_dir
        }
    
    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> "NumericalEmbedder":
        """
        Create an embedder from a YAML config file.
        
        Args:
            config_path: Path to YAML config file
            **kwargs: Override config parameters
            
        Returns:
            Configured NumericalEmbedder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with kwargs
        config.update(kwargs)
        
        return cls(**config)
    
    def process(self, numerical_data: torch.Tensor) -> torch.Tensor:
        """
        Complete pipeline: embed numerical features in one step.
        
        Args:
            numerical_data: Input numerical features [..., n_features]
            
        Returns:
            Embeddings [..., n_features, d_embedding]
        """
        if numerical_data.device != self._get_device_from_params():
            numerical_data = numerical_data.to(self._get_device_from_params())
            
        return self.forward(numerical_data)
    
    def _get_device_from_params(self) -> torch.device:
        """Helper to get current actual device from parameters."""
        return next(self.parameters()).device
    
    def to(self, device: Union[str, torch.device]) -> "NumericalEmbedder":
        """
        Move embedder to specified device.
        
        Args:
            device: Device to move to ("cpu" or "cuda:x")
            
        Returns:
            Self (for method chaining)
        """
        self.device = device
        return super().to(device)
    
    def set_trainable(self, trainable: bool) -> "NumericalEmbedder":
        """
        Set whether embedding weights should be trainable.
        
        Args:
            trainable: Whether embedding weights should be trainable
            
        Returns:
            Self (for method chaining)
        """
        self._set_trainable_internal(trainable)
        return self
    
    def is_trainable(self) -> bool:
        """
        Check if embedding weights are trainable.
        
        Returns:
            True if trainable, False if frozen
        """
        return self.trainable
    
    def save_pretrained(self, directory: Optional[str] = None) -> None:
        """
        Save embedder state to disk.
        
        Args:
            directory: Directory to save embedder state, defaults to self.output_dir
        """
        directory = directory or self.output_dir
        os.makedirs(directory, exist_ok=True)
        
        # Save the embedding weights and buffers
        torch.save(self.state_dict(), os.path.join(directory, "numerical_embedder.pt"))
        
        # Save metadata
        with open(os.path.join(directory, "numerical_embedder_metadata.yaml"), "w") as f:
            yaml.dump(self.get_config(), f, default_flow_style=False)
            
        logger.info(f"Saved NumericalEmbedder to {directory}")
            
    @classmethod
    def from_pretrained(cls, directory: str, device: Optional[str] = None) -> "NumericalEmbedder":
        """
        Load embedder from saved state.
        
        Args:
            directory: Directory with saved embedder state
            device: Optional device override
            
        Returns:
            Loaded NumericalEmbedder instance
        """
        # Load metadata
        metadata_path = os.path.join(directory, "numerical_embedder_metadata.yaml")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"No metadata file found at {metadata_path}")
        
        # Override device if specified
        if device is not None:
            config["device"] = device
        
        # Create instance
        instance = cls(**config)
        
        # Load weights and buffers
        weight_path = os.path.join(directory, "numerical_embedder.pt")
        instance.load_state_dict(torch.load(weight_path, map_location=config["device"]))
        
        logger.info(f"Loaded NumericalEmbedder from {directory}")
        
        return instance
    
    def __repr__(self) -> str:
        """String representation with key information."""
        return (f"NumericalEmbedder("
                f"d_embedding={self.d_embedding}, "
                f"n_bands={self.n_bands}, "
                f"sigma={self.sigma:.3f}, "
                f"bias={self.bias}, "
                f"trainable={self.trainable})")
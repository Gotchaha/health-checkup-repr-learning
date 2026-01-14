# src/models/embedders/factory.py

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .TextEmbedder import TextEmbedder
from .CategoricalEmbedder import CategoricalEmbedder
from .NumericalEmbedder import NumericalEmbedder


class EmbedderBundle:
    """
    Simple container for the three base embedders used in medical exam processing.
    
    Provides unified interface for device management, save/load operations,
    and convenient access to individual embedders.
    """
    
    def __init__(
        self,
        text_embedder: TextEmbedder,
        categorical_embedder: CategoricalEmbedder,
        numerical_embedder: NumericalEmbedder
    ):
        """
        Initialize EmbedderBundle.
        
        Args:
            text_embedder: Text embedder for test values and result text
            categorical_embedder: Categorical embedder for codes and categorical values
            numerical_embedder: Numerical embedder for numerical test values
        """
        self.text = text_embedder
        self.categorical = categorical_embedder
        self.numerical = numerical_embedder
    
    def to(self, device: Union[str, int]) -> "EmbedderBundle":
        """
        Move all embedders to specified device.
        
        Args:
            device: Device to move to ("cpu", "cuda", "cuda:0", etc.)
            
        Returns:
            Self for method chaining
        """
        self.text.to(device)
        self.categorical.to_device(device)
        self.numerical.to(device)
        return self
    
    def train(self) -> "EmbedderBundle":
        """Set all embedders to training mode."""
        self.text.embedding_layer.train()
        self.categorical.train()
        self.numerical.train()
        return self
    
    def eval(self) -> "EmbedderBundle":
        """Set all embedders to evaluation mode."""
        self.text.embedding_layer.eval()
        self.categorical.eval()
        self.numerical.eval()
        return self
    
    def set_trainable(self, trainable: bool) -> "EmbedderBundle":
        """
        Set trainable status for all embedders.
        
        Args:
            trainable: Whether embedders should be trainable
            
        Returns:
            Self for method chaining
        """
        self.text.set_trainable(trainable)
        self.categorical.set_trainable(trainable)
        self.numerical.set_trainable(trainable)
        return self

    def parameters(self):
        """Yield parameters from all embedders."""
        yield from self.text.parameters()
        yield from self.categorical.parameters() 
        yield from self.numerical.parameters()
    
    def save_all(self, directory: Union[str, Path]) -> Dict[str, Path]:
        """
        Save all embedders to directory.
        
        Args:
            directory: Directory to save embedders
            
        Returns:
            Dictionary mapping embedder type to saved path
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save individual embedders to subdirectories
        text_dir = directory / "text_embedder"
        self.text.save_pretrained(text_dir)
        paths['text'] = text_dir
        
        cat_dir = directory / "categorical_embedder"
        self.categorical.save_pretrained(cat_dir)
        paths['categorical'] = cat_dir
        
        num_dir = directory / "numerical_embedder"
        self.numerical.save_pretrained(num_dir)
        paths['numerical'] = num_dir
        
        # Save bundle metadata
        bundle_meta = {
            'embedder_types': ['text', 'categorical', 'numerical'],
            'text_vocab_size': self.text.get_vocab_size(),
            'categorical_vocab_size': self.categorical.vocab_size,
            'embedding_dimensions': {
                'text': self.text.get_embedding_dimension(),
                'categorical': self.categorical.embedding_dim,
                'numerical': self.numerical.get_embedding_dimension()
            }
        }
        
        meta_path = directory / "bundle_metadata.yaml"
        with open(meta_path, 'w') as f:
            yaml.dump(bundle_meta, f, default_flow_style=False)
        
        return paths
    
    @classmethod
    def load_all(cls, directory: Union[str, Path]) -> "EmbedderBundle":
        """
        Load all embedders from directory.
        
        Args:
            directory: Directory containing saved embedders
            
        Returns:
            Loaded EmbedderBundle instance
        """
        directory = Path(directory)
        
        # Load individual embedders
        text_embedder = TextEmbedder.from_pretrained(directory / "text_embedder")
        cat_embedder = CategoricalEmbedder.from_pretrained(directory / "categorical_embedder")
        num_embedder = NumericalEmbedder.from_pretrained(directory / "numerical_embedder")
        
        return cls(text_embedder, cat_embedder, num_embedder)
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for all embedders.
        
        Returns:
            Dictionary mapping embedder type to vocabulary size
        """
        return {
            'text': self.text.get_vocab_size(),
            'categorical': self.categorical.vocab_size,
            'numerical': None  # Numerical embedder doesn't have vocabulary
        }
    
    def get_embedding_dims(self) -> Dict[str, int]:
        """
        Get embedding dimensions for all embedders.
        
        Returns:
            Dictionary mapping embedder type to embedding dimension
        """
        return {
            'text': self.text.get_embedding_dimension(),
            'categorical': self.categorical.embedding_dim,
            'numerical': self.numerical.get_embedding_dimension()
        }
    
    def __repr__(self) -> str:
        """String representation with key information."""
        vocab_sizes = self.get_vocab_sizes()
        dims = self.get_embedding_dims()
        
        return (f"EmbedderBundle(\n"
                f"  text: vocab={vocab_sizes['text']}, dim={dims['text']}\n"
                f"  categorical: vocab={vocab_sizes['categorical']}, dim={dims['categorical']}\n"
                f"  numerical: dim={dims['numerical']}\n"
                f")")


def create_embedders_from_config(
    embedders_config: Dict[str, Any],
    device: str = "cpu"
) -> EmbedderBundle:
    """
    Create embedders from unified configuration.
    
    Args:
        embedders_config: Configuration dictionary for embedders section
        device: Device to place embedders on
        
    Returns:
        EmbedderBundle with configured embedders
        
    Example:
        config = {
            'text': {
                'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
                'max_length': 512,
                'add_phi_tokens': True
            },
            'categorical': {
                'vocab_path': 'config/embedders/cat_vocab.yaml',
                'embedding_dim': 768
            },
            'numerical': {
                'd_embedding': 768,
                'n_bands': 32,
                'sigma': 1.0
            }
        }
        embedders = create_embedders_from_config(config)
    """
    # Validate config structure
    required_keys = ['text', 'categorical', 'numerical']
    for key in required_keys:
        if key not in embedders_config:
            raise ValueError(f"Missing required embedder config: {key}")
    
    # Create text embedder
    text_config = embedders_config['text'].copy()
    text_config['device'] = device
    text_embedder = TextEmbedder(**text_config)
    
    # Create categorical embedder
    cat_config = embedders_config['categorical'].copy()
    cat_config['device'] = device
    cat_embedder = CategoricalEmbedder(**cat_config)
    
    # Create numerical embedder
    num_config = embedders_config['numerical'].copy()
    num_config['device'] = device
    num_embedder = NumericalEmbedder(**num_config)
    
    # Bundle and return
    bundle = EmbedderBundle(text_embedder, cat_embedder, num_embedder)
    return bundle


def validate_embedder_dimensions(embedders: EmbedderBundle) -> None:
    """
    Validate that all embedders have compatible dimensions.
    
    Args:
        embedders: EmbedderBundle to validate
        
    Raises:
        ValueError: If dimensions are incompatible
    """
    dims = embedders.get_embedding_dims()
    
    # Check that all dimensions match for downstream fusion
    text_dim = dims['text']
    cat_dim = dims['categorical']
    num_dim = dims['numerical']
    
    if not (text_dim == cat_dim == num_dim):
        raise ValueError(
            f"Embedder dimensions must match for fusion: "
            f"text={text_dim}, categorical={cat_dim}, numerical={num_dim}"
        )


def create_embedders_for_training(
    config: Dict[str, Any],
    device: str = "cpu",
    validate_dims: bool = True
) -> EmbedderBundle:
    """
    Convenience function to create embedders for training pipeline.
    
    Args:
        config: Full experiment configuration
        device: Device to place embedders on
        validate_dims: Whether to validate dimension compatibility
        
    Returns:
        EmbedderBundle ready for training
    """
    # Extract embedders config from full config
    if 'model' not in config or 'embedders' not in config['model']:
        raise ValueError("Config must contain 'model.embedders' section")
    
    embedders_config = config['model']['embedders']
    
    # Create embedders
    embedders = create_embedders_from_config(embedders_config, device=device)
    
    # Validate dimensions if requested
    if validate_dims:
        validate_embedder_dimensions(embedders)
    
    # Set training mode by default
    embedders.train()
    
    return embedders
# src/models/embedders/__init__.py

"""
Multi-modal embedders for medical examination data.

This module provides embedders for different modalities in medical exam data:
- Text: Medical narratives and text test values
- Categorical: Test codes and categorical test values  
- Numerical: Numerical test values
- Tabular: Multi-modal fusion of all test data
- Encoding: Positional encodings for sequences

Usage:
    # Individual embedder usage
    from src.models.embedders import TextEmbedder, CategoricalEmbedder, NumericalEmbedder
    
    # Factory usage for training pipeline
    from src.models.embedders import create_embedders_from_config
    embedders = create_embedders_from_config(config['model']['embedders'])
    
    # Tabular fusion
    from src.models.embedders import TabEmbedder
    tab_embedder = TabEmbedder(D=768)
    
    # Positional encodings
    from src.models.embedders import SinusoidalPositionalEncoding, TimeAwarePositionalEmbedding
"""

# Core embedder classes
from .TextEmbedder import TextEmbedder
from .CategoricalEmbedder import CategoricalEmbedder
from .NumericalEmbedder import NumericalEmbedder
from .TabEmbedder import TabEmbedder

# Positional encoding modules
from .encoding import (
    SinusoidalPositionalEncoding,
    TimeAwarePositionalEmbedding
)

# Factory functions and bundle for convenient usage
from .factory import (
    EmbedderBundle,
    create_embedders_from_config,
    create_embedders_for_training,
    validate_embedder_dimensions
)

# Define what gets imported with "from src.models.embedders import *"
__all__ = [
    # Core embedders
    'TextEmbedder',
    'CategoricalEmbedder', 
    'NumericalEmbedder',
    'TabEmbedder',
    
    # Positional encodings
    'SinusoidalPositionalEncoding',
    'TimeAwarePositionalEmbedding',
    
    # Factory and utilities
    'EmbedderBundle',
    'create_embedders_from_config',
    'create_embedders_for_training',
    'validate_embedder_dimensions'
]
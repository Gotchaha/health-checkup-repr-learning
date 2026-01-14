# src/models/__init__.py

"""
Models and data processing components for medical examination analysis.

This module provides all the core components needed for multi-modal medical data processing:
- Data loading: Dataset, finite/infinite samplers, and collation functions
- Embedders: Text, categorical, numerical, and tabular embedders
- Architecture: Transformers and fusion components for multi-modal learning
- Main SSL Model: Complete self-supervised learning model orchestrating all components
- Utilities: Helper functions for data preparation and processing

Usage:
    # Data loading pipeline
    from src.models import HealthExamDataset, PersonBatchSampler, InfinitePersonBatchSampler, collate_exams
    
    # Main SSL model
    from src.models import MedicalSSLModel, ModelOutputs
    
    # Embedders (full factory)
    from src.models import create_embedders_from_config, TabEmbedder
    
    # Architecture components
    from src.models import BiCrossAttLayer, TextCompressor, UniTransformerLayer
    
    # Individual embedders if needed
    from src.models import TextEmbedder, CategoricalEmbedder, NumericalEmbedder
"""

# Data loading and processing
from .dataset import HealthExamDataset
from .sampler import PersonBatchSampler, InfinitePersonBatchSampler
from .collate_fn import collate_exams

# Embedders - import the full factory interface
from .embedders import (
    # Individual embedders
    TextEmbedder,
    CategoricalEmbedder, 
    NumericalEmbedder,
    TabEmbedder,
    
    # Positional encodings
    SinusoidalPositionalEncoding,
    TimeAwarePositionalEmbedding,
    
    # Factory functions (most commonly used)
    EmbedderBundle,
    create_embedders_from_config,
    create_embedders_for_training,
    validate_embedder_dimensions
)

# Transformer architectures
from .transformers import (
    TinyTextTransformer,
    BiCrossAttLayer,
    UniTransformerLayer, 
    IndCausalTransformer
)

# Fusion components
from .fusion import (
    TextCompressor,
    ImportanceWeightedConcat
)

# Main SSL Model
from .medical_ssl_model import (
    MedicalSSLModel,
    ModelOutputs
)

# Model utilities
from .utils import prepare_individual_sequences

# Define what gets imported with "from src.models import *"
__all__ = [
    # Data loading pipeline
    'HealthExamDataset',
    'PersonBatchSampler', 
    'InfinitePersonBatchSampler',
    'collate_exams',
    
    # Main SSL model
    'MedicalSSLModel',
    'ModelOutputs',
    
    # Embedders - factory interface (most common)
    'create_embedders_from_config',
    'create_embedders_for_training',
    'EmbedderBundle',
    'TabEmbedder',
    
    # Individual embedders
    'TextEmbedder',
    'CategoricalEmbedder',
    'NumericalEmbedder',
    
    # Positional encodings
    'SinusoidalPositionalEncoding',
    'TimeAwarePositionalEmbedding',
    
    # Transformer architectures  
    'TinyTextTransformer',
    'BiCrossAttLayer',
    'UniTransformerLayer',
    'IndCausalTransformer',
    
    # Fusion components
    'TextCompressor',
    'ImportanceWeightedConcat',
    
    # Utilities
    'prepare_individual_sequences',
    'validate_embedder_dimensions'
]
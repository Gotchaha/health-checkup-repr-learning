# tests/models/embedders/test_init.py

import torch
import tempfile
import pytest
from pathlib import Path

# Test imports work correctly
from src.models.embedders import (
    TextEmbedder, CategoricalEmbedder, NumericalEmbedder, TabEmbedder,
    EmbedderBundle, create_embedders_from_config, create_embedders_for_training,
    validate_embedder_dimensions
)


def test_imports():
    """Test that all imports work correctly."""
    # If we get here without import errors, the test passes
    assert TextEmbedder is not None
    assert CategoricalEmbedder is not None
    assert NumericalEmbedder is not None
    assert TabEmbedder is not None
    assert EmbedderBundle is not None
    assert create_embedders_from_config is not None


def test_create_embedders_from_config():
    """Test factory function creates embedders with minimal config."""
    # Minimal working config
    config = {
        'text': {
            'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
            'max_length': 128,  # Smaller for testing
            'add_phi_tokens': False,  # Skip PHI for simpler test
            'device': 'cpu'
        },
        'categorical': {
            'vocab_path': 'config/embedders/cat_vocab.yaml',
            'embedding_dim': 768,
            'device': 'cpu'
        },
        'numerical': {
            'd_embedding': 768,
            'n_bands': 16,  # Smaller for testing
            'device': 'cpu'
        }
    }
    
    # Create embedders
    embedders = create_embedders_from_config(config, device='cpu')
    
    # Basic checks
    assert isinstance(embedders, EmbedderBundle)
    assert embedders.text is not None
    assert embedders.categorical is not None
    assert embedders.numerical is not None
    
    # Check vocab sizes
    vocab_sizes = embedders.get_vocab_sizes()
    assert vocab_sizes['text'] > 0
    assert vocab_sizes['categorical'] > 0
    assert vocab_sizes['numerical'] is None  # Numerical doesn't have vocab
    
    # Check embedding dimensions
    dims = embedders.get_embedding_dims()
    assert dims['text'] == 768
    assert dims['categorical'] == 768
    assert dims['numerical'] == 768


def test_embedder_bundle_device_management():
    """Test EmbedderBundle device management works."""
    # Create with minimal config
    config = {
        'text': {
            'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
            'max_length': 128,
            'add_phi_tokens': False
        },
        'categorical': {
            'vocab_path': 'config/embedders/cat_vocab.yaml',
            'embedding_dim': 768
        },
        'numerical': {
            'd_embedding': 768,
            'n_bands': 16
        }
    }
    
    embedders = create_embedders_from_config(config, device='cpu')
    
    # Test device movement (CPU only for CI compatibility)
    embedders.to('cpu')
    
    # Test training/eval mode switching
    embedders.train()
    assert embedders.text.embedding_layer.training
    assert embedders.categorical.training
    assert embedders.numerical.training
    
    embedders.eval()
    assert not embedders.text.embedding_layer.training
    assert not embedders.categorical.training
    assert not embedders.numerical.training


def test_dimension_validation():
    """Test dimension validation catches mismatches."""
    # Create embedders with mismatched dimensions
    config = {
        'text': {
            'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
            'max_length': 128,
            'add_phi_tokens': False
        },
        'categorical': {
            'vocab_path': 'config/embedders/cat_vocab.yaml',
            'embedding_dim': 512  # Different dimension
        },
        'numerical': {
            'd_embedding': 768
        }
    }
    
    embedders = create_embedders_from_config(config, device='cpu')
    
    # Should raise ValueError due to dimension mismatch
    with pytest.raises(ValueError, match="Embedder dimensions must match"):
        validate_embedder_dimensions(embedders)


def test_create_embedders_for_training():
    """Test convenience function for training pipeline."""
    # Full config structure like training would use
    full_config = {
        'experiment_name': 'test_experiment',
        'model': {
            'embedders': {
                'text': {
                    'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
                    'max_length': 128,
                    'add_phi_tokens': False
                },
                'categorical': {
                    'vocab_path': 'config/embedders/cat_vocab.yaml',
                    'embedding_dim': 768
                },
                'numerical': {
                    'd_embedding': 768,
                    'n_bands': 16
                }
            }
        },
        'training': {
            'batch_size': 32
        }
    }
    
    # Create embedders for training
    embedders = create_embedders_for_training(full_config, device='cpu')
    
    # Should be in training mode by default
    assert embedders.text.embedding_layer.training
    assert embedders.categorical.training
    assert embedders.numerical.training
    
    # Should have validated dimensions (no exception)
    assert embedders is not None


def test_save_load_bundle():
    """Test EmbedderBundle save/load functionality."""
    # Create embedders
    config = {
        'text': {
            'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
            'max_length': 128,
            'add_phi_tokens': False
        },
        'categorical': {
            'vocab_path': 'config/embedders/cat_vocab.yaml',
            'embedding_dim': 768
        },
        'numerical': {
            'd_embedding': 768,
            'n_bands': 16
        }
    }
    
    embedders = create_embedders_from_config(config, device='cpu')
    
    # Save to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        save_paths = embedders.save_all(temp_dir)
        
        # Check that files were created
        assert 'text' in save_paths
        assert 'categorical' in save_paths
        assert 'numerical' in save_paths
        
        # Check bundle metadata was created
        meta_path = Path(temp_dir) / "bundle_metadata.yaml"
        assert meta_path.exists()
        
        # Load embedders back
        loaded_embedders = EmbedderBundle.load_all(temp_dir)
        
        # Basic checks on loaded embedders
        assert isinstance(loaded_embedders, EmbedderBundle)
        assert loaded_embedders.text is not None
        assert loaded_embedders.categorical is not None
        assert loaded_embedders.numerical is not None
        
        # Check dimensions match
        original_dims = embedders.get_embedding_dims()
        loaded_dims = loaded_embedders.get_embedding_dims()
        assert original_dims == loaded_dims


def test_bundle_repr():
    """Test EmbedderBundle string representation."""
    config = {
        'text': {
            'pretrained_model_name': 'alabnii/jmedroberta-base-sentencepiece',
            'max_length': 128,
            'add_phi_tokens': False
        },
        'categorical': {
            'vocab_path': 'config/embedders/cat_vocab.yaml',
            'embedding_dim': 768
        },
        'numerical': {
            'd_embedding': 768,
            'n_bands': 16
        }
    }
    
    embedders = create_embedders_from_config(config, device='cpu')
    
    # Should generate meaningful string representation
    repr_str = repr(embedders)
    assert 'EmbedderBundle' in repr_str
    assert 'text:' in repr_str
    assert 'categorical:' in repr_str
    assert 'numerical:' in repr_str
    assert 'vocab=' in repr_str
    assert 'dim=' in repr_str


if __name__ == "__main__":
    # Run basic smoke test
    print("Running embedders integration smoke test...")
    
    test_imports()
    print("✓ Imports work")
    
    test_create_embedders_from_config()
    print("✓ Factory function works")
    
    test_embedder_bundle_device_management()
    print("✓ Device management works")
    
    test_create_embedders_for_training()
    print("✓ Training convenience function works")
    
    test_bundle_repr()
    print("✓ String representation works")
    
    print("All smoke tests passed!")
# tests/models/embedders/test_NumericalEmbedder.py

import os
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import yaml

from src.models.embedders.NumericalEmbedder import NumericalEmbedder

class TestNumericalEmbedder:
    """Test suite for NumericalEmbedder using Random Fourier Features."""
    
    @pytest.fixture
    def test_dir(self):
        """Create and clean up a temporary directory for test artifacts."""
        tmp_dir = tempfile.mkdtemp()
        yield tmp_dir
        shutil.rmtree(tmp_dir)
    
    @pytest.fixture
    def basic_embedder(self):
        """Create a basic embedder for testing."""
        return NumericalEmbedder(
            d_embedding=32,
            n_bands=16,
            sigma=1.0,
            seed=42  # For reproducible testing
        )
    
    def test_init_and_forward(self, basic_embedder):
        """Test basic initialization and forward pass."""
        # Create sample input with 10 features
        batch_size = 5
        n_features = 10
        x = torch.randn(batch_size, n_features)
        
        # Get embeddings
        embeddings = basic_embedder(x)
        
        # Check output shape
        assert embeddings.shape == (batch_size, n_features, 32)
        
        # Check output is not NaN
        assert not torch.isnan(embeddings).any()
        
        # Check output is not infinite
        assert not torch.isinf(embeddings).any()
    
    def test_output_shape(self, basic_embedder):
        """Test get_output_shape method."""
        shape = basic_embedder.get_output_shape()
        assert shape == torch.Size([32])  # Per feature embedding dimension
        assert basic_embedder.get_embedding_dimension() == 32
    
    def test_different_feature_counts(self, basic_embedder):
        """Test with different numbers of features."""
        # Test with various feature counts
        for n_features in [1, 5, 10, 20]:
            x = torch.randn(3, n_features)
            embeddings = basic_embedder(x)
            assert embeddings.shape == (3, n_features, 32)
            assert not torch.isnan(embeddings).any()
    
    def test_different_batch_shapes(self, basic_embedder):
        """Test with different batch dimensions."""
        # Single example
        x1 = torch.randn(1, 10)
        out1 = basic_embedder(x1)
        assert out1.shape == (1, 10, 32)
        
        # 3D tensor (e.g., batch + sequence)
        x2 = torch.randn(3, 4, 10)
        out2 = basic_embedder(x2)
        assert out2.shape == (3, 4, 10, 32)
        
        # 4D tensor (e.g., batch + height + width)
        x3 = torch.randn(2, 3, 4, 10)
        out3 = basic_embedder(x3)
        assert out3.shape == (2, 3, 4, 10, 32)
    
    def test_handle_nan_values(self, basic_embedder):
        """Test handling of NaN and infinite values."""
        # Create input with NaNs and infinities
        x = torch.tensor([
            [1.0, 2.0, float('nan'), 4.0, 5.0],
            [11.0, float('inf'), 13.0, float('-inf'), 15.0]
        ])
        
        # Forward pass should not raise errors
        embeddings = basic_embedder(x)
        
        # Verify output shape
        assert embeddings.shape == (2, 5, 32)
        
        # Verify no NaNs in output
        assert not torch.isnan(embeddings).any()
        
        # Verify no infinities in output
        assert not torch.isinf(embeddings).any()
    
    def test_zero_statistics_normalization(self, basic_embedder):
        """Test the zero-statistics normalization x' = x / (1 + |x|)."""
        # Test with extreme values
        x = torch.tensor([[1000.0, -1000.0, 0.0, 0.5, -0.5]])
        
        # Forward pass should handle extreme values
        embeddings = basic_embedder(x)
        
        # Check output shape
        assert embeddings.shape == (1, 5, 32)
        
        # Check output is finite
        assert torch.isfinite(embeddings).all()
    
    def test_trainability(self, basic_embedder):
        """Test trainability control."""
        # Embedder should be trainable by default
        assert basic_embedder.is_trainable()
        assert basic_embedder.proj.weight.requires_grad
        
        # Freeze embedder
        basic_embedder.set_trainable(False)
        assert not basic_embedder.is_trainable()
        assert not basic_embedder.proj.weight.requires_grad
        
        # Unfreeze embedder
        basic_embedder.set_trainable(True)
        assert basic_embedder.is_trainable()
        assert basic_embedder.proj.weight.requires_grad
    
    def test_device_handling(self, basic_embedder):
        """Test device handling."""
        # Skip test if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Move to GPU
        basic_embedder.to("cuda")
        
        # Check device
        assert next(basic_embedder.parameters()).device.type == "cuda"
        
        # Test forward pass on GPU
        x = torch.randn(5, 10, device="cuda")
        embeddings = basic_embedder(x)
        assert embeddings.device.type == "cuda"
        
        # Test with mismatched device
        x_cpu = torch.randn(5, 10)  # on CPU
        embeddings_cpu = basic_embedder(x_cpu)  # should handle device mismatch
        assert embeddings_cpu.device.type == "cuda"
        
        # Move back to CPU
        basic_embedder.to("cpu")
        assert next(basic_embedder.parameters()).device.type == "cpu"
    
    def test_process_method(self, basic_embedder):
        """Test process convenience method."""
        x = torch.randn(5, 10)
        embeddings = basic_embedder.process(x)
        assert embeddings.shape == (5, 10, 32)
    
    def test_save_and_load(self, basic_embedder, test_dir):
        """Test save_pretrained and from_pretrained methods."""
        save_path = os.path.join(test_dir, "embedder")
        
        # Save embedder
        basic_embedder.save_pretrained(save_path)
        
        # Verify files exist
        assert os.path.exists(os.path.join(save_path, "numerical_embedder.pt"))
        assert os.path.exists(os.path.join(save_path, "numerical_embedder_metadata.yaml"))
        
        # Load embedder
        loaded_embedder = NumericalEmbedder.from_pretrained(save_path)
        
        # Check loaded embedder
        assert loaded_embedder.d_embedding == basic_embedder.d_embedding
        assert loaded_embedder.n_bands == basic_embedder.n_bands
        assert loaded_embedder.sigma == basic_embedder.sigma
        
        # Verify embeddings are the same
        x = torch.randn(5, 10)
        original_embeddings = basic_embedder(x)
        loaded_embeddings = loaded_embedder(x)
        assert torch.allclose(original_embeddings, loaded_embeddings)
    
    def test_config_management(self, basic_embedder, test_dir):
        """Test config management."""
        # Get config
        config = basic_embedder.get_config()
        assert config["d_embedding"] == 32
        assert config["n_bands"] == 16
        assert config["sigma"] == 1.0
        assert config["seed"] == 42
        
        # Save config to file
        config_path = os.path.join(test_dir, "embedder_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Load from config
        loaded_embedder = NumericalEmbedder.from_config(config_path)
        
        # Check loaded embedder
        assert loaded_embedder.d_embedding == basic_embedder.d_embedding
        assert loaded_embedder.n_bands == basic_embedder.n_bands
        assert loaded_embedder.sigma == basic_embedder.sigma
        
        # Test config overrides
        loaded_embedder = NumericalEmbedder.from_config(
            config_path, 
            d_embedding=64
        )
        assert loaded_embedder.d_embedding == 64
    
    def test_reproducibility_with_seed(self):
        """Test that the same seed produces identical random frequencies."""
        seed = 123
        
        # Create two embedders with the same seed
        embedder1 = NumericalEmbedder(d_embedding=32, n_bands=16, seed=seed)
        embedder2 = NumericalEmbedder(d_embedding=32, n_bands=16, seed=seed)
        
        # Random frequencies should be identical
        assert torch.allclose(embedder1.random_freq, embedder2.random_freq)
        
        # But final outputs will still differ due to different projection weights
        x = torch.randn(5, 10)
        out1 = embedder1(x)
        out2 = embedder2(x)
        
        # Outputs should be different despite same frequencies (due to different projection weights)
        assert not torch.allclose(out1, out2)
        
        # Create embedder with different seed
        embedder3 = NumericalEmbedder(d_embedding=32, n_bands=16, seed=seed + 1)
        
        # Random frequencies should be different
        assert not torch.allclose(embedder1.random_freq, embedder3.random_freq)
    
    def test_no_seed_randomness(self):
        """Test that without seed, different instances produce different results."""
        # Create two embedders without seed
        embedder1 = NumericalEmbedder(d_embedding=32, n_bands=16)
        embedder2 = NumericalEmbedder(d_embedding=32, n_bands=16)
        
        # Test with same input
        x = torch.randn(5, 10)
        out1 = embedder1(x)
        out2 = embedder2(x)
        
        # Should produce different results (with high probability)
        assert not torch.allclose(out1, out2)
    
    def test_bias_parameter(self):
        """Test with and without bias in projection layer."""
        # With bias (default is False)
        embedder_with_bias = NumericalEmbedder(d_embedding=32, n_bands=16, bias=True)
        assert embedder_with_bias.proj.bias is not None
        
        # Without bias (default)
        embedder_no_bias = NumericalEmbedder(d_embedding=32, n_bands=16, bias=False)
        assert embedder_no_bias.proj.bias is None
        
        # Test forward passes work for both
        x = torch.randn(5, 10)
        out_with_bias = embedder_with_bias(x)
        out_no_bias = embedder_no_bias(x)
        
        assert out_with_bias.shape == (5, 10, 32)
        assert out_no_bias.shape == (5, 10, 32)
    
    def test_different_sigma_values(self):
        """Test with different sigma values for frequency sampling."""
        x = torch.randn(5, 10)
        
        # Test different sigma values
        for sigma in [0.1, 1.0, 10.0]:
            embedder = NumericalEmbedder(d_embedding=32, n_bands=16, sigma=sigma, seed=42)
            embeddings = embedder(x)
            assert embeddings.shape == (5, 10, 32)
            assert torch.isfinite(embeddings).all()
    
    def test_different_n_bands(self):
        """Test with different numbers of frequency bands."""
        x = torch.randn(5, 10)
        
        # Test different n_bands values
        for n_bands in [4, 8, 16, 32]:
            embedder = NumericalEmbedder(d_embedding=32, n_bands=n_bands, seed=42)
            embeddings = embedder(x)
            assert embeddings.shape == (5, 10, 32)
            assert torch.isfinite(embeddings).all()
            
            # Check that the projection layer has correct input size
            assert embedder.proj.in_features == 2 * n_bands
    
    def test_empty_input(self, basic_embedder):
        """Test handling of empty input."""
        # Empty batch
        x = torch.empty(0, 10)
        embeddings = basic_embedder(x)
        assert embeddings.shape == (0, 10, 32)
        
        # Empty features
        x = torch.empty(5, 0)
        embeddings = basic_embedder(x)
        assert embeddings.shape == (5, 0, 32)
    
    def test_single_value_input(self, basic_embedder):
        """Test with single value inputs."""
        # Single feature, single batch
        x = torch.tensor([[5.0]])
        embeddings = basic_embedder(x)
        assert embeddings.shape == (1, 1, 32)
        assert torch.isfinite(embeddings).all()
    
    def test_repr_method(self, basic_embedder):
        """Test string representation."""
        repr_str = repr(basic_embedder)
        assert "NumericalEmbedder" in repr_str
        assert "d_embedding=32" in repr_str
        assert "n_bands=16" in repr_str
        assert "sigma=1.000" in repr_str
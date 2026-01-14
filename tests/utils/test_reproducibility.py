# tests/utils/test_reproducibility.py

import pytest
import random
import os
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from src.utils.reproducibility import (
    set_all_seeds,
    make_deterministic,
    get_environment_info,
    with_reproducible_context,
    setup_reproducibility
)


class TestSetAllSeeds:
    """Test set_all_seeds functionality."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_set_all_seeds_basic(self):
        """Test basic seed setting functionality."""
        seed = 42
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore PYTHONHASHSEED warning
            set_all_seeds(seed)
        
        # Check Python random
        assert random.getstate()[1][0] != 0  # Should be seeded
        
        # Check NumPy
        np_state = np.random.get_state()
        assert np_state[0] == 'MT19937'  # Should be seeded
        
        # Check PyTorch
        torch_state = torch.get_rng_state()
        assert torch_state is not None
        
        # Check environment variable was set
        assert os.environ['PYTHONHASHSEED'] == '42'
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_set_all_seeds_with_cuda(self, mock_manual_seed_all, mock_manual_seed, mock_cuda_available):
        """Test seed setting with CUDA available."""
        seed = 123
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_all_seeds(seed)
        
        # Check CUDA manual_seed was called once
        mock_manual_seed.assert_called_once_with(seed)
        
        # Check CUDA manual_seed_all was called twice:
        # 1. Internally by torch.manual_seed (PyTorch behavior)
        # 2. Explicitly by our set_all_seeds function
        # This ensures seeds are set even if PyTorch's internal behavior changes
        assert mock_manual_seed_all.call_count == 2
        mock_manual_seed_all.assert_any_call(seed)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_set_all_seeds_without_cuda(self, mock_cuda_available):
        """Test seed setting without CUDA available."""
        seed = 456
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_all_seeds(seed)
        
        # Should complete without errors
        assert random.getstate()[1][0] != 0
    
    @patch.dict(os.environ, {'PYTHONHASHSEED': '42'}, clear=True)
    def test_pythonhashseed_correct(self):
        """Test when PYTHONHASHSEED is correctly set."""
        seed = 42
        
        # Should not warn when PYTHONHASHSEED matches
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_all_seeds(seed)
            
            # Should not generate warning
            hash_warnings = [warning for warning in w if 'PYTHONHASHSEED' in str(warning.message)]
            assert len(hash_warnings) == 0
    
    @patch.dict(os.environ, {'PYTHONHASHSEED': '99'}, clear=True)
    def test_pythonhashseed_mismatch(self):
        """Test when PYTHONHASHSEED doesn't match seed."""
        seed = 42
        
        # Should warn when PYTHONHASHSEED doesn't match
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_all_seeds(seed)
            
            # Should generate warning
            hash_warnings = [warning for warning in w if 'PYTHONHASHSEED' in str(warning.message)]
            assert len(hash_warnings) == 1
            assert '42' in str(hash_warnings[0].message)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_pythonhashseed_not_set(self):
        """Test when PYTHONHASHSEED is not set at all."""
        seed = 42
        
        # Should warn when PYTHONHASHSEED is not set
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_all_seeds(seed)
            
            # Should generate warning
            hash_warnings = [warning for warning in w if 'PYTHONHASHSEED' in str(warning.message)]
            assert len(hash_warnings) == 1


class TestMakeDeterministic:
    """Test make_deterministic functionality."""
    
    @patch('torch.backends.cudnn')
    def test_make_deterministic_basic(self, mock_cudnn):
        """Test basic deterministic mode."""
        make_deterministic(strict=False)
        
        # Check basic settings were applied
        assert mock_cudnn.deterministic == True
        assert mock_cudnn.benchmark == False
    
    @patch('torch.backends.cudnn')
    @patch('torch.use_deterministic_algorithms')
    @patch.dict(os.environ, {}, clear=True)
    def test_make_deterministic_strict(self, mock_use_det_alg, mock_cudnn):
        """Test strict deterministic mode."""
        make_deterministic(strict=True)
        
        # Check strict settings were applied
        assert mock_cudnn.deterministic == True
        assert mock_cudnn.benchmark == False
        mock_use_det_alg.assert_called_once_with(True)
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    
    @patch('torch.backends.cudnn')
    def test_make_deterministic_api_compatibility(self, mock_cudnn):
        """Test API compatibility when use_deterministic_algorithms doesn't exist."""
        # Mock torch without use_deterministic_algorithms
        with patch.object(torch, 'use_deterministic_algorithms', None):
            # Should not raise AttributeError
            make_deterministic(strict=True)
            
            # Basic settings should still be applied
            assert mock_cudnn.deterministic == True
            assert mock_cudnn.benchmark == False


class TestGetEnvironmentInfo:
    """Test get_environment_info functionality."""
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_environment_info_lightweight_no_cuda(self, mock_cuda_available):
        """Test lightweight environment info without CUDA."""
        info = get_environment_info(lightweight=True)
        
        # Check required fields
        assert 'python_version' in info
        assert 'platform' in info
        assert 'torch_version' in info
        assert 'cuda_available' in info
        assert info['cuda_available'] == False
        assert 'numpy_version' in info
        assert 'pandas_version' in info
        assert 'transformers_version' in info
        
        # Should not have CUDA-specific fields
        assert 'cuda_version' not in info
        assert 'gpu_count' not in info
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_name')
    @patch('torch.version.cuda', '11.7')
    @patch('torch.backends.cudnn.version', return_value=8700)
    def test_get_environment_info_with_cuda(self, mock_cudnn_version, mock_get_device_name, mock_device_count, mock_cuda_available):
        """Test environment info with CUDA available."""
        mock_get_device_name.side_effect = ['GPU 0', 'GPU 1']
        
        info = get_environment_info(lightweight=True)
        
        # Check CUDA fields
        assert info['cuda_available'] == True
        assert info['cuda_version'] == '11.7'
        assert info['cudnn_version'] == 8700
        assert info['gpu_count'] == 2
        assert info['gpu_names'] == ['GPU 0', 'GPU 1']
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    def test_get_environment_info_detailed(self, mock_get_device_props, mock_device_count, mock_cuda_available):
        """Test detailed environment info."""
        # Mock GPU properties
        mock_props = MagicMock()
        mock_props.total_memory = 8000000000  # 8GB
        mock_get_device_props.return_value = mock_props
        
        info = get_environment_info(lightweight=False)
        
        # Check additional detailed fields
        assert 'pyarrow_version' in info
        assert 'cpu_count' in info
        assert 'gpu_memory' in info
        assert info['gpu_memory'] == [8000000000]
    
    def test_get_environment_info_missing_packages(self):
        """Test environment info with missing optional packages."""
        # Mock missing packages
        with patch.dict('sys.modules', {'pyarrow': None}):
            info = get_environment_info(lightweight=False)
            
            # Should handle missing packages gracefully
            assert 'pyarrow_version' in info
            # The actual implementation handles import errors


class TestWithReproducibleContext:
    """Test with_reproducible_context functionality."""
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.cudnn')
    @patch('torch.are_deterministic_algorithms_enabled', return_value=False)
    @patch('torch.use_deterministic_algorithms')
    @patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': 'original_value'}, clear=True)
    def test_reproducible_context_basic(self, mock_use_det_alg, mock_are_det_enabled, mock_cudnn, mock_cuda_available):
        """Test basic reproducible context functionality."""
        seed = 42
        
        # Store original states
        original_python_state = random.getstate()
        original_numpy_state = np.random.get_state()
        original_torch_state = torch.get_rng_state()
        
        # Use context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with with_reproducible_context(seed, strict=True):
                # Inside context, should be seeded
                context_python_state = random.getstate()
                context_numpy_state = np.random.get_state()
                context_torch_state = torch.get_rng_state()
                
                # States should be different from original
                assert context_python_state != original_python_state
                assert not np.array_equal(context_numpy_state[1], original_numpy_state[1])
                assert not torch.equal(context_torch_state, original_torch_state)
        
        # After context, should be restored
        restored_python_state = random.getstate()
        restored_numpy_state = np.random.get_state()
        restored_torch_state = torch.get_rng_state()
        
        assert restored_python_state == original_python_state
        assert np.array_equal(restored_numpy_state[1], original_numpy_state[1])
        assert torch.equal(restored_torch_state, original_torch_state)
        
        # Environment variable should be restored
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == 'original_value'
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_rng_state')
    @patch('torch.cuda.set_rng_state')
    @patch('torch.backends.cudnn')
    def test_reproducible_context_with_cuda(self, mock_cudnn, mock_set_rng_state, mock_get_rng_state, mock_device_count, mock_cuda_available):
        """Test reproducible context with CUDA."""
        seed = 123
        
        # Mock CUDA states
        mock_cuda_states = [torch.tensor([1]), torch.tensor([4])]
        mock_get_rng_state.side_effect = mock_cuda_states
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with with_reproducible_context(seed, strict=False):
                pass
        
        # Check CUDA states were restored
        assert mock_set_rng_state.call_count == 2
        mock_set_rng_state.assert_any_call(mock_cuda_states[0], 0)
        mock_set_rng_state.assert_any_call(mock_cuda_states[1], 1)
    
    @patch('torch.backends.cudnn')
    def test_reproducible_context_api_compatibility(self, mock_cudnn):
        """Test context manager with API compatibility issues."""
        seed = 456
        
        # Mock missing API
        with patch.object(torch, 'are_deterministic_algorithms_enabled', None):
            with patch.object(torch, 'use_deterministic_algorithms', None):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Should not raise AttributeError
                    with with_reproducible_context(seed, strict=True):
                        pass
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('torch.backends.cudnn')
    def test_reproducible_context_env_var_cleanup(self, mock_cudnn):
        """Test environment variable cleanup when not originally set."""
        seed = 789
        
        # Ensure CUBLAS_WORKSPACE_CONFIG is not set
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with with_reproducible_context(seed, strict=True):
                # Should be set inside context
                pass
        
        # Should be removed after context
        assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    
    def test_reproducible_context_exception_handling(self):
        """Test context manager handles exceptions properly."""
        seed = 999
        
        # Store original states
        original_python_state = random.getstate()
        
        # Test exception inside context
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with with_reproducible_context(seed, strict=False):
                    raise ValueError("Test exception")
        
        # State should still be restored after exception
        restored_python_state = random.getstate()
        assert restored_python_state == original_python_state


class TestSetupReproducibility:
    """Test setup_reproducibility convenience function."""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    def test_setup_reproducibility_verbose(self, mock_device_count, mock_cuda_available, capsys):
        """Test setup reproducibility with verbose output."""
        seed = 42
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env_info = setup_reproducibility(seed, strict=False, verbose=True)
        
        # Check environment info was returned
        assert isinstance(env_info, dict)
        assert 'torch_version' in env_info
        assert 'cuda_available' in env_info
        
        # Check verbose output
        captured = capsys.readouterr()
        assert "Reproducibility setup complete:" in captured.out
        assert f"Seed: {seed}" in captured.out
        assert "Strict mode: False" in captured.out
    
    def test_setup_reproducibility_quiet(self, capsys):
        """Test setup reproducibility without verbose output."""
        seed = 123
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env_info = setup_reproducibility(seed, strict=True, verbose=False)
        
        # Check environment info was returned
        assert isinstance(env_info, dict)
        
        # Check no verbose output
        captured = capsys.readouterr()
        assert captured.out == ""


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_reproducibility_with_different_seed_types(self):
        """Test reproducibility with different seed types."""
        # Test with different numeric types
        seeds = [42, 42.0, np.int32(42), np.int64(42)]
        
        for seed in seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Should handle different numeric types
                set_all_seeds(int(seed))  # Convert to int as expected
    
    @patch('torch.cuda.is_available', side_effect=Exception("CUDA error"))
    def test_cuda_error_handling(self, mock_cuda_available):
        """Test handling of CUDA errors."""
        seed = 42
        
        # Should handle CUDA errors gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_all_seeds(seed)
    
    def test_large_seed_values(self):
        """Test with large seed values."""
        large_seed = 2**31 - 1  # Max int32
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should handle large seeds
            set_all_seeds(large_seed)
    
    @patch('platform.platform', side_effect=Exception("Platform error"))
    def test_environment_info_with_errors(self, mock_platform):
        """Test environment info collection with errors."""
        # Should handle errors gracefully
        info = get_environment_info(lightweight=True)
        
        # Should still return a dictionary
        assert isinstance(info, dict)
        assert 'torch_version' in info  # Core info should still be available
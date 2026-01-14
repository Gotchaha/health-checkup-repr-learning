import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
from typing import Dict, Any

# Add the source directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.downstream.lab_test.training.utils import WandbLogger, TASK_NAMES, PANELS


class TestWandbLogger:
    """Test suite for the enhanced WandbLogger with medical formatting."""
    
    def test_format_medical_metrics_basic_formatting(self):
        """Test basic medical metric formatting with known task IDs."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        input_metrics = {
            '000000800008_loss': 114.24,  # LDL (Lipid Panel)
            '000000800005_loss': 52.82,   # HDL (Lipid Panel)
            '000000800026_loss': 3.73,    # AST (Liver Panel)
            'val_macro_auroc': 0.6174,    # Non-medical metric
            'epoch': 0                    # Non-medical metric
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        expected = {
            'lipid_panel/LDL_loss': 114.24,
            'lipid_panel/HDL_loss': 52.82,
            'liver_panel/AST_loss': 3.73,
            'val_macro_auroc': 0.6174,    # Pass-through unchanged
            'epoch': 0                    # Pass-through unchanged
        }
        
        assert formatted == expected
    
    def test_format_medical_metrics_with_prefixes_and_suffixes(self):
        """Test medical formatting with complex key structures."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        input_metrics = {
            'epoch/val_000000800008_loss': 114.24,                    # Prefix + suffix
            'train/000000800008_uncertainty_weight': 0.999,           # Prefix + suffix
            'val_000000800026_mae': 2.45,                            # Prefix + suffix
            '000000800005': 52.82,                                   # No suffix
            'test_000000800006_flag_auroc': 0.8                      # Complex suffix
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        expected = {
            'epoch/val_lipid_panel/LDL_loss': 114.24,
            'train/lipid_panel/LDL_uncertainty_weight': 0.999,
            'val_liver_panel/AST_mae': 2.45,
            'lipid_panel/HDL': 52.82,
            'test_lipid_panel/TG_flag_auroc': 0.8
        }
        
        assert formatted == expected
    
    def test_format_medical_metrics_bp_obesity_panel_slash_fix(self):
        """Test that BP/Obesity Panel slash is converted to underscore."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        input_metrics = {
            '000000200006_loss': 22.02,  # BMI (in "BP/Obesity Panel")
            '000000500009_loss': 118.71, # SBP (in "BP/Obesity Panel")
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        expected = {
            'bp_obesity_panel/BMI_loss': 22.02,     # Slash replaced with underscore
            'bp_obesity_panel/SBP_loss': 118.71     # Slash replaced with underscore
        }
        
        assert formatted == expected
    
    def test_format_medical_metrics_risk_flags_panel(self):
        """Test Risk Flags panel formatting."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        input_metrics = {
            '000002800010_auroc': 0.733,  # Med1 (Risk Flags)
            '000003100001_loss': 2.73,    # Smoking (Risk Flags)
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        expected = {
            'risk_flags/Med1_auroc': 0.733,
            'risk_flags/Smoking_loss': 2.73
        }
        
        assert formatted == expected
    
    def test_format_medical_metrics_unknown_task_passthrough(self):
        """Test that unknown task IDs pass through unchanged."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        input_metrics = {
            'unknown_task_999_loss': 5.0,     # Unknown task ID
            '000000800008_loss': 114.24,      # Known task ID
            'completely_unrelated': 1.0       # No task ID at all
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        expected = {
            'unknown_task_999_loss': 5.0,     # Pass-through unchanged
            'lipid_panel/LDL_loss': 114.24,   # Formatted
            'completely_unrelated': 1.0       # Pass-through unchanged
        }
        
        assert formatted == expected
    
    def test_format_medical_metrics_non_string_keys(self):
        """Test handling of non-string keys."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        input_metrics = {
            123: 'numeric_key',
            ('tuple', 'key'): 'tuple_key',
            None: 'none_key',
            '000000800008_loss': 114.24
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        expected = {
            '123': 'numeric_key',
            "('tuple', 'key')": 'tuple_key',
            'None': 'none_key',
            'lipid_panel/LDL_loss': 114.24
        }
        
        assert formatted == expected
    
    def test_format_medical_metrics_first_occurrence_replacement(self):
        """Test that only first occurrence of task ID is replaced."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        # Edge case: task ID appears multiple times in key
        input_metrics = {
            '000000800008_some_000000800008_loss': 114.24,  # Task ID appears twice
        }
        
        formatted = logger._format_medical_metrics(input_metrics)
        
        # Should only replace first occurrence
        expected = {
            'lipid_panel/LDL_some_000000800008_loss': 114.24
        }
        
        assert formatted == expected
    
    def test_log_method_with_formatting_enabled(self):
        """Test that log method applies formatting and calls wandb.log when enabled."""
        # Mock the wandb module at import time
        mock_wandb_module = MagicMock()
        mock_wandb_instance = MagicMock()
        mock_wandb_module.init.return_value = mock_wandb_instance
        
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   mock_wandb_module if name == 'wandb' else __import__(name, *args, **kwargs)):
            
            config = {
                'experiment_name': 'test',
                'wandb': {
                    'project': 'test-project',
                    'entity': 'test-entity'
                }
            }
            logger = WandbLogger(config, enabled=True)
            
            # Test metrics
            metrics = {
                '000000800008_loss': 114.24,
                'val_macro_auroc': 0.6174,
                'epoch': 1
            }
            
            # Call log method
            logger.log(metrics, step=100)
            
            # Verify wandb.log was called with formatted metrics (on the MODULE, not instance)
            expected_formatted = {
                'lipid_panel/LDL_loss': 114.24,
                'val_macro_auroc': 0.6174,
                'epoch': 1
            }
            
            # FIXED: Check mock_wandb_module.log, not mock_wandb_instance.log
            mock_wandb_module.log.assert_called_once_with(expected_formatted, step=100)
    
    def test_log_method_disabled(self):
        """Test that disabled logger doesn't attempt wandb operations."""
        config = {'experiment_name': 'test'}
        logger = WandbLogger(config, enabled=False)
        
        metrics = {'000000800008_loss': 114.24}
        
        # Should not raise any exceptions
        logger.log(metrics, step=100)
        
        # Verify logger is disabled
        assert logger.enabled is False
        assert logger.wandb is None
    
    def test_log_method_error_handling(self):
        """Test that log method handles wandb errors gracefully."""
        # Mock wandb to raise exception on log
        mock_wandb_module = MagicMock()
        mock_wandb_instance = MagicMock()
        mock_wandb_module.log.side_effect = Exception("Mock wandb error")  # FIXED: Set on module, not instance
        mock_wandb_module.init.return_value = mock_wandb_instance
        
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   mock_wandb_module if name == 'wandb' else __import__(name, *args, **kwargs)):
            
            config = {
                'experiment_name': 'test',
                'wandb': {'project': 'test-project'}
            }
            logger = WandbLogger(config, enabled=True)
            
            metrics = {'test_metric': 1.0}
            
            # Should not raise exception, should handle gracefully
            logger.log(metrics, step=100)
            
            # FIXED: Verify log was attempted on the module
            mock_wandb_module.log.assert_called_once()
    
    @patch('builtins.print')  # Mock print to capture warning messages
    def test_init_wandb_import_error(self, mock_print):
        """Test initialization when wandb is not available."""
        # Mock import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == 'wandb':
                raise ImportError("No module named 'wandb'")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            config = {'experiment_name': 'test'}
            logger = WandbLogger(config, enabled=True)
            
            # Should gracefully disable when wandb not available
            assert logger.enabled is False
            assert logger.wandb is None
            
            # Should print warning message
            mock_print.assert_called_with("Warning: wandb not available, disabling wandb logging")
    
    @patch('builtins.print')
    def test_init_wandb_initialization_error(self, mock_print):
        """Test initialization when wandb.init fails."""
        # Mock wandb module but make init fail
        mock_wandb_module = MagicMock()
        mock_wandb_module.init.side_effect = Exception("Mock wandb init error")
        
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   mock_wandb_module if name == 'wandb' else __import__(name, *args, **kwargs)):
            
            config = {
                'experiment_name': 'test',
                'wandb': {'project': 'test-project'}
            }
            logger = WandbLogger(config, enabled=True)
            
            # Should gracefully disable when init fails
            assert logger.enabled is False
            assert logger.wandb is not None  # wandb module imported but init failed
            
            # Should print warning message
            mock_print.assert_called_with("Warning: wandb initialization failed: Mock wandb init error")
    
    def test_is_enabled_method(self):
        """Test is_enabled method returns correct status."""
        # Test disabled logger
        config = {'experiment_name': 'test'}
        logger_disabled = WandbLogger(config, enabled=False)
        assert logger_disabled.is_enabled() is False
        
        # Test enabled logger with successful mock
        mock_wandb_module = MagicMock()
        mock_wandb_instance = MagicMock()
        mock_wandb_module.init.return_value = mock_wandb_instance
        
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                   mock_wandb_module if name == 'wandb' else __import__(name, *args, **kwargs)):
            
            logger_enabled = WandbLogger(config, enabled=True)
            assert logger_enabled.is_enabled() is True
    
    def test_task_names_and_panels_consistency(self):
        """Test that TASK_NAMES and PANELS constants are properly defined."""
        # Verify TASK_NAMES is not empty
        assert len(TASK_NAMES) > 0
        
        # Verify PANELS is not empty  
        assert len(PANELS) > 0
        
        # Verify sample task IDs are present
        assert '000000800008' in TASK_NAMES  # LDL
        assert '000000800005' in TASK_NAMES  # HDL
        assert '000000200006' in TASK_NAMES  # BMI
        
        # Verify sample panels are present
        assert 'Lipid Panel' in PANELS
        assert 'BP/Obesity Panel' in PANELS
        
        # Verify BP/Obesity Panel contains expected tasks
        bp_obesity_tasks = PANELS['BP/Obesity Panel']
        assert '000000200006' in bp_obesity_tasks  # BMI
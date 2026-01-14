# tests/utils/test_wandb_logger.py

# Absolutely prevent any real wandb usage in tests
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_SILENT', 'true')
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from src.utils.wandb_logger import (
    WandbLogger,
    create_wandb_logger,
    get_wandb_url_from_experiment_name
)


class TestWandbLoggerInitialization:
    """Test WandbLogger initialization and configuration."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_wandb_logger_basic_init(self, mock_wandb):
        """Test basic WandbLogger initialization."""
        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/user/project/runs/test_exp"
        mock_run.id = "test_exp"
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'test_exp',
            'model': {'type': 'ssl'},
            'training': {'batch_size': 32}
        }
        
        logger = WandbLogger(config, enabled=True)
        
        # Check initialization
        assert logger.enabled == True
        assert logger.experiment_name == 'test_exp'
        assert logger.run == mock_run
        assert logger.wandb_url == "https://wandb.ai/user/project/runs/test_exp"
        assert logger.wandb_id == "test_exp"
        
        # Check wandb.init was called correctly
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs['project'] == 'medical-ssl-research'
        assert call_kwargs['name'] == 'test_exp'
        assert call_kwargs['id'] == 'test_exp'  # Forced ID = experiment name
        assert call_kwargs['mode'] == 'online'
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_wandb_logger_with_wandb_config(self, mock_wandb):
        """Test WandbLogger initialization with wandb-specific config."""
        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/team/custom_project/runs/custom_exp"
        mock_run.id = "custom_exp"
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'custom_exp',
            'wandb': {
                'project': 'custom_project',
                'entity': 'team',
                'group': 'ssl_experiments',
                'tags': ['multimodal', 'medical'],
                'mode': 'offline',
                'notes': 'Test experiment'
            }
        }
        
        logger = WandbLogger(config, enabled=True)
        
        # Check wandb.init was called with custom config
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs['project'] == 'custom_project'
        assert call_kwargs['entity'] == 'team'
        assert call_kwargs['group'] == 'ssl_experiments'
        assert call_kwargs['tags'] == ['multimodal', 'medical']
        assert call_kwargs['mode'] == 'offline'
        assert call_kwargs['notes'] == 'Test experiment'
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', False)
    def test_wandb_logger_unavailable(self):
        """Test WandbLogger when wandb is not available."""
        config = {'experiment_name': 'test_exp'}
        
        logger = WandbLogger(config, enabled=True)
        
        # Should be disabled
        assert logger.enabled == False
        assert logger.run is None
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_wandb_logger_init_failure(self, mock_wandb):
        """Test WandbLogger when wandb.init fails."""
        mock_wandb.init.side_effect = Exception("wandb init failed")
        
        config = {'experiment_name': 'test_exp'}
        
        with pytest.warns(UserWarning, match="Failed to initialize wandb"):
            logger = WandbLogger(config, enabled=True)
        
        # Should be disabled after failure
        assert logger.enabled == False
        assert logger.run is None
    
    def test_wandb_logger_disabled(self):
        """Test WandbLogger when explicitly disabled."""
        config = {'experiment_name': 'test_exp'}
        
        logger = WandbLogger(config, enabled=False)
        
        assert logger.enabled == False
        assert logger.run is None


class TestConfigFlattening:
    """Test configuration flattening for wandb."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_config_flattening(self, mock_wandb):
        """Test nested config flattening."""
        mock_wandb.init.return_value = MagicMock()
        
        config = {
            'experiment_name': 'flatten_test',
            'model': {
                'embedders': {
                    'text': {'d_model': 768, 'max_length': 512},
                    'categorical': {'vocab_size': 1000}
                },
                'architecture': {'layers': 6}
            },
            'training': {
                'optimizer': {'lr': 0.001, 'weight_decay': 0.01},
                'batch_size': 32
            },
            '_meta': {'should_be_filtered': True},  # Should be filtered out
            'wandb': {'project': 'test'}  # Should be filtered out
        }
        
        logger = WandbLogger(config, enabled=True)
        
        # Get the flattened config that was passed to wandb
        call_kwargs = mock_wandb.init.call_args[1]
        flattened_config = call_kwargs['config']
        
        # Check flattening worked correctly
        assert flattened_config['model.embedders.text.d_model'] == 768
        assert flattened_config['model.embedders.text.max_length'] == 512
        assert flattened_config['model.embedders.categorical.vocab_size'] == 1000
        assert flattened_config['model.architecture.layers'] == 6
        assert flattened_config['training.optimizer.lr'] == 0.001
        assert flattened_config['training.optimizer.weight_decay'] == 0.01
        assert flattened_config['training.batch_size'] == 32
        
        # Check filtered fields
        assert 'experiment_name' in flattened_config  # Should be kept
        assert '_meta.should_be_filtered' not in flattened_config  # Should be filtered
        assert 'wandb.project' not in flattened_config  # Should be filtered
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_config_serialization(self, mock_wandb):
        """Test handling of non-serializable config values."""
        mock_wandb.init.return_value = MagicMock()
        
        # Create non-serializable object
        class NonSerializable:
            def __str__(self):
                return "non_serializable_object"
        
        config = {
            'experiment_name': 'serialization_test',
            'serializable': {'number': 42, 'string': 'test'},
            'non_serializable': {'object': NonSerializable()}
        }
        
        logger = WandbLogger(config, enabled=True)
        
        # Get the config that was passed to wandb
        call_kwargs = mock_wandb.init.call_args[1]
        flattened_config = call_kwargs['config']
        
        # Check serializable values are preserved
        assert flattened_config['serializable.number'] == 42
        assert flattened_config['serializable.string'] == 'test'
        
        # Check non-serializable values are converted to strings
        assert flattened_config['non_serializable.object'] == 'non_serializable_object'


class TestMetricLogging:
    """Test metric logging functionality."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_metrics(self, mock_wandb):
        """Test basic metric logging."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {'experiment_name': 'metric_test'}
        logger = WandbLogger(config, enabled=True)
        
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        logger.log_metrics(metrics, step=10)
        
        mock_wandb.log.assert_called_once_with(metrics, step=10)
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_step_metrics(self, mock_wandb):
        """Test step metrics logging with prefixes."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {'experiment_name': 'step_test'}
        logger = WandbLogger(config, enabled=True)
        
        train_metrics = {'loss': 1.0, 'accuracy': 0.7}
        val_metrics = {'loss': 1.2, 'accuracy': 0.65}
        
        logger.log_step_metrics(5, train_metrics, val_metrics)
        
        # Check metrics were logged with prefixes
        expected_metrics = {
            'train/loss': 1.0,
            'train/accuracy': 0.7,
            'val/loss': 1.2,
            'val/accuracy': 0.65
        }
        mock_wandb.log.assert_called_once_with(expected_metrics, step=5)
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_test_results(self, mock_wandb):
        """Test test results logging."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Table = MagicMock()
        
        config = {
            'experiment_name': 'test_results',
            'wandb': {'log': {'test_details': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        test_results = {
            'accuracy': 0.85,
            'f1_score': 0.82,
            'per_class': {'class_0': {'precision': 0.9}, 'class_1': {'recall': 0.8}},
            'confusion_matrix': [[100, 10], [5, 95]]  # Non-numeric
        }
        
        logger.log_test_results(test_results)
        
        # Check that only top-level numeric metrics were logged
        expected_numeric = {
            'test/accuracy': 0.85,
            'test/f1_score': 0.82
        }
        
        # Should have called log twice: once for metrics, once for table
        assert mock_wandb.log.call_count == 2
        
        # Check first call (numeric metrics - only top-level)
        first_call = mock_wandb.log.call_args_list[0][0][0]
        assert first_call == expected_numeric
    
    def test_log_metrics_disabled(self):
        """Test metric logging when logger is disabled."""
        config = {'experiment_name': 'disabled_test'}
        logger = WandbLogger(config, enabled=False)
        
        # Should not raise any errors
        logger.log_metrics({'loss': 0.5})
        logger.log_step_metrics(1, {'loss': 1.0}, {'loss': 1.2})
        logger.log_test_results({'accuracy': 0.85})

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_metrics_disabled_by_config(self, mock_wandb):
        """Test metric logging when metrics gate is disabled."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'metric_gate_test',
            'wandb': {'log': {'metrics': False}}
        }
        logger = WandbLogger(config, enabled=True)
        
        logger.log_metrics({'loss': 0.5})
        logger.log_step_metrics(1, {'loss': 1.0}, {'loss': 1.2})
        logger.log_test_results({'accuracy': 0.9})
        
        mock_wandb.log.assert_not_called()

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_test_results_details_disabled(self, mock_wandb):
        """Ensure detailed tables obey the test_details gate."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Table = MagicMock()
        
        config = {
            'experiment_name': 'test_details_disabled',
            'wandb': {'log': {'test_details': False}}
        }
        logger = WandbLogger(config, enabled=True)
        
        test_results = {'accuracy': 0.9, 'details': {'cls': 1}}
        logger.log_test_results(test_results)
        
        mock_wandb.Table.assert_not_called()
        # Only metrics should be logged once
        mock_wandb.log.assert_called_once()

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_test_results_metrics_disabled(self, mock_wandb):
        """Ensure numeric metrics obey the metrics gate."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Table = MagicMock()
        
        config = {
            'experiment_name': 'test_metrics_disabled',
            'wandb': {'log': {'metrics': False, 'test_details': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        test_results = {'accuracy': 0.9, 'details': {'cls': 1}}
        logger.log_test_results(test_results)
        
        # Only detailed table should be logged
        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args[0][0]
        assert 'test/accuracy' not in call_args


class TestArtifactLogging:
    """Test artifact logging functionality."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_artifact_new_api(self, mock_wandb, tmp_path):
        """Test artifact logging with new wandb API."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact = MagicMock()
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact
        
        # Mock hasattr to return True (new API available)
        with patch('builtins.hasattr', return_value=True):
            config = {
                'experiment_name': 'artifact_test',
                'wandb': {'log': {'artifacts': True}}
            }
            logger = WandbLogger(config, enabled=True)
            
            # Create test file
            test_file = tmp_path / "test_model.pt"
            test_file.write_text("fake model data")
            
            logger.log_artifact(test_file, "model", "test_artifact", "Test artifact")
            
            # Check artifact was created and logged
            mock_wandb.Artifact.assert_called_once_with(
                name="test_artifact",
                type="model",
                description="Test artifact"
            )
            mock_artifact.add_file.assert_called_once_with(str(test_file))
            mock_wandb.log_artifact.assert_called_once_with(mock_artifact)
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_artifact_old_api(self, mock_wandb, tmp_path):
        """Test artifact logging with old wandb API."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact = MagicMock()
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact
        
        # Mock hasattr to return False (new API not available)
        with patch('builtins.hasattr', return_value=False):
            config = {
                'experiment_name': 'artifact_test_old',
                'wandb': {'log': {'artifacts': True}}
            }
            logger = WandbLogger(config, enabled=True)
            
            # Create test file
            test_file = tmp_path / "test_model.pt"
            test_file.write_text("fake model data")
            
            logger.log_artifact(test_file, "model", "test_artifact", "Test artifact")
            
            # Check artifact was created and logged via run
            mock_wandb.Artifact.assert_called_once()
            mock_artifact.add_file.assert_called_once_with(str(test_file))
            mock_run.log_artifact.assert_called_once_with(mock_artifact)
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_checkpoint(self, mock_wandb, tmp_path):
        """Test checkpoint logging."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact = MagicMock()
        
        config = {
            'experiment_name': 'checkpoint_test',
            'wandb': {'log': {'artifacts': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        # Create test checkpoint
        checkpoint_file = tmp_path / "checkpoint_step_10.pt"
        checkpoint_file.write_text("fake checkpoint data")
        
        logger.log_checkpoint(checkpoint_file, step=10, is_best=True)
        
        # Check artifact was created with correct name
        mock_wandb.Artifact.assert_called_once_with(
            name="checkpoint_step_10_best",
            type="model",
            description="Model checkpoint at step 10 (best validation performance)"
        )
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_config_file(self, mock_wandb, tmp_path):
        """Test config file logging."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact = MagicMock()
        
        config = {
            'experiment_name': 'config_file_test',
            'wandb': {'log': {'artifacts': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        # Create test config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("experiment_name: test")
        
        logger.log_config_file(config_file)
        
        # Check artifact was created with correct name
        mock_wandb.Artifact.assert_called_once_with(
            name="config_file_test_config",
            type="config",
            description="Experiment configuration file"
        )
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_artifact_missing_file(self, mock_wandb, tmp_path):
        """Test artifact logging with missing file."""
        mock_wandb.init.return_value = MagicMock()
        
        config = {
            'experiment_name': 'missing_file_test',
            'wandb': {'log': {'artifacts': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        missing_file = tmp_path / "nonexistent.pt"
        
        with pytest.warns(UserWarning, match="Artifact file not found"):
            logger.log_artifact(missing_file, "model")

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_artifact_disabled_by_config(self, mock_wandb, tmp_path):
        """Ensure artifacts are skipped when gate is disabled."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact = MagicMock()
        
        config = {
            'experiment_name': 'artifact_gate_disabled',
            'wandb': {'log': {'artifacts': False}}
        }
        logger = WandbLogger(config, enabled=True)
        
        test_file = tmp_path / "skip.pt"
        test_file.write_text("data")
        
        logger.log_artifact(test_file, "model")
        
        mock_wandb.Artifact.assert_not_called()


class TestModelWatching:
    """Test model watching functionality."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model(self, mock_wandb):
        """Test model watching."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_test',
            'wandb': {'log': {'watch_model': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        # Mock model
        mock_model = MagicMock()
        
        logger.watch_model(mock_model, log_freq=50, log_graph=False)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, log='gradients', log_freq=50, log_graph=False
        )
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_error(self, mock_wandb):
        """Test model watching with error."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.watch.side_effect = Exception("Watch failed")
        
        config = {
            'experiment_name': 'watch_error_test',
            'wandb': {'log': {'watch_model': True}}
        }
        logger = WandbLogger(config, enabled=True)
        
        mock_model = MagicMock()
        
        with pytest.warns(UserWarning, match="Failed to watch model"):
            logger.watch_model(mock_model)

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_config_log_freq(self, mock_wandb):
        """Test model watching with config-driven log_freq."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_config_test',
            'wandb': {
                'log': {'watch_model': True},
                'watch': {'log_freq': 250}
            }
        }
        logger = WandbLogger(config, enabled=True)
        
        # Mock model
        mock_model = MagicMock()
        
        # Call without explicit log_freq - should use config value
        logger.watch_model(mock_model, log_graph=False)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, log='gradients', log_freq=250, log_graph=False
        )

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_gate_disabled(self, mock_wandb):
        """Watch should be skipped when gate is disabled."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_gate_disabled',
            'wandb': {'log': {'watch_model': False}}
        }
        logger = WandbLogger(config, enabled=True)
        
        logger.watch_model(MagicMock())
        mock_wandb.watch.assert_not_called()

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_global_log_freq_fallback(self, mock_wandb):
        """log_freq should fall back to wandb.log_freq when watch config omits it."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_global_freq',
            'wandb': {
                'log': {'watch_model': True},
                'log_freq': 400
            }
        }
        logger = WandbLogger(config, enabled=True)
        mock_model = MagicMock()
        
        logger.watch_model(mock_model)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, log='gradients', log_freq=400, log_graph=True
        )

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_log_graph_override(self, mock_wandb):
        """watch.log_graph should override function defaults."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_log_graph_override',
            'wandb': {
                'log': {'watch_model': True},
                'watch': {'log_graph': False}
            }
        }
        logger = WandbLogger(config, enabled=True)
        mock_model = MagicMock()
        
        logger.watch_model(mock_model, log_graph=True)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, log='gradients', log_freq=1000, log_graph=False
        )

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_log_mode_parameters(self, mock_wandb):
        """Ensure custom log mode is honored."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_parameters',
            'wandb': {
                'log': {'watch_model': True},
                'watch': {'log': 'parameters', 'log_freq': 30}
            }
        }
        logger = WandbLogger(config, enabled=True)
        mock_model = MagicMock()
        
        logger.watch_model(mock_model)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, log='parameters', log_freq=30, log_graph=True
        )

    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_watch_model_invalid_log_mode(self, mock_wandb):
        """Invalid log modes should fall back to gradients."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {
            'experiment_name': 'watch_invalid_mode',
            'wandb': {
                'log': {'watch_model': True},
                'watch': {'log': 'invalid'}
            }
        }
        logger = WandbLogger(config, enabled=True)
        mock_model = MagicMock()
        
        with pytest.warns(UserWarning, match="Unknown wandb.watch log mode"):
            logger.watch_model(mock_model)
        
        mock_wandb.watch.assert_called_once_with(
            mock_model, log='gradients', log_freq=1000, log_graph=True
        )


class TestContextManager:
    """Test context manager functionality."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_context_manager(self, mock_wandb):
        """Test context manager usage."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {'experiment_name': 'context_test'}
        
        with WandbLogger(config, enabled=True) as logger:
            assert logger.enabled == True
            assert logger.run == mock_run
        
        # Check finish was called
        mock_wandb.finish.assert_called_once()
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_context_manager_with_exception(self, mock_wandb):
        """Test context manager with exception."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        
        config = {'experiment_name': 'context_exception_test'}
        
        with pytest.raises(ValueError):
            with WandbLogger(config, enabled=True) as logger:
                raise ValueError("Test exception")
        
        # Check finish was still called
        mock_wandb.finish.assert_called_once()
    
    def test_manual_finish(self):
        """Test manual finish call."""
        config = {'experiment_name': 'manual_finish_test'}
        logger = WandbLogger(config, enabled=False)
        
        # Should not raise error even when disabled
        logger.finish()
        assert logger.run is None


class TestStandaloneFunctions:
    """Test standalone utility functions."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_create_wandb_logger(self, mock_wandb):
        """Test create_wandb_logger convenience function."""
        mock_wandb.init.return_value = MagicMock()
        
        config = {'experiment_name': 'convenience_test'}
        logger = create_wandb_logger(config, enabled=True)
        
        assert isinstance(logger, WandbLogger)
        assert logger.experiment_name == 'convenience_test'
    
    def test_get_wandb_url_from_experiment_name(self):
        """Test URL generation from experiment name."""
        experiment_name = "test_exp_2023-01-01_12-00-00"
        
        # Test without entity
        url = get_wandb_url_from_experiment_name(experiment_name)
        expected = "https://wandb.ai/medical-ssl-research/runs/test_exp_2023-01-01_12-00-00"
        assert url == expected
        
        # Test with custom project
        url = get_wandb_url_from_experiment_name(experiment_name, project="custom_project")
        expected = "https://wandb.ai/custom_project/runs/test_exp_2023-01-01_12-00-00"
        assert url == expected
        
        # Test with entity
        url = get_wandb_url_from_experiment_name(experiment_name, entity="team")
        expected = "https://wandb.ai/team/medical-ssl-research/runs/test_exp_2023-01-01_12-00-00"
        assert url == expected


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_log_metrics_error(self, mock_wandb):
        """Test metric logging with wandb error."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.log.side_effect = Exception("Logging failed")
        
        config = {'experiment_name': 'error_test'}
        logger = WandbLogger(config, enabled=True)
        
        with pytest.warns(UserWarning, match="Failed to log metrics"):
            logger.log_metrics({'loss': 0.5})
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_is_enabled(self, mock_wandb):
        """Test is_enabled method."""
        # Test enabled logger
        mock_wandb.init.return_value = MagicMock()
        config = {'experiment_name': 'enabled_test'}
        logger = WandbLogger(config, enabled=True)
        assert logger.is_enabled() == True
        
        # Test disabled logger
        logger_disabled = WandbLogger(config, enabled=False)
        assert logger_disabled.is_enabled() == False
    
    def test_empty_config(self):
        """Test with empty config."""
        config = {}
        logger = WandbLogger(config, enabled=False)
        
        assert logger.experiment_name == 'unknown_experiment'
        assert logger.enabled == False
    
    @patch('src.utils.wandb_logger.WANDB_AVAILABLE', True)
    @patch('src.utils.wandb_logger.wandb')
    def test_complex_config_flattening(self, mock_wandb):
        """Test flattening of deeply nested and complex configs."""
        mock_wandb.init.return_value = MagicMock()
        
        config = {
            'experiment_name': 'complex_test',
            'deeply': {
                'nested': {
                    'config': {
                        'with': {
                            'many': {
                                'levels': 42
                            }
                        }
                    }
                }
            },
            'list_values': [1, 2, 3],  # Should be converted to string
            'none_value': None
        }
        
        logger = WandbLogger(config, enabled=True)
        
        # Get flattened config
        call_kwargs = mock_wandb.init.call_args[1]
        flattened_config = call_kwargs['config']
        
        # Check deep nesting
        assert flattened_config['deeply.nested.config.with.many.levels'] == 42
        
        # Check list and None handling - lists are JSON serializable so stay as lists
        assert flattened_config['list_values'] == [1, 2, 3]  # Lists stay as lists
        assert flattened_config['none_value'] is None  # None stays as None

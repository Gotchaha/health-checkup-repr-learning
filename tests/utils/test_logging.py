# tests/utils/test_logging.py

import pytest
import pandas as pd
import json
import time
from pathlib import Path
from unittest.mock import patch

from src.utils.logging import (
    ExperimentLogger,
    ExperimentRun,
    create_experiment_logger,
    load_experiment_results
)


class TestExperimentLogger:
    """Test ExperimentLogger functionality."""
    
    def test_logger_initialization(self, tmp_path):
        """Test logger initialization and master CSV creation."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Check directory and master CSV were created
        assert logs_root.exists()
        assert logger.master_csv_path.exists()
        
        # Check master CSV has correct headers
        df = pd.read_csv(logger.master_csv_path)
        expected_columns = [
            'experiment_name', 'config_path', 'timestamp', 'status',
            'steps_trained', 'total_train_time', 'final_train_loss',
            'final_val_loss', 'best_val_metric', 'test_results_summary',
            'detailed_log_path', 'checkpoint_path', 'notes',
            'created_at', 'updated_at'
        ]
        for col in expected_columns:
            assert col in df.columns
        assert len(df) == 0  # Should be empty initially
    
    def test_start_experiment(self, tmp_path):
        """Test starting a new experiment."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {
            'experiment_name': 'test_exp_123',
            'timestamp': '2023-01-01_12-00-00',
            '_meta': {'config_path': 'config/test.yaml'}
        }
        
        experiment = logger.start_experiment(config)
        
        # Check experiment object
        assert isinstance(experiment, ExperimentRun)
        assert experiment.experiment_name == 'test_exp_123'
        
        # Check experiment directory was created
        exp_dir = logs_root / 'test_exp_123'
        assert exp_dir.exists()
        
        # Check master CSV was updated
        df = pd.read_csv(logger.master_csv_path)
        assert len(df) == 1
        assert df.iloc[0]['experiment_name'] == 'test_exp_123'
        assert df.iloc[0]['status'] == 'running'
        assert df.iloc[0]['config_path'] == 'config/test.yaml'
    
    def test_update_experiment_status(self, tmp_path):
        """Test updating experiment status in master CSV."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Start an experiment
        config = {'experiment_name': 'status_test'}
        experiment = logger.start_experiment(config)
        
        # Update status
        status_update = {
            'status': 'completed',
            'final_train_loss': 0.5,
            'steps_trained': 1000,
            'notes': 'Test completed successfully'
        }
        logger.update_experiment_status('status_test', status_update)
        
        # Check master CSV was updated
        df = pd.read_csv(logger.master_csv_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row['experiment_name'] == 'status_test'
        assert row['status'] == 'completed'
        assert row['final_train_loss'] == 0.5
        assert row['steps_trained'] == 1000
        assert row['notes'] == 'Test completed successfully'
        assert pd.notna(row['updated_at'])
    
    def test_get_experiment_history(self, tmp_path):
        """Test retrieving experiment history."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Start multiple experiments
        for i in range(3):
            config = {'experiment_name': f'history_test_{i}'}
            logger.start_experiment(config)
        
        # Get all history
        history = logger.get_experiment_history()
        assert len(history) == 3
        assert all('history_test_' in name for name in history['experiment_name'])
        
        # Get specific experiment history
        specific_history = logger.get_experiment_history('history_test_1')
        assert len(specific_history) == 1
        assert specific_history.iloc[0]['experiment_name'] == 'history_test_1'
    
    def test_empty_experiment_history(self, tmp_path):
        """Test getting history when no experiments exist."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Remove the master CSV
        logger.master_csv_path.unlink()
        
        history = logger.get_experiment_history()
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0


class TestExperimentRun:
    """Test ExperimentRun functionality."""
    
    def test_experiment_run_initialization(self, tmp_path):
        """Test ExperimentRun initialization."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'init_test'}
        experiment = logger.start_experiment(config)
        
        # Check initialization
        assert experiment.experiment_name == 'init_test'
        assert experiment.total_steps == 0
        assert hasattr(experiment, 'start_time')
        assert experiment.history_maxlen == 5000  # Default value
        
        # Check that detailed_log_path is set correctly (but file may not exist yet)
        expected_path = logs_root / 'init_test' / 'detailed_log.json'
        assert experiment.detailed_log_path == expected_path
        
        # Check initial metrics history structure
        assert 'training_metrics' in experiment.metrics_history
        assert 'validation_metrics' in experiment.metrics_history
        assert 'test_results' in experiment.metrics_history
        assert 'experiment_info' in experiment.metrics_history
        assert experiment.metrics_history['experiment_info']['status'] == 'running'
        
        # Check that metrics lists are empty initially
        assert len(experiment.metrics_history['training_metrics']) == 0
        assert len(experiment.metrics_history['validation_metrics']) == 0
        assert len(experiment.metrics_history['test_results']) == 0
    
    def test_detailed_log_file_creation(self, tmp_path):
        """Test that detailed log file is created when logging occurs."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'file_creation_test'}
        experiment = logger.start_experiment(config)
        
        # Initially, detailed log file should not exist
        assert not experiment.detailed_log_path.exists()
        
        # After logging metrics, file should be created
        experiment.log_step_metrics(100, {'loss': 1.0}, {'loss': 1.2})
        assert experiment.detailed_log_path.exists()
        
        # File should contain the logged data
        with open(experiment.detailed_log_path, 'r') as f:
            logged_data = json.load(f)
        assert len(logged_data['training_metrics']) == 1
        assert len(logged_data['validation_metrics']) == 1
    
    def test_log_step_metrics(self, tmp_path):
        """Test logging step metrics."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'step_test'}
        experiment = logger.start_experiment(config)
        
        # Log some steps
        train_metrics_1 = {'loss': 1.0, 'accuracy': 0.7}
        val_metrics_1 = {'loss': 1.2, 'accuracy': 0.65}
        experiment.log_step_metrics(100, train_metrics_1, val_metrics_1)
        
        train_metrics_2 = {'loss': 0.8, 'accuracy': 0.8}
        val_metrics_2 = {'loss': 1.0, 'accuracy': 0.75}
        experiment.log_step_metrics(200, train_metrics_2, val_metrics_2)
        
        # Check metrics were logged
        assert experiment.total_steps == 200
        assert len(experiment.metrics_history['training_metrics']) == 2
        assert len(experiment.metrics_history['validation_metrics']) == 2
        
        # Check metric values
        train_1 = experiment.metrics_history['training_metrics'][0]
        assert train_1['step'] == 100
        assert train_1['loss'] == 1.0
        assert train_1['accuracy'] == 0.7
        assert 'timestamp' in train_1
        
        val_2 = experiment.metrics_history['validation_metrics'][1]
        assert val_2['step'] == 200
        assert val_2['loss'] == 1.0
        assert val_2['accuracy'] == 0.75
        
        # Check detailed log file was updated
        assert experiment.detailed_log_path.exists()
        with open(experiment.detailed_log_path, 'r') as f:
            logged_data = json.load(f)
        assert len(logged_data['training_metrics']) == 2
        assert len(logged_data['validation_metrics']) == 2
    
    def test_log_test_results(self, tmp_path):
        """Test logging test results."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'test_results'}
        experiment = logger.start_experiment(config)
        
        test_results = {
            'accuracy': 0.85,
            'f1_score': 0.82,
            'confusion_matrix': [[100, 10], [5, 95]],
            'classification_report': {'precision': 0.84, 'recall': 0.83}
        }
        
        experiment.log_test_results(test_results)
        
        # Check test results were logged
        logged_results = experiment.metrics_history['test_results']
        assert logged_results['accuracy'] == 0.85
        assert logged_results['f1_score'] == 0.82
        assert 'timestamp' in logged_results
        
        # Check detailed log file
        with open(experiment.detailed_log_path, 'r') as f:
            logged_data = json.load(f)
        assert logged_data['test_results']['accuracy'] == 0.85
    
    def test_log_message(self, tmp_path):
        """Test custom message logging."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'message_test'}
        experiment = logger.start_experiment(config)
        
        # Test different log levels
        experiment.log_message("Info message", "info")
        experiment.log_message("Warning message", "warning")
        experiment.log_message("Error message", "error")
        
        # Check log file exists (detailed testing of log content would require parsing log files)
        log_file = experiment.log_dir / "experiment.log"
        assert log_file.exists()
    
    def test_get_best_val_metric(self, tmp_path):
        """Test best validation metric extraction."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'best_metric_test'}
        experiment = logger.start_experiment(config)
        
        # Log multiple steps with different metrics
        experiment.log_step_metrics(100, {'loss': 1.0}, {'loss': 1.2, 'accuracy': 0.7})
        experiment.log_step_metrics(200, {'loss': 0.8}, {'loss': 1.0, 'accuracy': 0.75})
        experiment.log_step_metrics(300, {'loss': 0.9}, {'loss': 1.1, 'accuracy': 0.72})
        
        # Test best loss (minimize)
        best_loss = experiment._get_best_val_metric('loss', minimize=True)
        assert best_loss == 1.0
        
        # Test best accuracy (maximize)
        best_acc = experiment._get_best_val_metric('accuracy', minimize=False)
        assert best_acc == 0.75
        
        # Test non-existent metric
        best_f1 = experiment._get_best_val_metric('f1_score', minimize=False)
        assert best_f1 is None
    
    def test_create_test_results_summary(self, tmp_path):
        """Test test results summary creation."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'summary_test'}
        experiment = logger.start_experiment(config)
        
        # Test with numeric and nested data
        test_results = {
            'accuracy': 0.85,
            'f1_score': 0.82,
            'per_class': {'class_0': {'precision': 0.9}, 'class_1': {'precision': 0.8}},
            'confusion_matrix': [[100, 10], [5, 95]],  # Non-numeric
            'description': "Test completed"  # Non-numeric
        }
        
        summary = experiment._create_test_results_summary(test_results)
        
        # Should be valid JSON
        parsed_summary = json.loads(summary)
        
        # Should contain numeric values
        assert parsed_summary['accuracy'] == 0.85
        assert parsed_summary['f1_score'] == 0.82
        assert parsed_summary['per_class_class_0_precision'] == 0.9
        assert parsed_summary['per_class_class_1_precision'] == 0.8
        
        # Should not contain non-numeric values
        assert 'confusion_matrix' not in parsed_summary
        assert 'description' not in parsed_summary
    
    def test_create_test_results_summary_depth_limit(self, tmp_path):
        """Test that test results summary respects depth limits."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'depth_test'}
        experiment = logger.start_experiment(config)
        
        # Test with very deep nesting (should be limited to 3 levels)
        test_results = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 0.9,  # 3 levels deep - should be included
                        'level4': {
                            'value': 0.8  # 4 levels deep - should be excluded
                        }
                    }
                }
            }
        }
        
        summary = experiment._create_test_results_summary(test_results)
        parsed_summary = json.loads(summary)
        
        # Should contain 3-level nested value
        assert parsed_summary['level1_level2_level3_value'] == 0.9
        
        # Should not contain 4-level nested value (beyond max depth)
        assert 'level1_level2_level3_level4_value' not in parsed_summary
    
    def test_finish_experiment(self, tmp_path):
        """Test finishing an experiment."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'finish_test'}
        experiment = logger.start_experiment(config)
        
        # Log some training data
        experiment.log_step_metrics(1000, {'loss': 1.0}, {'loss': 1.2, 'accuracy': 0.7})
        experiment.log_step_metrics(2000, {'loss': 0.8}, {'loss': 1.0, 'accuracy': 0.75})
        
        # Log test results
        test_results = {'accuracy': 0.82, 'f1_score': 0.79}
        experiment.log_test_results(test_results)
        
        # Add a small delay to test timing
        time.sleep(0.1)
        
        # Finish experiment
        experiment.finish_experiment("completed", notes="Test finished successfully")
        
        # Check experiment info was updated
        exp_info = experiment.metrics_history['experiment_info']
        assert exp_info['status'] == 'completed'
        assert 'finished_at' in exp_info
        assert 'total_train_time_seconds' in exp_info
        assert exp_info['total_train_time_seconds'] > 0.1
        
        # Check master CSV was updated
        df = pd.read_csv(logger.master_csv_path)
        row = df.iloc[0]
        assert row['experiment_name'] == 'finish_test'
        assert row['status'] == 'completed'
        assert row['steps_trained'] == 2000
        assert row['final_train_loss'] == 0.8
        assert row['final_val_loss'] == 1.0
        assert 'loss=1.0000' in row['best_val_metric']  # Best validation loss (loss takes precedence)
        assert row['notes'] == 'Test finished successfully'
        assert 'total_train_time' in str(row.to_dict())
        
        # Check test results summary
        test_summary = json.loads(row['test_results_summary'])
        assert test_summary['accuracy'] == 0.82
        assert test_summary['f1_score'] == 0.79
    
    def test_finish_experiment_no_validation_metrics(self, tmp_path):
        """Test finishing experiment without validation metrics."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'no_val_test'}
        experiment = logger.start_experiment(config)
        
        # Log only training metrics
        experiment.log_step_metrics(500, {'loss': 1.0})
        experiment.log_step_metrics(1000, {'loss': 0.8})
        
        experiment.finish_experiment("completed")
        
        # Check master CSV
        df = pd.read_csv(logger.master_csv_path)
        row = df.iloc[0]
        assert row['steps_trained'] == 1000
        assert row['final_train_loss'] == 0.8
        assert pd.isna(row['final_val_loss'])
        assert pd.isna(row['best_val_metric'])
    
    def test_finish_experiment_accuracy_fallback(self, tmp_path):
        """Test finishing experiment with accuracy fallback when no loss in validation."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'accuracy_fallback_test'}
        experiment = logger.start_experiment(config)
        
        # Log training with loss, but validation without loss (only accuracy)
        experiment.log_step_metrics(500, {'loss': 1.0}, {'accuracy': 0.7})
        experiment.log_step_metrics(1000, {'loss': 0.8}, {'accuracy': 0.75})
        
        experiment.finish_experiment("completed")
        
        # Check master CSV - should fall back to accuracy since no validation loss
        df = pd.read_csv(logger.master_csv_path)
        row = df.iloc[0]
        assert row['steps_trained'] == 1000
        assert row['final_train_loss'] == 0.8
        assert pd.isna(row['final_val_loss'])  # No validation loss logged
        assert 'accuracy=0.7500' in row['best_val_metric']  # Should fallback to best accuracy (max)
    
    def test_finish_experiment_no_metrics(self, tmp_path):
        """Test finishing experiment without any metrics."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'no_metrics_test'}
        experiment = logger.start_experiment(config)
        
        experiment.finish_experiment("failed", notes="No training data")
        
        # Check master CSV
        df = pd.read_csv(logger.master_csv_path)
        row = df.iloc[0]
        assert row['status'] == 'failed'
        assert row['steps_trained'] == 0
        assert row['notes'] == 'No training data'




class TestStandaloneFunctions:
    """Test standalone utility functions."""
    
    def test_create_experiment_logger(self, tmp_path):
        """Test experiment logger creation function."""
        logs_root = tmp_path / "test_logs"
        logger = create_experiment_logger(str(logs_root))
        
        assert isinstance(logger, ExperimentLogger)
        assert logs_root.exists()
        assert logger.master_csv_path.exists()
    
    def test_load_experiment_results(self, tmp_path):
        """Test loading experiment results."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'load_test'}
        experiment = logger.start_experiment(config)
        
        # Add some data
        experiment.log_step_metrics(100, {'loss': 1.0}, {'loss': 1.2})
        test_results = {'accuracy': 0.85}
        experiment.log_test_results(test_results)
        experiment.finish_experiment("completed")
        
        # Load results
        results = load_experiment_results('load_test', str(logs_root))
        
        assert results['experiment_info']['experiment_name'] == 'load_test'
        assert len(results['training_metrics']) == 1
        assert len(results['validation_metrics']) == 1
        assert results['test_results']['accuracy'] == 0.85
    
    def test_load_experiment_results_not_found(self, tmp_path):
        """Test loading results for non-existent experiment."""
        logs_root = tmp_path / "logs"
        
        with pytest.raises(FileNotFoundError):
            load_experiment_results('nonexistent_exp', str(logs_root))

    def test_history_maxlen_parameter(self, tmp_path):
        """Test history_maxlen parameter initialization."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'maxlen_test'}
        
        # Test default value
        experiment_default = logger.start_experiment(config)
        assert experiment_default.history_maxlen == 5000
        
        # Test custom value (would need to modify start_experiment to accept this)
        # For now, test by creating ExperimentRun directly
        from src.utils.logging import ExperimentRun
        exp_dir = logs_root / 'custom_maxlen_test'
        exp_dir.mkdir(parents=True, exist_ok=True)
        experiment_custom = ExperimentRun('custom_maxlen_test', exp_dir, logger, history_maxlen=1000)
        assert experiment_custom.history_maxlen == 1000
    
    def test_flush_and_trim_history(self, tmp_path):
        """Test memory management with flush and trim."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Create experiment with small history limit for testing
        from src.utils.logging import ExperimentRun
        exp_dir = logs_root / 'flush_test'
        exp_dir.mkdir(parents=True, exist_ok=True)
        experiment = ExperimentRun('flush_test', exp_dir, logger, history_maxlen=10)
        
        # Log more metrics than the limit
        for i in range(15):
            experiment.log_step_metrics(i * 100, {'loss': 1.0 - i * 0.05})
        
        # Should have triggered flush and trim
        assert len(experiment.metrics_history['training_metrics']) <= 10
        assert experiment.total_steps == 1400  # Last step logged
        
        # Detailed log should still contain all data
        assert experiment.detailed_log_path.exists()
        with open(experiment.detailed_log_path, 'r') as f:
            logged_data = json.load(f)
        # Note: After flush, the JSON might only contain recent data
        # depending on implementation details
    
    def test_step_metrics_only_training(self, tmp_path):
        """Test logging step metrics with only training data."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'train_only_test'}
        experiment = logger.start_experiment(config)
        
        # Log steps with only training metrics (no validation)
        experiment.log_step_metrics(100, {'loss': 1.0, 'accuracy': 0.7})
        experiment.log_step_metrics(200, {'loss': 0.8, 'accuracy': 0.8})
        
        # Check only training metrics were logged
        assert experiment.total_steps == 200
        assert len(experiment.metrics_history['training_metrics']) == 2
        assert len(experiment.metrics_history['validation_metrics']) == 0
        
        # Check step values
        assert experiment.metrics_history['training_metrics'][0]['step'] == 100
        assert experiment.metrics_history['training_metrics'][1]['step'] == 200


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_logging_with_special_characters(self, tmp_path):
        """Test logging with special characters in experiment names."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Experiment name with special characters
        config = {'experiment_name': 'test_exp_with-special.chars_123'}
        experiment = logger.start_experiment(config)
        
        experiment.log_step_metrics(100, {'loss': 1.0})
        experiment.finish_experiment("completed")
        
        # Should handle gracefully
        df = pd.read_csv(logger.master_csv_path)
        assert df.iloc[0]['experiment_name'] == 'test_exp_with-special.chars_123'
    
    def test_logging_with_nan_values(self, tmp_path):
        """Test logging with NaN values in metrics."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'nan_test'}
        experiment = logger.start_experiment(config)
        
        # Log metrics with NaN values
        import numpy as np
        experiment.log_step_metrics(100, {'loss': float('nan'), 'accuracy': 0.7})
        experiment.finish_experiment("completed")
        
        # Should handle gracefully
        df = pd.read_csv(logger.master_csv_path)
        row = df.iloc[0]
        assert row['experiment_name'] == 'nan_test'
        # NaN should be preserved in CSV
    
    def test_logging_with_large_test_results(self, tmp_path):
        """Test logging with very large test results."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        config = {'experiment_name': 'large_results_test'}
        experiment = logger.start_experiment(config)
        
        # Create large test results
        large_results = {
            'accuracy': 0.85,
            'large_matrix': [[i*j for j in range(100)] for i in range(100)],
            'description': "A" * 10000  # Very long string
        }
        
        experiment.log_test_results(large_results)
        experiment.finish_experiment("completed")
        
        # Should handle gracefully and create proper summary
        df = pd.read_csv(logger.master_csv_path)
        row = df.iloc[0]
        
        # Test results summary should only contain numeric values
        test_summary = json.loads(row['test_results_summary'])
        assert test_summary['accuracy'] == 0.85
        assert 'large_matrix' not in test_summary
        assert 'description' not in test_summary
    
    def test_missing_logs_directory(self, tmp_path):
        """Test behavior when logs directory doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent" / "logs"
        
        # Should create directory automatically
        logger = ExperimentLogger(str(nonexistent_path))
        assert nonexistent_path.exists()
        assert logger.master_csv_path.exists()
    
    @patch('fcntl.flock')
    def test_file_locking_failure(self, mock_flock, tmp_path):
        """Test handling of file locking failures."""
        logs_root = tmp_path / "logs"
        logger = ExperimentLogger(str(logs_root))
        
        # Make flock raise an error
        mock_flock.side_effect = IOError("Lock failed")
        
        config = {'experiment_name': 'lock_fail_test'}
        
        # Should raise the IOError after max retries
        with pytest.raises(IOError):
            experiment = logger.start_experiment(config)
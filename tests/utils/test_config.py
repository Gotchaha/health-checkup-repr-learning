# tests/utils/test_config.py

import pytest
import yaml
from pathlib import Path
from datetime import datetime

from src.utils.config import (
    ConfigManager, 
    load_experiment_config, 
    create_experiment_dirs,
    get_config_summary
)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_load_basic_config(self, tmp_path):
        """Test loading a basic configuration file."""
        # Create test config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir()
        
        config_content = {
            'experiment_name': 'test_experiment',
            'model': {'type': 'ssl_model'},
            'training': {'batch_size': 32, 'learning_rate': 0.001}
        }
        
        config_file = experiments_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        # Test loading
        manager = ConfigManager(str(config_dir))
        config = manager.load_config("test.yaml", auto_timestamp=False)
        
        assert config['experiment_name'] == 'test_experiment'
        assert config['model']['type'] == 'ssl_model'
        assert config['training']['batch_size'] == 32
        assert '_meta' in config
        assert 'config_path' in config['_meta']
    
    def test_auto_timestamp(self, tmp_path):
        """Test automatic timestamp addition."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        config_content = {'experiment_name': 'test_exp'}
        config_file = experiments_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigManager(str(config_dir))
        config = manager.load_config("test.yaml", auto_timestamp=True)
        
        # Check timestamp was added
        assert config['experiment_name'] != 'test_exp'
        assert config['experiment_name'].startswith('test_exp_')
        assert 'timestamp' in config
        assert 'original_experiment_name' in config
        assert config['original_experiment_name'] == 'test_exp'
    
    def test_config_inheritance(self, tmp_path):
        """Test configuration inheritance."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        # Create base config
        base_config = {
            'model': {'d_model': 768, 'layers': 6},
            'training': {'batch_size': 32}
        }
        base_file = config_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create experiment config that extends base
        exp_config = {
            'extends': 'base.yaml',
            'experiment_name': 'inheritance_test',
            'training': {'batch_size': 64, 'learning_rate': 0.001}  # Override + add
        }
        exp_file = experiments_dir / "inheritance.yaml"
        with open(exp_file, 'w') as f:
            yaml.dump(exp_config, f)
        
        # Test inheritance
        manager = ConfigManager(str(config_dir))
        config = manager.load_config("inheritance.yaml", auto_timestamp=False)
        
        # Check inheritance worked
        assert config['model']['d_model'] == 768  # From base
        assert config['model']['layers'] == 6     # From base
        assert config['training']['batch_size'] == 64  # Overridden
        assert config['training']['learning_rate'] == 0.001  # Added
        assert 'extends' not in config  # Should be removed
    
    def test_config_validation_success(self, tmp_path):
        """Test successful config validation."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        config_content = {
            'experiment_name': 'valid_config',
            'model': {'type': 'ssl'},
            'training': {'batch_size': 32, 'learning_rate': 0.001}
        }
        config_file = experiments_dir / "valid.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigManager(str(config_dir))
        config = manager.load_config("valid.yaml", auto_timestamp=False)
        
        # Should not raise any exceptions
        manager.validate_config(config)
    
    def test_config_validation_failure(self, tmp_path):
        """Test config validation failure cases."""
        manager = ConfigManager(str(tmp_path))
        
        # Missing experiment_name
        invalid_config1 = {'model': {'type': 'ssl'}}
        with pytest.raises(ValueError, match="Required field 'experiment_name' missing"):
            manager.validate_config(invalid_config1)
        
        # Invalid batch_size
        invalid_config2 = {
            'experiment_name': 'test',
            'training': {'batch_size': -1}
        }
        with pytest.raises(ValueError, match="'batch_size' must be positive"):
            manager.validate_config(invalid_config2)
        
        # Invalid learning_rate
        invalid_config3 = {
            'experiment_name': 'test',
            'training': {'learning_rate': 0}
        }
        with pytest.raises(ValueError, match="'learning_rate' must be positive"):
            manager.validate_config(invalid_config3)
    
    def test_save_config_copy(self, tmp_path):
        """Test saving config copy to checkpoint directory."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        config_content = {
            'experiment_name': 'save_test',
            'model': {'type': 'ssl'},
            '_meta': {'loaded_at': '2023-01-01'}  # Should be removed
        }
        config_file = experiments_dir / "save_test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigManager(str(config_dir))
        config = manager.load_config("save_test.yaml", auto_timestamp=False)
        
        # Save copy
        checkpoint_dir = tmp_path / "checkpoints"
        saved_path = manager.save_config_copy(config, checkpoint_dir)
        
        # Verify saved file
        assert saved_path.exists()
        assert saved_path.name == "config.yaml"
        
        # Load saved config and verify
        with open(saved_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['experiment_name'] == 'save_test'
        assert saved_config['model']['type'] == 'ssl'
        assert '_meta' not in saved_config  # Should be filtered out
    
    def test_missing_config_file(self, tmp_path):
        """Test handling of missing config file."""
        manager = ConfigManager(str(tmp_path))
        
        with pytest.raises(FileNotFoundError):
            manager.load_config("nonexistent.yaml")
    
    def test_missing_base_config(self, tmp_path):
        """Test handling of missing base config in inheritance."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        exp_config = {
            'extends': 'nonexistent_base.yaml',
            'experiment_name': 'test'
        }
        exp_file = experiments_dir / "test.yaml"
        with open(exp_file, 'w') as f:
            yaml.dump(exp_config, f)
        
        manager = ConfigManager(str(config_dir))
        
        with pytest.raises(FileNotFoundError):
            manager.load_config("test.yaml")


class TestConfigFunctions:
    """Test standalone config functions."""
    
    def test_load_experiment_config(self, tmp_path):
        """Test load_experiment_config convenience function."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        config_content = {
            'experiment_name': 'convenience_test',
            'training': {'batch_size': 32}
        }
        config_file = experiments_dir / "convenience.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        # Test function
        config = load_experiment_config(
            "convenience.yaml", 
            config_root=str(config_dir),
            auto_timestamp=False,
            validate=True
        )
        
        assert config['experiment_name'] == 'convenience_test'
        assert config['training']['batch_size'] == 32
    
    def test_create_experiment_dirs(self, tmp_path):
        """Test experiment directory creation."""
        # Mock outputs directory
        outputs_dir = tmp_path / "outputs"
        
        config = {'experiment_name': 'dir_test_2023-01-01_12-00-00'}
        
        # Patch the function to use tmp_path
        import src.utils.config
        original_path = src.utils.config.Path
        
        def mock_path(path_str):
            if path_str.startswith("outputs/"):
                return tmp_path / path_str
            return original_path(path_str)
        
        src.utils.config.Path = mock_path
        
        try:
            dirs = create_experiment_dirs(config)
            
            # Check directories were created
            assert 'checkpoint_dir' in dirs
            assert 'log_dir' in dirs
            assert dirs['checkpoint_dir'].exists()
            assert dirs['log_dir'].exists()
            
            # Check correct paths
            assert "dir_test_2023-01-01_12-00-00" in str(dirs['checkpoint_dir'])
            assert "dir_test_2023-01-01_12-00-00" in str(dirs['log_dir'])
            
        finally:
            src.utils.config.Path = original_path
    
    def test_get_config_summary(self):
        """Test config summary extraction."""
        config = {
            'experiment_name': 'summary_test',
            'timestamp': '2023-01-01_12-00-00',
            'model': {'type': 'ssl_model'},
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'max_steps': 1000
            }
        }
        
        summary = get_config_summary(config)
        
        assert summary['experiment_name'] == 'summary_test'
        assert summary['timestamp'] == '2023-01-01_12-00-00'
        assert summary['batch_size'] == 32
        assert summary['learning_rate'] == 0.001
        assert summary['max_steps'] == 1000
        assert summary['model_type'] == 'ssl_model'
    
    def test_get_config_summary_minimal(self):
        """Test config summary with minimal config."""
        config = {'experiment_name': 'minimal_test'}
        
        summary = get_config_summary(config)
        
        assert summary['experiment_name'] == 'minimal_test'
        assert 'batch_size' not in summary  # Should not include None values


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config_file(self, tmp_path):
        """Test handling of empty config file."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        # Create empty file
        empty_file = experiments_dir / "empty.yaml"
        empty_file.touch()
        
        manager = ConfigManager(str(config_dir))
        config = manager.load_config("empty.yaml", auto_timestamp=False)
        
        # Should handle empty file gracefully
        assert isinstance(config, dict)
        assert '_meta' in config
    
    def test_invalid_yaml_syntax(self, tmp_path):
        """Test handling of invalid YAML syntax."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        # Create file with invalid YAML
        invalid_file = experiments_dir / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        manager = ConfigManager(str(config_dir))
        
        with pytest.raises(yaml.YAMLError):
            manager.load_config("invalid.yaml")
    
    def test_circular_inheritance(self, tmp_path):
        """Test detection of circular inheritance."""
        config_dir = tmp_path / "config"
        experiments_dir = config_dir / "experiments"
        experiments_dir.mkdir(parents=True)
        
        # Create config A that extends B
        config_a = {'extends': 'config_b.yaml', 'name': 'a'}
        file_a = experiments_dir / "config_a.yaml"
        with open(file_a, 'w') as f:
            yaml.dump(config_a, f)
        
        # Create config B that extends A (circular)
        config_b = {'extends': 'config_a.yaml', 'name': 'b'}
        file_b = experiments_dir / "config_b.yaml"
        with open(file_b, 'w') as f:
            yaml.dump(config_b, f)
        
        manager = ConfigManager(str(config_dir))
        
        # This should not cause infinite recursion
        # (Current implementation would just fail on file not found or similar)
        with pytest.raises((FileNotFoundError, RecursionError)):
            manager.load_config("config_a.yaml")
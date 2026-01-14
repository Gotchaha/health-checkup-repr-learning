# src/utils/config.py

import os
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigManager:
    """
    Configuration manager for experiment settings.
    
    Supports hierarchical YAML configs with inheritance.
    Automatically handles config copying to checkpoint directories for reproducibility.
    """
    
    def __init__(self, config_root: str = "config"):
        """
        Initialize ConfigManager.
        
        Args:
            config_root: Root directory for configuration files
        """
        self.config_root = Path(config_root)
        self.base_config_path = self.config_root / "base.yaml"
        self.experiments_dir = self.config_root / "experiments"
    
    def load_config(self, config_path: Union[str, Path], auto_timestamp: bool = True) -> Dict[str, Any]:
        """
        Load configuration from YAML file with support for inheritance.
        
        Args:
            config_path: Path to experiment config file (relative to config root or absolute)
            auto_timestamp: Whether to automatically add timestamp to experiment name
            
        Returns:
            Complete configuration dictionary
        """
        config_path = Path(config_path)
        
        # Handle relative paths
        if not config_path.is_absolute():
            if config_path.parent.name != "experiments":
                config_path = self.experiments_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load experiment config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Handle inheritance
        if 'extends' in config:
            base_config = self._load_base_config(config['extends'])
            config = self._merge_configs(base_config, config)
            del config['extends']  # Remove extends key from final config
        
        # Add timestamp to experiment name if requested
        if auto_timestamp and 'experiment_name' in config:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            original_name = config['experiment_name']
            config['experiment_name'] = f"{original_name}_{timestamp}"
            config['timestamp'] = timestamp
            config['original_experiment_name'] = original_name
        

        
        # Store config metadata
        config['_meta'] = {
            'config_path': str(config_path),
            'loaded_at': datetime.now().isoformat(),
            'config_root': str(self.config_root)
        }
        
        return config
    
    def _load_base_config(self, base_name: str) -> Dict[str, Any]:
        """Load base configuration file."""
        if base_name == "base.yaml" or base_name == "base":
            base_path = self.base_config_path
        else:
            base_path = self.config_root / base_name
        
        if not base_path.exists():
            raise FileNotFoundError(f"Base config file not found: {base_path}")
        
        with open(base_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        Override values take precedence over base values.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    

    
    def save_config_copy(self, config: Dict[str, Any], checkpoint_dir: Union[str, Path]) -> Path:
        """
        Save a copy of the configuration to the checkpoint directory.
        
        Args:
            config: Configuration dictionary
            checkpoint_dir: Directory to save config copy
            
        Returns:
            Path to saved config file
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        config_copy_path = checkpoint_dir / "config.yaml"
        
        # Remove metadata before saving
        config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
        
        with open(config_copy_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
        
        return config_copy_path
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Basic configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['experiment_name']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' missing from configuration")
        
        # Validate model config structure
        if 'model' in config:
            model_config = config['model']
            if not isinstance(model_config, dict):
                raise ValueError("'model' configuration must be a dictionary")
        
        # Validate training config structure
        if 'training' in config:
            training_config = config['training']
            if not isinstance(training_config, dict):
                raise ValueError("'training' configuration must be a dictionary")
            
            # Check for common training parameters
            if 'batch_size' in training_config and training_config['batch_size'] <= 0:
                raise ValueError("'batch_size' must be positive")
            
            if 'learning_rate' in training_config and training_config['learning_rate'] <= 0:
                raise ValueError("'learning_rate' must be positive")


def load_experiment_config(config_path: Union[str, Path], 
                          config_root: str = "config",
                          auto_timestamp: bool = True,
                          validate: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load and validate experiment configuration.
    
    Args:
        config_path: Path to experiment config file
        config_root: Root directory for configuration files
        auto_timestamp: Whether to automatically add timestamp to experiment name
        validate: Whether to validate the configuration
        
    Returns:
        Complete configuration dictionary
    """
    manager = ConfigManager(config_root)
    config = manager.load_config(config_path, auto_timestamp=auto_timestamp)
    
    if validate:
        manager.validate_config(config)
    
    return config


def create_experiment_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create experiment directories based on configuration.
    
    Args:
        config: Configuration dictionary with experiment_name
        
    Returns:
        Dictionary with created directory paths
    """
    experiment_name = config['experiment_name']
    paths_cfg = config.get('paths', {})

    outputs_root = Path(paths_cfg.get('outputs_root', 'outputs'))
    checkpoints_subdir = paths_cfg.get('checkpoints_subdir', 'checkpoints')
    logs_subdir = paths_cfg.get('logs_subdir', 'logs')

    checkpoint_dir = outputs_root / checkpoints_subdir / experiment_name
    log_dir = outputs_root / logs_subdir / experiment_name

    # Always create log directory (used for experiment logs, monitoring, etc.)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint directory only if enabled
    training_cfg = config.get('training', {})
    checkpoint_cfg = training_cfg.get('checkpoint', {})
    if checkpoint_cfg.get('enabled', True):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
    }


def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key configuration parameters for logging.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Summary dictionary with key parameters
    """
    summary = {
        'experiment_name': config.get('experiment_name', 'unknown'),
        'timestamp': config.get('timestamp', 'unknown'),
    }
    
    # Training parameters
    if 'training' in config:
        training = config['training']
        summary.update({
            'batch_size': training.get('batch_size'),
            'learning_rate': training.get('learning_rate'),
            'max_steps': training.get('max_steps'),
        })
    
    # Model parameters
    if 'model' in config:
        model = config['model']
        summary['model_type'] = model.get('type', 'unknown')
    
    return {k: v for k, v in summary.items() if v is not None}

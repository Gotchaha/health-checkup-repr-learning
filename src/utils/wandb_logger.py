# src/utils/wandb_logger.py

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with: pip install wandb")


class WandbLogger:
    """
    Weights & Biases logger that integrates with local experiment system.
    
    Uses the same config and experiment naming as local logging for perfect consistency.
    Supports optional usage - can be disabled via config or if wandb is not installed.
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Initialize WandbLogger.
        
        Args:
            config: Experiment configuration dictionary (same as local logging)
            enabled: Whether to enable wandb logging
        """
        self.config = config
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.experiment_name = config.get('experiment_name', 'unknown_experiment')
        wandb_section = self.config.get('wandb', {})
        self._wandb_config = wandb_section if isinstance(wandb_section, dict) else {}
        log_section = self._wandb_config.get('log', {})
        self._log_config = log_section if isinstance(log_section, dict) else {}
        watch_section = self._wandb_config.get('watch', {})
        self._watch_config = watch_section if isinstance(watch_section, dict) else {}
        
        if not WANDB_AVAILABLE and enabled:
            warnings.warn("wandb not available but enabled=True. Logging will be disabled.")
            self.enabled = False
        
        if self.enabled:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize wandb run with config settings."""
        wandb_config = self._wandb_config
        # Use experiment_name as run name for consistency with local logging
        run_name = self.experiment_name
        mode = self._resolve_wandb_mode(wandb_config)
        
        if mode == 'disabled':
            self.enabled = False
            self.run = None
            return
        
        # Set up wandb initialization parameters
        init_kwargs = {
            'project': wandb_config.get('project', 'medical-ssl-research'),
            'name': run_name,
            'id': run_name,  # Force run ID = experiment name for consistent linking
            'config': self._prepare_config_for_wandb(),
            'mode': mode,
            'save_code': wandb_config.get('save_code', True),
            'notes': wandb_config.get('notes', ''),
        }
        
        # Optional parameters
        if 'group' in wandb_config:
            init_kwargs['group'] = wandb_config['group']
        
        if 'tags' in wandb_config:
            init_kwargs['tags'] = wandb_config['tags']
        
        if 'entity' in wandb_config:
            init_kwargs['entity'] = wandb_config['entity']
        
        try:
            self.run = wandb.init(**init_kwargs)
            
            # Store wandb info for easy access
            if self.run:
                self.wandb_url = self.run.url
                self.wandb_id = self.run.id
                version_str = getattr(wandb, '__version__', 'unknown')
                print(f"wandb run initialized: {self.wandb_url} (wandb v{version_str})")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
            self.enabled = False
            self.run = None
    
    def _resolve_wandb_mode(self, wandb_config: Dict[str, Any]) -> str:
        """Determine wandb run mode."""
        mode = str(wandb_config.get('mode', 'online')).lower()
        if mode not in {'online', 'offline', 'disabled'}:
            warnings.warn(f"Unknown wandb mode '{mode}', defaulting to 'online'.")
            mode = 'online'
        return mode
    
    def _prepare_config_for_wandb(self) -> Dict[str, Any]:
        """
        Prepare config for wandb, removing internal fields and flattening nested structures.
        
        Returns:
            Flattened config dictionary for wandb dashboard
        """
        # Remove internal fields that shouldn't go to wandb
        config_for_wandb = {}
        
        for key, value in self.config.items():
            # Skip internal metadata and wandb config itself
            if key.startswith('_') or key == 'wandb':
                continue
            
            config_for_wandb[key] = value
        
        # Flatten nested dictionaries for better wandb dashboard visualization
        flattened_config = self._flatten_dict(config_for_wandb, sep='.')
        
        # Ensure all values are JSON serializable
        serializable_config = {}
        for key, value in flattened_config.items():
            try:
                import json
                json.dumps(value)
                serializable_config[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable values to strings
                serializable_config[key] = str(value)
        
        return serializable_config
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (training iteration, global step, etc.)
        """
        if not self.is_enabled():
            return
        if not self._log_feature_enabled('metrics', default=True):
            return
        
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            warnings.warn(f"Failed to log metrics to wandb: {e}")
    
    def log_step_metrics(self, step: int, train_metrics: Dict[str, float], 
                        val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log training and validation metrics for a step.
        
        Args:
            step: Step number
            train_metrics: Training metrics dictionary
            val_metrics: Optional validation metrics dictionary
        """
        if not self.is_enabled():
            return
        if not self._log_feature_enabled('metrics', default=True):
            return
        
        # Prepare metrics with prefixes for clarity
        metrics_to_log = {}
        
        # Add training metrics with prefix
        for key, value in train_metrics.items():
            metrics_to_log[f"train/{key}"] = value
        
        # Add validation metrics with prefix
        if val_metrics:
            for key, value in val_metrics.items():
                metrics_to_log[f"val/{key}"] = value
        
        # Log with step as step
        self.log_metrics(metrics_to_log, step=step)
    
    def log_test_results(self, test_results: Dict[str, Any]) -> None:
        """
        Log final test results.
        
        Args:
            test_results: Dictionary with test results
        """
        if not self.is_enabled():
            return

        metrics_logging_enabled = self._log_feature_enabled('metrics', default=True)
        details_logging_enabled = self._log_feature_enabled('test_details', default=False)
        
        # Prepare test metrics with prefix
        test_metrics = {}
        contains_non_numeric = any(not isinstance(v, (int, float)) for v in test_results.values())
        if metrics_logging_enabled:
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    test_metrics[f"test/{key}"] = value
            if test_metrics:
                self.log_metrics(test_metrics)
        
        if details_logging_enabled and contains_non_numeric:
            try:
                import pandas as pd
                flattened = self._flatten_dict(test_results)
                df = pd.DataFrame([flattened])
                table = wandb.Table(dataframe=df)
                wandb.log({"test_results_detailed": table})
            except Exception as e:
                warnings.warn(f"Failed to log detailed test results: {e}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested dictionary for wandb table display."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _log_feature_enabled(self, feature: str, default: bool = False) -> bool:
        """Check whether a specific wandb logging feature is enabled."""
        return bool(self._log_config.get(feature, default))
    
    def _safe_int(self, value: Any, default: int) -> int:
        """Safely convert a value to int with fallback."""
        try:
            return int(value)
        except (TypeError, ValueError):
            warnings.warn(f"Invalid integer value '{value}', defaulting to {default}.")
            return default
    
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "model", 
                    name: Optional[str] = None, description: Optional[str] = None) -> None:
        """
        Log file as wandb artifact.
        
        Args:
            file_path: Path to file to upload
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Optional artifact name (defaults to filename)
            description: Optional description
        """
        if not self.is_enabled():
            return
        if not self._log_feature_enabled('artifacts', default=False):
            return
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                warnings.warn(f"Artifact file not found: {file_path}")
                return
            
            artifact_name = name or f"{self.experiment_name}_{artifact_type}"
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=description
            )
            
            artifact.add_file(str(file_path))
            
            # Use recommended wandb.log_artifact() for better compatibility
            if hasattr(wandb, 'log_artifact'):
                wandb.log_artifact(artifact)
            else:
                # Fallback for older wandb versions
                self.run.log_artifact(artifact)
            
        except Exception as e:
            warnings.warn(f"Failed to log artifact: {e}")
    
    def log_checkpoint(self, checkpoint_path: Union[str, Path], step: int, 
                      is_best: bool = False) -> None:
        """
        Log model checkpoint as artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Step number
            is_best: Whether this is the best checkpoint
        """
        if not self.is_enabled():
            return
        
        artifact_name = f"checkpoint_step_{step}"
        if is_best:
            artifact_name += "_best"
        
        description = f"Model checkpoint at step {step}"
        if is_best:
            description += " (best validation performance)"
        
        self.log_artifact(
            checkpoint_path, 
            artifact_type="model",
            name=artifact_name,
            description=description
        )
    
    def log_config_file(self, config_path: Union[str, Path]) -> None:
        """
        Log configuration file as artifact.
        
        Args:
            config_path: Path to config file
        """
        if not self.is_enabled():
            return
        
        self.log_artifact(
            config_path,
            artifact_type="config", 
            name=f"{self.experiment_name}_config",
            description="Experiment configuration file"
        )
    
    def watch_model(self, model, log_freq: Optional[int] = None, log_graph: bool = True) -> None:
        """
        Configure wandb to watch a PyTorch model.
        
        Args:
            model: PyTorch model to watch
            log_freq: Optional override for histogram logging frequency
            log_graph: Whether to log the computation graph
        """
        if not self.is_enabled():
            return
        if not self._log_feature_enabled('watch_model', default=False):
            return
        
        watch_cfg = self._watch_config
        
        log_mode = str(watch_cfg.get('log', 'gradients')).lower()
        if log_mode not in {'gradients', 'parameters', 'all'}:
            warnings.warn(f"Unknown wandb.watch log mode '{log_mode}', defaulting to 'gradients'.")
            log_mode = 'gradients'
        
        resolved_log_freq = log_freq
        if resolved_log_freq is None:
            cfg_freq = watch_cfg.get('log_freq')
            if cfg_freq is None:
                cfg_freq = self._wandb_config.get('log_freq', 1000)
            resolved_log_freq = self._safe_int(cfg_freq, default=1000)
        
        if 'log_graph' in watch_cfg:
            log_graph = bool(watch_cfg['log_graph'])
        
        try:
            wandb.watch(model, log=log_mode, log_freq=resolved_log_freq, log_graph=log_graph)
        except Exception as e:
            warnings.warn(f"Failed to watch model: {e}")
    
    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                print(f"wandb run finished: {self.wandb_url}")
            except Exception as e:
                warnings.warn(f"Error finishing wandb run: {e}")
            finally:
                self.run = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
    
    def is_enabled(self) -> bool:
        """Check if wandb logging is enabled and working."""
        return self.enabled and self.run is not None


def create_wandb_logger(config: Dict[str, Any], enabled: bool = True) -> WandbLogger:
    """
    Convenience function to create a WandbLogger.
    
    Args:
        config: Experiment configuration dictionary
        enabled: Whether to enable wandb logging
        
    Returns:
        WandbLogger instance
    """
    return WandbLogger(config, enabled=enabled)


def get_wandb_url_from_experiment_name(experiment_name: str, 
                                      project: str = "medical-ssl-research",
                                      entity: Optional[str] = None) -> str:
    """
    Generate wandb URL from experiment name for easy linking.
    
    Args:
        experiment_name: Name of experiment (matches local logging)
        project: wandb project name
        entity: Optional wandb entity (username/team)
        
    Returns:
        wandb run URL
    """
    if entity:
        return f"https://wandb.ai/{entity}/{project}/runs/{experiment_name}"
    else:
        return f"https://wandb.ai/{project}/runs/{experiment_name}"

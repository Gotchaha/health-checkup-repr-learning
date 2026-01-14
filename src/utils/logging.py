# src/utils/logging.py

import json
import torch
import logging
import pandas as pd
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import fcntl
import os


class ExperimentLogger:
    """
    Two-tier logging system for ML experiments.
    
    - Master CSV: High-level experiment tracking (one row per experiment)
    - Individual JSON: Detailed results and metrics per experiment
    
    Thread-safe with file locking for concurrent experiment runs.
    """
    
    def __init__(self, logs_root: Union[str, Path] = "outputs/logs"):
        """
        Initialize ExperimentLogger.
        
        Args:
            logs_root: Root directory for all logging outputs
        """
        self.logs_root = Path(logs_root)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        
        self.master_csv_path = self.logs_root / "master_log.csv"
        self._init_master_csv()
    
    def _init_master_csv(self) -> None:
        """Initialize master CSV file with headers if it doesn't exist."""
        if not self.master_csv_path.exists():
            # Define master CSV columns
            columns = [
                'experiment_name',
                'config_path', 
                'timestamp',
                'status',
                'steps_trained',
                'total_train_time',
                'final_train_loss',
                'final_val_loss',
                'best_val_metric',
                'test_results_summary',
                'detailed_log_path',
                'checkpoint_path',
                'notes',
                'created_at',
                'updated_at'
            ]
            
            # Create empty DataFrame with columns
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.master_csv_path, index=False)
    
    def start_experiment(self, config: Dict[str, Any]) -> 'ExperimentRun':
        """
        Start a new experiment run with logging setup.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            ExperimentRun object for detailed logging
        """
        experiment_name = config['experiment_name']
        logging_cfg = config.get('experiment_logging', {})
        if not logging_cfg.get('enabled', True):
            return NullExperimentRun(experiment_name)

        # Determine experiment-specific directories using config paths
        paths_cfg = config.get('paths', {})
        outputs_root = Path(paths_cfg.get('outputs_root', 'outputs'))
        checkpoints_subdir = paths_cfg.get('checkpoints_subdir', 'checkpoints')

        exp_log_dir = self.logs_root / experiment_name
        exp_log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = outputs_root / checkpoints_subdir / experiment_name
        
        # Create initial master CSV entry
        initial_entry = {
            'experiment_name': experiment_name,
            'config_path': config.get('_meta', {}).get('config_path', 'unknown'),
            'timestamp': config.get('timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            'status': 'running',
            'steps_trained': 0,
            'detailed_log_path': str(exp_log_dir / "detailed_log.json"),
            'checkpoint_path': str(checkpoint_dir),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self._append_to_master_csv(initial_entry)
        
        # Create ExperimentRun for detailed logging
        return ExperimentRun(experiment_name, exp_log_dir, self)
    
    def _append_to_master_csv(self, entry: Dict[str, Any]) -> None:
        """
        Safely append entry to master CSV with atomic writes and proper locking.
        
        Args:
            entry: Dictionary with experiment data
        """
        # Define dtypes for columns that can contain mixed types to avoid pandas warnings
        dtype_spec = {
            'experiment_name': 'object',
            'config_path': 'object',
            'timestamp': 'object',
            'status': 'object',
            'steps_trained': 'object',  # Can be int or NaN
            'total_train_time': 'object',
            'final_train_loss': 'object',  # Can be float or NaN
            'final_val_loss': 'object',   # Can be float or NaN
            'best_val_metric': 'object',  # String like "loss=0.1234"
            'test_results_summary': 'object',  # JSON string
            'detailed_log_path': 'object',
            'checkpoint_path': 'object',
            'notes': 'object',
            'created_at': 'object',
            'updated_at': 'object'
        }
        
        # Read current CSV with proper dtypes
        try:
            df = pd.read_csv(self.master_csv_path, dtype=dtype_spec)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame(columns=list(dtype_spec.keys())).astype(dtype_spec)
        
        # Check if experiment already exists (update instead of append)
        experiment_name = entry['experiment_name']
        if 'experiment_name' in df.columns and experiment_name in df['experiment_name'].values:
            # Update existing entry
            mask = df['experiment_name'] == experiment_name
            for key, value in entry.items():
                if key in df.columns:
                    df.loc[mask, key] = value
        else:
            # Append new entry
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        
        # Atomic write with blocking lock
        self._atomic_csv_write(df, self.master_csv_path)
    
    def _atomic_csv_write(self, df: pd.DataFrame, file_path: Path, max_retries: int = 3) -> None:
        """
        Atomically write DataFrame to CSV with proper file locking.
        
        Args:
            df: DataFrame to write
            file_path: Target file path
            max_retries: Maximum number of lock retry attempts
        """
        # Write to temporary file first
        temp_dir = file_path.parent
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', dir=temp_dir, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            try:
                # Try to acquire lock with retries
                for attempt in range(max_retries):
                    try:
                        fcntl.flock(temp_file.fileno(), fcntl.LOCK_EX)
                        df.to_csv(temp_file, index=False)
                        fcntl.flock(temp_file.fileno(), fcntl.LOCK_UN)
                        break
                    except IOError:
                        if attempt == max_retries - 1:
                            # Final attempt failed, raise error
                            raise
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
                # Atomic replacement
                temp_path.replace(file_path)
                
            except Exception:
                # Clean up temp file if anything goes wrong
                temp_path.unlink(missing_ok=True)
                raise
            finally:
                # Ensure temp file is cleaned up even if replace succeeded
                # (replace moves the file, so this will typically be a no-op)
                temp_path.unlink(missing_ok=True)
    
    def update_experiment_status(self, experiment_name: str, status_update: Dict[str, Any]) -> None:
        """
        Update experiment status in master CSV.
        
        Args:
            experiment_name: Name of experiment to update
            status_update: Dictionary with fields to update
        """
        status_update['experiment_name'] = experiment_name
        status_update['updated_at'] = datetime.now().isoformat()
        self._append_to_master_csv(status_update)
    
    def get_experiment_history(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get experiment history from master CSV.
        
        Args:
            experiment_name: Optional filter by experiment name
            
        Returns:
            DataFrame with experiment history
        """
        try:
            # Use object dtype for consistent reading
            dtype_spec = {
                'experiment_name': 'object',
                'config_path': 'object',
                'timestamp': 'object',
                'status': 'object',
                'steps_trained': 'object',
                'total_train_time': 'object',
                'final_train_loss': 'object',
                'final_val_loss': 'object',
                'best_val_metric': 'object',
                'test_results_summary': 'object',
                'detailed_log_path': 'object',
                'checkpoint_path': 'object',
                'notes': 'object',
                'created_at': 'object',
                'updated_at': 'object'
            }
            df = pd.read_csv(self.master_csv_path, dtype=dtype_spec)
            if experiment_name:
                df = df[df['experiment_name'] == experiment_name]
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()


class ExperimentRun:
    """
    Individual experiment run with detailed JSON logging.
    
    Handles metrics logging, checkpointing coordination, and final result summarization.
    """
    
    def __init__(self, experiment_name: str, log_dir: Path, parent_logger: ExperimentLogger, history_maxlen: int = 5000):
        """
        Initialize ExperimentRun.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for this experiment's detailed logs
            parent_logger: Parent ExperimentLogger instance
            history_maxlen: Maximum metrics to keep in memory (default 5000)
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.parent_logger = parent_logger
        self.history_maxlen = history_maxlen
        
        # Track timing
        self.start_time = time.time()
        self.total_steps = 0
        self._last_tic: float | None = None   # Record last print time 
        
        self.detailed_log_path = log_dir / "detailed_log.json"
        self.metrics_history = {
            'training_metrics': [],
            'validation_metrics': [],
            'test_results': {},
            'experiment_info': {
                'experiment_name': experiment_name,
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
        }
        
        # Setup structured logging
        self.logger = self._setup_structured_logger()
    
    def _setup_structured_logger(self) -> logging.Logger:
        """Setup structured logger for this experiment."""
        logger = logging.getLogger(f"experiment.{self.experiment_name}")
        logger.setLevel(logging.INFO)
    
        # If already configured, reuse to avoid duplicate handlers
        if logger.handlers:
            return logger
    
        # ---------- Common formatter ----------
        full_fmt = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
        # ---------- Experiment-specific handlers ----------
        log_file = self.log_dir / "experiment.log"
    
        exp_file_h = logging.FileHandler(log_file)
        exp_file_h.setFormatter(full_fmt)
    
        exp_console_h = logging.StreamHandler()
        exp_console_h.setFormatter(logging.Formatter("%(message)s"))   # minimal line
    
        logger.addHandler(exp_file_h)
        logger.addHandler(exp_console_h)
        logger.propagate = False       # step-metrics don't bubble up
    
        # ---------- Root logger (for module logs) ----------
        root = logging.getLogger()
        if not root.handlers:          # Configure only once
            # ① Console: full format for module INFO logs
            root_console_h = logging.StreamHandler()
            root_console_h.setFormatter(full_fmt)
            root.addHandler(root_console_h)
    
            # ② File: a separate FileHandler writing to the same file
            root_file_h = logging.FileHandler(log_file)
            root_file_h.setFormatter(full_fmt)
            root.addHandler(root_file_h)
    
            root.setLevel(logging.INFO)
    
        return logger
    
    def _flush_and_trim_history(self) -> None:
        """Flush full history to disk, keep only recent entries in memory."""
        # Save everything to disk first
        self._save_detailed_log()
        
        # Keep half of the configured max (scales with user preference)
        keep_recent = self.history_maxlen // 2
        
        if len(self.metrics_history['training_metrics']) > keep_recent:
            self.metrics_history['training_metrics'] = \
                self.metrics_history['training_metrics'][-keep_recent:]
        
        if len(self.metrics_history['validation_metrics']) > keep_recent:
            self.metrics_history['validation_metrics'] = \
                self.metrics_history['validation_metrics'][-keep_recent:]
        
        self.logger.info(f"Flushed metrics history, keeping {keep_recent} recent entries in memory")

    def _format_metrics(
            self,
            metrics: Dict[str, float],
            keep_keys: Optional[set] = None,
            precision: int = 4
        ) -> str:
            """
            Convert a metrics dict to 'k=v k=v …'; tensors auto .item()
            """
            parts = []
            for k, v in metrics.items():
                if keep_keys and k not in keep_keys:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if isinstance(v, (float, int)):
                    parts.append(f"{k.rstrip('_loss')}={v:.{precision}f}")
                else:
                    parts.append(f"{k}={v}")
            return " ".join(parts)

    def log_step_metrics(self, step: int, train_metrics: Dict[str, float],
                        *, console_freq: int = 100, precision_freq: int = 500) -> None:
        """
        Log training metrics for a step.
        
        Args:
            step: Global step number
            train_metrics: Training metrics dictionary
            console_freq: Print to console every N steps (default: 100)
            precision_freq: Show precision metrics every N steps (default: 500)
        """
        # ---------- Compute s/it ----------
        now = time.time()
        s_it = 0.0 if self._last_tic is None else now - self._last_tic
        self._last_tic = now
        train_metrics = {**train_metrics, 's_per_it': s_it}
    
        # ---------- Select which keys to keep ----------
        core_keys = {k for k in train_metrics if k.endswith('_loss')}
        
        if step % precision_freq == 0:
            core_keys |= {k for k in train_metrics if k.endswith('_precision')}
    
        # ---------- Build training line ----------
        timing_str = f"{s_it:5.2f}s/it"
        train_line = (
            f"Step {step:>7} | {timing_str} | "
            f"{self._format_metrics(train_metrics, keep_keys=core_keys)}"
        )
        
        msg = train_line

        # ---------- Always write to file ----------
        self.logger.debug(msg)          # DEBUG level → file only
    
        # ---------- Throttle console output ----------
        if step % console_freq == 0:
            self.logger.info(msg)       # INFO level → file + console
    
        # ---------- Keep existing JSON/history logic ----------
        # Track total steps
        self.total_steps = max(self.total_steps, step)
        
        # Add to metrics history
        train_entry = {'step': step, 'timestamp': datetime.now().isoformat(), **train_metrics}
        self.metrics_history['training_metrics'].append(train_entry)
        
        # Memory management: flush if getting too large
        if len(self.metrics_history['training_metrics']) >= self.history_maxlen:
            self._flush_and_trim_history()
        else:
            # Regular save for smaller history
            self._save_detailed_log()

    def log_validation_metrics(
        self,
        step: int,
        val_metrics: Dict[str, float],
        *,
        val_type: str,
        processed_batches: int,
        total_batches: Union[int, str],
        elapsed_time_s: float,
        console_freq: int = 100,
        precision: int = 4
    ) -> None:
        """
        Log validation metrics separately from training logs.
        
        Args:
            step: Global step number associated with this validation run
            val_metrics: Validation metrics dictionary
            val_type: "quick" or "full"
            processed_batches: Number of validation batches processed
            total_batches: Total validation batches (or "unknown")
            elapsed_time_s: Total validation wall time in seconds
            console_freq: Console log frequency (mirrors train logging)
            precision: Decimal precision for metric formatting
        """
        # ---------- Select validation keys ----------
        val_core_keys = {k for k in val_metrics if k.endswith('_loss')}
        if 'raw_total_loss' in val_metrics:
            val_core_keys.add('raw_total_loss')

        header = (
            f"Validation ({val_type}) @ step {step} | "
            f"{processed_batches}/{total_batches} batches | "
            f"{elapsed_time_s:.2f}s total"
        )
        metrics_str = self._format_metrics(val_metrics, keep_keys=val_core_keys, precision=precision)
        msg = f"{header} | {metrics_str}".strip()

        self.logger.debug(msg)
        if step % console_freq == 0:
            self.logger.info(msg)

        # Record history with timing metadata
        val_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'val_type': val_type,
            'processed_batches': processed_batches,
            'total_batches': total_batches,
            'elapsed_time_s': elapsed_time_s,
            **val_metrics
        }
        self.metrics_history['validation_metrics'].append(val_entry)

        if len(self.metrics_history['validation_metrics']) >= self.history_maxlen:
            self._flush_and_trim_history()
        else:
            self._save_detailed_log()
    
    def log_test_results(self, test_results: Dict[str, Any]) -> None:
        """
        Log final test results.
        
        Args:
            test_results: Dictionary with test results
        """
        self.metrics_history['test_results'] = {
            'timestamp': datetime.now().isoformat(),
            **test_results
        }
        
        self.logger.info(f"Test Results: {test_results}")
        self._save_detailed_log()
    
    def log_message(self, message: str, level: str = "info") -> None:
        """
        Log a custom message.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        getattr(self.logger, level.lower())(message)
    
    def _save_detailed_log(self) -> None:
        """Save detailed metrics to JSON file."""
        with open(self.detailed_log_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def _get_best_val_metric(self, metric_name: str = 'loss', minimize: bool = True) -> Optional[float]:
        """
        Extract best validation metric from history.
        
        Args:
            metric_name: Name of metric to extract (e.g., 'loss', 'accuracy')
            minimize: Whether lower values are better (True for loss, False for accuracy)
            
        Returns:
            Best metric value or None if not found
        """
        if not self.metrics_history['validation_metrics']:
            return None
        
        values = [entry.get(metric_name) for entry in self.metrics_history['validation_metrics']]
        values = [v for v in values if v is not None]
        
        if not values:
            return None
        
        return min(values) if minimize else max(values)
    
    def _create_test_results_summary(self, test_results: Dict[str, Any]) -> str:
        """
        Create a proper summary of test results for CSV storage.
        
        Recursively flattens nested dictionaries to extract numeric metrics,
        with a reasonable depth limit to prevent unwieldy column names.
        
        Args:
            test_results: Test results dictionary
            
        Returns:
            JSON-formatted summary string
        """
        def flatten_dict(d: Dict[str, Any], prefix: str = "", max_depth: int = 3) -> Dict[str, Union[int, float]]:
            """
            Recursively flatten dictionary to extract numeric values.
            
            Args:
                d: Dictionary to flatten
                prefix: Current key prefix for nested keys
                max_depth: Maximum nesting depth to explore
                
            Returns:
                Flattened dictionary with only numeric values
            """
            summary = {}
                
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, (int, float)):
                    summary[new_key] = value
                elif isinstance(value, dict) and max_depth > 0:
                    # Recursively flatten nested dictionaries
                    nested_summary = flatten_dict(value, new_key, max_depth - 1)
                    summary.update(nested_summary)
                # Skip non-numeric, non-dict values (lists, strings, etc.)
                    
            return summary
        
        # Extract key numeric metrics with recursive flattening (max 3 levels deep)
        summary = flatten_dict(test_results, max_depth=3)
        
        # Convert to JSON string (properly formatted, not truncated)
        try:
            return json.dumps(summary, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback to string representation if JSON serialization fails
            return str(summary)
    
    def finish_experiment(self, status: str = "completed", 
                         final_summary: Optional[Dict[str, Any]] = None,
                         notes: str = "") -> None:
        """
        Finish experiment and update master CSV.
        
        Args:
            status: Final experiment status
            final_summary: Optional summary statistics
            notes: Optional notes about the experiment
        """
        # Calculate total training time
        total_train_time = time.time() - self.start_time
        
        # Update experiment info
        self.metrics_history['experiment_info'].update({
            'status': status,
            'finished_at': datetime.now().isoformat(),
            'total_train_time_seconds': total_train_time,
            'final_summary': final_summary or {}
        })
        
        # Save final detailed log
        self._save_detailed_log()
        
        # Extract summary for master CSV
        summary_update = {
            'status': status, 
            'notes': notes,
            'total_train_time': f"{total_train_time:.2f}s",
            'steps_trained': self.total_steps
        }
        
        # Extract final training metrics
        if self.metrics_history['training_metrics']:
            last_train = self.metrics_history['training_metrics'][-1]
            summary_update['final_train_loss'] = last_train.get('total_loss')
        
        # Extract final validation metrics and best validation metric
        if self.metrics_history['validation_metrics']:
            last_val = self.metrics_history['validation_metrics'][-1]
            summary_update['final_val_loss'] = last_val.get('raw_total_loss')
            
            best_raw = self._get_best_val_metric('raw_total_loss', minimize=True)
            if best_raw is not None:
                summary_update['best_val_metric'] = f"raw_total_loss={best_raw:.4f}"
        
        # Handle test results with proper JSON serialization
        if self.metrics_history['test_results']:
            test_summary = self._create_test_results_summary(self.metrics_history['test_results'])
            summary_update['test_results_summary'] = test_summary
        
        # Update master CSV
        self.parent_logger.update_experiment_status(self.experiment_name, summary_update)
        
        self.logger.info(f"Experiment {self.experiment_name} finished with status: {status}")
        self.logger.info(f"Total training time: {total_train_time:.2f}s")
        if 'best_val_metric' in summary_update:
            self.logger.info(f"Best validation metric: {summary_update['best_val_metric']}")


class NullExperimentRun:
    """No-op drop-in replacement when experiment logging is disabled."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_dir = Path(".")

    def _format_metrics(
        self,
        metrics: Dict[str, float],
        keep_keys: Optional[set] = None,
        precision: int = 4
    ) -> str:
        parts = []
        for k, v in metrics.items():
            if keep_keys and k not in keep_keys:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (float, int)):
                parts.append(f"{k.rstrip('_loss')}={v:.{precision}f}")
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def log_step_metrics(self, *args, **kwargs) -> None:
        return

    def log_validation_metrics(self, *args, **kwargs) -> None:
        return

    def log_test_results(self, *args, **kwargs) -> None:
        return

    def log_message(self, *args, **kwargs) -> None:
        return

    def finish_experiment(self, *args, **kwargs) -> None:
        return


def create_experiment_logger(logs_root: Union[str, Path] = "outputs/logs") -> ExperimentLogger:
    """
    Convenience function to create an ExperimentLogger.
    
    Args:
        logs_root: Root directory for logging outputs
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(logs_root)


def load_experiment_results(experiment_name: str, logs_root: Union[str, Path] = "outputs/logs") -> Dict[str, Any]:
    """
    Load detailed results for a specific experiment.
    
    Args:
        experiment_name: Name of experiment to load
        logs_root: Root directory for logging outputs
        
    Returns:
        Dictionary with detailed experiment results
    """
    detailed_log_path = Path(logs_root) / experiment_name / "detailed_log.json"
    
    if not detailed_log_path.exists():
        raise FileNotFoundError(f"Detailed log not found for experiment: {experiment_name}")
    
    with open(detailed_log_path, 'r') as f:
        return json.load(f)

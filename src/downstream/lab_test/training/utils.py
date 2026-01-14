# src/downstream/lab_test/training/utils.py

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Task ID to readable name mapping
TASK_NAMES = {
    # Lipid panel
    "000000800008": "LDL",
    "000000800005": "HDL", 
    "000000800006": "TG",
    
    # Liver panel
    "000000800026": "AST",
    "000000800027": "ALT",
    "000000800029": "γ-GTP",
    
    # Glycaemic panel
    "000000800424": "HbA1c",
    "000000800155": "Glucose",
    
    # BP/Obesity panel
    "000000200006": "BMI",
    "000000200009": "Waist",
    "000000500009": "SBP",
    "000000500010": "DBP",
    
    # Risk flags
    "000002800010": "Med1",
    "000002800013": "Med2", 
    "000002800016": "Med3",
    "000003100001": "Smoking"
}

# Panel groupings
PANELS = {
    "Lipid Panel": ["000000800008", "000000800005", "000000800006"],
    "Liver Panel": ["000000800026", "000000800027", "000000800029"],
    "Glycaemic Panel": ["000000800424", "000000800155"],
    "BP/Obesity Panel": ["000000200006", "000000200009", "000000500009", "000000500010"],
    "Risk Flags": ["000002800010", "000002800013", "000002800016", "000003100001"]
}

class Logger:
    """
    Simple logger for downstream training.
    
    Handles console and file logging for training progress and metrics.
    """
    
    def __init__(self, output_dir: Union[str, Path], experiment_name: str):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create log file
        self.log_file_path = self.output_dir / "training.log"
        self.metrics_file_path = self.output_dir / "metrics.jsonl"
        
        # Initialize log file
        with open(self.log_file_path, "w") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Training log started for experiment: {experiment_name}\n")
    
    def log_message(self, message: str) -> None:
        """Log a message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        # Console
        print(formatted_msg)
        
        # File
        with open(self.log_file_path, "a") as f:
            f.write(f"{formatted_msg}\n")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to console and JSON file."""
        
        # Organize metrics
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith('val_')}
        val_metrics = {k: v for k, v in metrics.items() if k.startswith('val_')}
        
        # Console logging - clean and focused
        console_lines = []
        
        # Training metrics (organized by panels)
        if train_metrics:
            formatted_lines = self._format_training_metrics(train_metrics)
            console_lines.append(f"Epoch {epoch} | {formatted_lines[0]}")  # Global metrics
            console_lines.extend(formatted_lines[1:])  # Panel details
        
        # Validation metrics - primary + panel summary
        if val_metrics:
            # Primary metrics
            macro_auroc = val_metrics.get('val_macro_auroc', 0.0)
            mean_mae = val_metrics.get('val_mean_mae', 0.0)
            
            primary_line = f"Epoch {epoch} | macro_auroc: {macro_auroc:.4f} | mean_mae: {mean_mae:.4f}"
            console_lines.append(primary_line)
            
            # Panel-level summary
            panel_summary = self._compute_panel_summary(val_metrics)
            for panel_line in panel_summary:
                console_lines.append(f"  {panel_line}")
        
        # Print to console
        for line in console_lines:
            print(line)
        
        # JSON file logging - keep ALL metrics for detailed analysis
        metrics_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.metrics_file_path, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")

    def log_step(self, epoch: int, step: int, total_steps: Optional[int], metrics: Dict[str, float]) -> None:
        """Lightweight step logging for training progress."""
        parts = []
        loss = metrics.get('total_loss')
        lr = metrics.get('learning_rate')
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")

        if total_steps:
            header = f"Epoch {epoch} Step {step}/{total_steps}"
        else:
            header = f"Epoch {epoch} Step {step}"

        if parts:
            message = f"{header} | {' '.join(parts)}"
        else:
            message = header

        self.log_message(message)
    
    def _compute_panel_summary(self, val_metrics: Dict[str, float]) -> List[str]:
        """Compute panel-level MAE and AUROC averages for console display."""
        
        # Define panel structure (based on your config)
        panels = {
            'lipid': {'has_regression': True, 'has_binary': True},
            'liver': {'has_regression': True, 'has_binary': True},
            'glycaemic': {'has_regression': True, 'has_binary': True},
            'bp_obesity': {'has_regression': True, 'has_binary': True},
            'risk_flags': {'has_regression': False, 'has_binary': True}  # Binary only
        }
        
        panel_lines = []
        
        for panel_name, panel_info in panels.items():
            # Collect MAE values for this panel (regression tasks)
            panel_maes = []
            if panel_info['has_regression']:
                for metric_name, value in val_metrics.items():
                    if (panel_name in metric_name and 'mae' in metric_name and 
                        not metric_name.endswith('_flag_mae')):  # Exclude flag MAEs
                        panel_maes.append(value)
            
            # Collect AUROC values for this panel (binary tasks) 
            panel_aurocs = []
            if panel_info['has_binary']:
                for metric_name, value in val_metrics.items():
                    if panel_name in metric_name and 'auroc' in metric_name:
                        panel_aurocs.append(value)
            
            # Format panel line
            panel_parts = []
            if panel_maes:
                avg_mae = sum(panel_maes) / len(panel_maes)
                panel_parts.append(f"MAE={avg_mae:.3f}")
            if panel_aurocs:
                avg_auroc = sum(panel_aurocs) / len(panel_aurocs)
                panel_parts.append(f"AUROC={avg_auroc:.3f}")
            
            if panel_parts:
                panel_display_name = panel_name.replace('_', '/').title()  # bp_obesity -> Bp/Obesity
                panel_lines.append(f"{panel_display_name}: {' '.join(panel_parts)}")
        
        return panel_lines

    def _format_training_metrics(self, train_metrics: Dict[str, float]) -> List[str]:
        """Format training metrics organized by medical panels."""
        lines = []
        
        # Global metrics first
        total_loss = train_metrics.get('total_loss', 0.0)
        lr = train_metrics.get('learning_rate', 0.0)
        num_batches = train_metrics.get('num_batches', 0.0)
        lines.append(f"total_loss: {total_loss:.2f} | lr: {lr:.6f} | batches: {num_batches:.0f}")
        
        # Organize by panels
        for panel_name, task_ids in PANELS.items():
            panel_tasks = []
            
            for task_id in task_ids:
                task_name = TASK_NAMES.get(task_id, task_id)
                
                # Main task (regression or primary binary)
                main_loss = train_metrics.get(f"{task_id}_loss")
                main_weight = train_metrics.get(f"{task_id}_weight")
                
                # Flag task (if exists)
                flag_loss = train_metrics.get(f"{task_id}_flag_loss")
                flag_weight = train_metrics.get(f"{task_id}_flag_weight")
                
                task_parts = []
                if main_loss is not None:
                    task_parts.append(f"loss={main_loss:.2f}")
                if main_weight is not None:
                    task_parts.append(f"wt={main_weight:.3f}")
                
                flag_parts = []
                if flag_loss is not None:
                    flag_parts.append(f"loss={flag_loss:.2f}")
                if flag_weight is not None:
                    flag_parts.append(f"wt={flag_weight:.3f}")
                
                # Format task line
                task_line = f"{task_name}: {', '.join(task_parts)}"
                if flag_parts:
                    task_line += f" | flag: {', '.join(flag_parts)}"
                
                if task_parts or flag_parts:  # Only add if we have data
                    panel_tasks.append(task_line)
            
            if panel_tasks:  # Only add panel if it has tasks
                lines.append(f"  {panel_name}:")
                for task_line in panel_tasks:
                    lines.append(f"    {task_line}")
        
        return lines

    def log_test_results(self, test_metrics: Dict[str, float]) -> None:
        """Log test results organized by medical panels."""
        
        lines = []
        lines.append("Final Test Results:")
        
        # Global metrics
        total_loss = test_metrics.get('test_total_loss', 0.0)
        num_batches = test_metrics.get('test_num_batches', 0.0)
        lines.append(f"  Overall: total_loss={total_loss:.4f}, batches={num_batches:.0f}")
        
        # Organize by panels
        for panel_name, task_ids in PANELS.items():
            panel_lines = []
            
            for task_id in task_ids:
                task_name = TASK_NAMES.get(task_id, task_id)
                
                # Collect metrics for this task
                mae = test_metrics.get(f'test_{task_id}_mae')
                rmse = test_metrics.get(f'test_{task_id}_rmse') 
                r2 = test_metrics.get(f'test_{task_id}_r2')
                pearson = test_metrics.get(f'test_{task_id}_pearson_r')
                auroc = test_metrics.get(f'test_{task_id}_auroc')
                auprc = test_metrics.get(f'test_{task_id}_auprc')
                
                # Format metrics (regression vs binary)
                metrics_parts = []
                if mae is not None:
                    metrics_parts.append(f"MAE={mae:.3f}")
                if rmse is not None:
                    metrics_parts.append(f"RMSE={rmse:.3f}")
                if r2 is not None:
                    metrics_parts.append(f"R²={r2:.3f}")
                if pearson is not None:
                    metrics_parts.append(f"r={pearson:.3f}")
                if auroc is not None:
                    metrics_parts.append(f"AUROC={auroc:.3f}")
                if auprc is not None:
                    metrics_parts.append(f"AUPRC={auprc:.3f}")
                
                if metrics_parts:
                    panel_lines.append(f"    {task_name}: {', '.join(metrics_parts)}")
            
            if panel_lines:
                lines.append(f"  {panel_name}:")
                lines.extend(panel_lines)
        
        # Log as single message (one timestamp)
        self.log_message("\n".join(lines))


class WandbLogger:
    """
    Optional Weights & Biases logger for downstream training.
    
    Provides simple wandb integration with graceful fallback if not available.
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = False):
        """
        Initialize wandb logger.

        Args:
            config: Experiment configuration
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled
        self.wandb = None
        self.mode: Optional[str] = None
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize wandb
                wandb_config = config.get('wandb', {})
                self.mode = wandb_config.get('mode', 'online')
                if self.mode not in ('online', 'offline'):
                    raise ValueError(f"wandb.mode must be 'online' or 'offline', got {self.mode}")
                experiment_name = config.get('experiment_name', 'lab_test_experiment')
                init_kwargs = {
                    'project': wandb_config.get('project', 'lab-test-downstream'),
                    'name': experiment_name,
                    'id': experiment_name,
                    'config': config,
                    'tags': wandb_config.get('tags', ['downstream', 'lab-test']),
                    'notes': wandb_config.get('notes', ''),
                    'group': wandb_config.get('group', 'downstream'),
                    'entity': wandb_config.get('entity', None)
                }
                if self.mode == 'offline':
                    init_kwargs['mode'] = 'offline'
                self.wandb.init(**init_kwargs)
                
                print("Wandb logging initialized")
                
            except ImportError:
                print("Warning: wandb not available, disabling wandb logging")
                self.enabled = False
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                self.enabled = False

    def _format_medical_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics using medical task names and panel organization.
        
        Converts raw metric names to hierarchical medical names for WandB grouping.
        
        Args:
            metrics: Raw metrics dictionary
            
        Returns:
            Formatted metrics dictionary with medical names and panel grouping
        """
        formatted_metrics = {}
        
        for key, value in metrics.items():
            key = str(key)  # Ensure string type
            
            # Handle non-medical metrics (pass through unchanged)
            if not any(task_id in key for task_id in TASK_NAMES.keys()):
                formatted_metrics[key] = value
                continue
            
            # Find the task ID in the key
            task_id = next((tid for tid in TASK_NAMES.keys() if tid in key), None)
            if task_id is None:
                formatted_metrics[key] = value
                continue
            
            # Get medical name and panel
            task_name = TASK_NAMES[task_id]
            panel_name = None
            for panel, tasks in PANELS.items():
                if task_id in tasks:
                    # Fix: Replace both spaces AND slashes to avoid unwanted WandB hierarchy
                    panel_name = panel.lower().replace(' ', '_').replace('/', '_')
                    break
            
            if panel_name is None:
                # Task not in any panel, use task name only (first occurrence only)
                new_key = key.replace(task_id, task_name, 1)
                formatted_metrics[new_key] = value
                continue
            
            # Create hierarchical key: prefix/panel/task_suffix
            idx = key.find(task_id)
            prefix = key[:idx]
            suffix = key[idx + len(task_id):]
            
            # Build hierarchical key
            if prefix:
                if suffix:
                    new_key = f"{prefix}{panel_name}/{task_name}{suffix}"
                else:
                    new_key = f"{prefix}{panel_name}/{task_name}"
            else:
                if suffix:
                    new_key = f"{panel_name}/{task_name}{suffix}"
                else:
                    new_key = f"{panel_name}/{task_name}"
            
            formatted_metrics[new_key] = value
        
        return formatted_metrics
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb with medical formatting."""
        if self.enabled and self.wandb:
            try:
                # Format medical metrics for better WandB organization
                formatted_metrics = self._format_medical_metrics(metrics)
                self.wandb.log(formatted_metrics, step=step)
            except Exception as e:
                print(f"Warning: wandb logging failed: {e}")
    
    def log_checkpoint(self, checkpoint_path: str, step: int, is_best: bool = False) -> None:
        """Log checkpoint information to wandb."""
        if self.enabled and self.wandb:
            try:
                checkpoint_info = {
                    'checkpoint_path': checkpoint_path,
                    'checkpoint_step': step,
                    'is_best': is_best
                }
                self.wandb.log(checkpoint_info, step=step)
            except Exception as e:
                print(f"Warning: wandb checkpoint logging failed: {e}")
    
    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled and self.wandb:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"Warning: wandb finish failed: {e}")
    
    def is_enabled(self) -> bool:
        """Check if wandb logging is enabled and working."""
        return self.enabled


class DualMetricEarlyStopping:
    """
    Early stopping based on TWO metrics: macro_auroc (max) and mean_mae (min).
    
    Stops training only when BOTH metrics stop improving for the given patience.
    This ensures we don't stop prematurely when one metric plateaus but the other improves.
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, verbose: bool = True):
        """
        Initialize dual-metric early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Track both metrics separately
        self.best_auroc = float('-inf')
        self.best_mae = float('inf')
        self.auroc_wait = 0
        self.mae_wait = 0
        self.should_stop = False
    
    def __call__(self, macro_auroc: float, mean_mae: float) -> bool:
        """
        Check if training should stop based on both metrics.
        
        Args:
            macro_auroc: Current macro AUROC (higher is better)
            mean_mae: Current mean MAE (lower is better)
            
        Returns:
            True if training should stop (both metrics plateau), False otherwise
        """
        # Check AUROC improvement (higher is better)
        auroc_improved = macro_auroc > self.best_auroc + self.min_delta
        if auroc_improved:
            self.best_auroc = macro_auroc
            self.auroc_wait = 0
        else:
            self.auroc_wait += 1
            
        # Check MAE improvement (lower is better)  
        mae_improved = mean_mae < self.best_mae - self.min_delta
        if mae_improved:
            self.best_mae = mean_mae
            self.mae_wait = 0
        else:
            self.mae_wait += 1
            
        # Stop only if BOTH metrics plateau
        should_stop = (self.auroc_wait >= self.patience and self.mae_wait >= self.patience)
        
        if self.verbose:
            print(f"AUROC: {macro_auroc:.4f} (best: {self.best_auroc:.4f}, wait: {self.auroc_wait}/{self.patience})")
            print(f"MAE: {mean_mae:.4f} (best: {self.best_mae:.4f}, wait: {self.mae_wait}/{self.patience})")
            if should_stop:
                print("Early stopping triggered: both metrics plateaued")
                
        self.should_stop = should_stop
        return should_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.auroc_wait = 0
        self.mae_wait = 0
        self.best_auroc = float('-inf')
        self.best_mae = float('inf')
        self.should_stop = False



def create_experiment_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create experiment directories based on configuration.
    
    Args:
        config: Experiment configuration containing experiment_name
        
    Returns:
        Dictionary mapping directory names to paths
    """
    experiment_name = config['experiment_name']
    
    dirs = {
        'checkpoint_dir': Path(f"outputs/downstream/checkpoints/{experiment_name}"),
        'log_dir': Path(f"outputs/downstream/logs/{experiment_name}"),
        'results_dir': Path(f"outputs/downstream/results/{experiment_name}")
    }
    
    # Create directories
    for dir_name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created {dir_name}: {dir_path}")
    
    return dirs


def setup_reproducibility(seed: int = 42, deterministic: bool = False) -> None:
    """
    Setup reproducibility for training.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    import random
    import numpy as np
    import torch
    import os
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Reproducibility enabled (seed={seed}, deterministic=True)")
    else:
        print(f"Reproducibility enabled (seed={seed}, deterministic=False)")


def unpack_individual_sequences(
    individual_emb: torch.Tensor,  # [B', E_max, D]
    segment_lengths: List[int]     # lengths per individual  
) -> torch.Tensor:
    """
    Unpack individual sequences back to exam-level embeddings.
    
    Reverse operation of prepare_individual_sequences() from SSL model.
    Extracts only valid (non-padded) embeddings from individual sequences.
    
    Args:
        individual_emb: Individual sequences [B', E_max, D]
        segment_lengths: Number of valid exams per individual
        
    Returns:
        Exam-level embeddings [B, D] where B = sum(segment_lengths)
    """
    device = individual_emb.device
    B = sum(segment_lengths)  # Total number of exams
    D = individual_emb.size(-1)
    
    exam_embeddings = torch.zeros(B, D, device=device)
    
    exam_idx = 0
    for i, length in enumerate(segment_lengths):
        # Copy valid embeddings (ignore padding)
        exam_embeddings[exam_idx:exam_idx + length] = individual_emb[i, :length]
        exam_idx += length
    
    return exam_embeddings  # [B, D]

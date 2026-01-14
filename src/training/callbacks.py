# src/training/callbacks.py

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
import numpy as np
from collections import deque

class CriticalTrainingError(Exception):
    """Raised when a non-finite value in a critical signal makes continuing training pointless."""
    pass

class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each training step (after parameter update)."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, trainer, batch: int) -> None:
        """Called at the beginning of each batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when a metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_raw_total_loss',
        min_delta: float = 0.001,
        patience: int = 30000,  # Step-based patience (matches base.yaml)
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize EarlyStopping callback.
        
        Args:
            monitor: Metric name to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of steps with no improvement after which to stop
            mode: 'min' or 'max' - whether lower or higher is better
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        
        self.wait_steps = 0
        self.stopped_step = 0
        self.best = None
        self.last_validation_step = 0
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def on_train_begin(self, trainer) -> None:
        """Reset state at training start."""
        self.wait_steps = 0
        self.stopped_step = 0
        self.last_validation_step = 0
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def on_train_end(self, trainer) -> None:
        """Log if stopped early."""
        if self.stopped_step > 0 and self.verbose:
            trainer.experiment.log_message(
                f"Early stopping triggered at step {self.stopped_step}"
            )
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """Check for early stopping when validation metrics are available."""
        # Only evaluate when validation metrics are present
        if self.monitor not in logs:
            return
            
        current_metric = logs[self.monitor]
        
        if self.best is None or self.is_better(current_metric, self.best):
            self.best = current_metric
            self.wait_steps = 0
            self.last_validation_step = step
            if self.verbose:
                trainer.experiment.log_message(
                    f"EarlyStopping: New best {self.monitor}: {current_metric:.6f} at step {step}"
                )
        else:
            # Calculate steps since last validation
            self.wait_steps += (step - self.last_validation_step)
            self.last_validation_step = step
            
            if self.verbose and self.wait_steps % trainer.config['training']['validation_freq'] == 0:
                trainer.experiment.log_message(
                    f"EarlyStopping: No improvement for {self.wait_steps} steps "
                    f"(patience: {self.patience})"
                )
            
            if self.wait_steps >= self.patience:
                self.stopped_step = step
                trainer.should_stop = True
                if self.verbose:
                    trainer.experiment.log_message(
                        f"EarlyStopping: Stopping at step {step}. "
                        f"Best {self.monitor}: {self.best:.6f}"
                    )
    
    def on_batch_begin(self, trainer, batch: int) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """No action needed at batch end."""
        pass


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints based on metrics.
    """
    
    def __init__(
        self,
        monitor: str = 'val_raw_total_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Initialize ModelCheckpoint callback.
        
        Args:
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' - whether lower or higher is better
            save_best_only: Whether to only save best model
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best = None
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
    
    def on_train_begin(self, trainer) -> None:
        """Initialize best metric."""
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def on_train_end(self, trainer) -> None:
        """No action needed at train end."""
        pass
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """Save checkpoint if appropriate."""
        # Only act when validation metrics are available
        if self.monitor not in logs:
            return
            
        current_metric = logs[self.monitor]
        
        if self.save_best_only:
            # Save only if this is the best model
            if self.best is None or self.is_better(current_metric, self.best):
                self.best = current_metric
                best_path = os.path.join(trainer.dirs['checkpoint_dir'], 'best_model.pt')
                trainer.save_checkpoint(best_path)
                if self.verbose:
                    trainer.experiment.log_message(
                        f"Best model saved at step {step}: {self.monitor}={current_metric:.6f}"
                    )
        else:
            # Save based on config save_freq (handled by trainer, not callback)
            pass
    
    def on_batch_begin(self, trainer, batch: int) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """No action needed at batch end."""
        pass


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate during training.
    """
    
    def __init__(
        self,
        scheduler,
        step_on: str = 'step',
        verbose: bool = True
    ):
        """
        Initialize LearningRateScheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
            step_on: When to step scheduler
            verbose: Whether to print LR changes
        """
        self.scheduler = scheduler
        self.step_on = step_on
        self.verbose = verbose
    
    def on_train_begin(self, trainer) -> None:
        """No action needed at train start."""
        pass
    
    def on_train_end(self, trainer) -> None:
        """No action needed at train end."""
        pass
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """Step scheduler if configured for step-based."""
        if self.step_on == 'step':
            # Only step scheduler if optimizer actually ran (check trainer attribute)
            optimizer_stepped = getattr(trainer, '_last_optimizer_stepped', True)
            if optimizer_stepped and hasattr(self.scheduler, 'step'):
                # Some schedulers need metrics (like ReduceLROnPlateau)
                if 'metrics' in self.scheduler.step.__code__.co_varnames:
                    metric = logs.get('val_raw_total_loss')
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()
    
    def on_batch_begin(self, trainer, batch: int) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """Step scheduler if configured for batch."""
        if self.step_on == 'batch':
            # Only step scheduler if optimizer actually ran (check trainer attribute)
            optimizer_stepped = getattr(trainer, '_last_optimizer_stepped', True)
            if optimizer_stepped and hasattr(self.scheduler, 'step'):
                self.scheduler.step()


class GradientMonitor(Callback):
    """
    Callback to monitor gradient statistics during training.
    """
    
    def __init__(
        self,
        monitor_freq: int = 100,
        components: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize GradientMonitor callback.
        
        Args:
            monitor_freq: How often to log gradient stats (batches)
            components: Specific components to monitor
            verbose: Whether to print warnings
        """
        self.monitor_freq = monitor_freq
        self.components = components
        self.verbose = verbose
        
        self.batch_count = 0
        self.gradient_history = deque(maxlen=1000)
    
    def on_train_begin(self, trainer) -> None:
        """Reset counters."""
        self.batch_count = 0
    
    def on_train_end(self, trainer) -> None:
        """No action needed at train end."""
        pass
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """No action needed at step end."""
        pass
    
    def on_batch_begin(self, trainer, batch: int) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """Monitor gradients after batch."""
        self.batch_count += 1
        
        if self.batch_count % self.monitor_freq == 0:
            grad_stats = self._compute_gradient_stats(trainer.model)
            self.gradient_history.append(grad_stats)
            
            # Check for gradient issues
            if grad_stats['total_norm'] > 100:
                if self.verbose:
                    trainer.experiment.log_message(
                        f"Warning: Large gradient norm detected: {grad_stats['total_norm']:.2f}"
                    )
            
            if grad_stats['total_norm'] < 1e-7:
                if self.verbose:
                    trainer.experiment.log_message(
                        f"Warning: Very small gradient norm detected: {grad_stats['total_norm']:.2e}"
                    )
    
    def _compute_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute gradient statistics."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        return {
            'total_norm': total_norm,
            'avg_norm': total_norm / max(param_count, 1),
            'param_count': param_count
        }


class NaNDetector(Callback):
    """
    Detect NaN / ±Inf in critical training signals.
    
    Critical signals
    ----------------
    1. total_loss
    2. Every task-specific *_loss (e.g. mlm_loss, cvr_loss, etc.)
    3. Uncertainty-weighted precisions (σ⁻²) ending with *_precision
    4. Gradients of all trainable parameters
    
    Strategy
    --------
    - Scan every `check_freq` mini-batches  
    - If a *critical* non-finite value is found:
        ① Log an ERROR message  
        ② Save an immediate checkpoint as `nan_ckpt_step{global_step}.pt`  
        ③ If `raise_on_nan` is True → raise `RuntimeError` to halt training  
    - Uses `torch.isfinite` to catch both NaN and ±Inf
    """
    
    def __init__(self, check_freq: int = 10, raise_on_nan: bool = True):
        """
        Initialize NaNDetector callback.
        
        Args:
            check_freq: How often to check gradients (batches). Losses checked every batch.
            raise_on_nan: Whether to raise exception on NaN detection
        """
        self.check_freq = check_freq
        self.raise_on_nan = raise_on_nan
        self.batch_count = 0
    
    def on_train_begin(self, trainer) -> None:
        """Reset counter."""
        self.batch_count = 0
    
    def on_train_end(self, trainer) -> None:
        """No action needed."""
        pass
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """No action needed at step end."""
        pass
    
    def on_batch_begin(self, trainer, batch: int) -> None:
        """No action needed."""
        pass
    
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """Run at the end of every mini-batch."""
        self.batch_count += 1
    
        # 1. Control check frequency (loss checked every batch; grads less often)
        loss_keys = [k for k in logs if k == "total_loss" or k.endswith("_loss")]
        precision_keys = [k for k in logs if k.endswith("_precision")]
    
        # -------- loss / precision: check every batch --------
        for key in loss_keys + precision_keys:
            if key in logs and not torch.isfinite(torch.as_tensor(logs[key])):
                self._handle_nan(f"{key}={logs[key]}", trainer)
    
        # -------- gradients: check with frequency --------
        if self.batch_count % self.check_freq == 0:
            for pname, p in trainer.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    self._handle_nan(f"grad of {pname}", trainer)
                    break  # stop at first hit

    def _handle_nan(self, what: str, trainer) -> None:
        """Record error, dump checkpoint, optionally raise."""
        step = trainer.global_step
        msg = f"[NaNDetector] Non-finite value detected in {what} at step {step}"
        trainer.experiment.log_message(f"ERROR: {msg}")
    
        # Save a step-tagged checkpoint for debugging
        ckpt_name = f"nan_ckpt_step{step}.pt"
        checkpoint_path = os.path.join(trainer.dirs["checkpoint_dir"], ckpt_name)
        trainer.save_checkpoint(checkpoint_path)
    
        if self.raise_on_nan:
            raise CriticalTrainingError(msg)


class CallbackList:
    """
    Container for managing multiple callbacks.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize CallbackList.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
    
    def on_train_begin(self, trainer) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, Any]) -> None:
        """Call on_step_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, logs)
    
    def on_batch_begin(self, trainer, batch: int) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch)
    
    def on_batch_end(self, trainer, batch: int, logs: Dict[str, Any]) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, logs)
    
    def append(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def remove(self, callback: Callback) -> None:
        """Remove a callback from the list."""
        self.callbacks.remove(callback)

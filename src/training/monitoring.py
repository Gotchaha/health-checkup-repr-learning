# src/training/monitoring.py

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable, List, Tuple
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import json
import math
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ArchitectureMonitor:
    """
    Monitor architecture-specific metrics during training.
    
    Tracks attention patterns, fusion weights, loss dynamics, and other
    model-specific behaviors for analysis and debugging.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        history_size: int = 1000
    ):
        """
        Initialize ArchitectureMonitor.
        
        Args:
            config: Monitoring configuration
            output_dir: Directory for saving monitoring outputs
            history_size: Number of steps to keep in memory
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.history_size = history_size
        
        # Metric storage
        self.metrics = defaultdict(lambda: deque(maxlen=history_size))
        
        # Attention weight storage (limited history due to memory)
        self.attention_history = deque(maxlen=10)
        
        # Hooks storage
        self.hooks = []
        self.hook_handles = []
        
        # Step counter
        self.global_step = 0

        # Summary logging
        self._summary_logs: List[Tuple[int, str]] = []
        self.uncertainty_config = config.get('uncertainty_summary', {})

    @staticmethod
    def _safe_append(target_deque, value, name="metric"):
        """
        Append metric safely.
        Accepts two formats:
          • float / int
          • (step:int, value:float)
        """
        # (step, value) format
        if isinstance(value, (tuple, list)) and len(value) == 2:
            step, val = value
            if isinstance(val, (int, float)) and math.isfinite(val):
                target_deque.append((int(step), float(val)))
            else:
                logger.warning(f"[Monitor] {name} invalid {value}; skipped.")
        # pure scalar
        elif isinstance(value, (int, float)) and math.isfinite(value):
            target_deque.append(float(value))
        else:
            logger.warning(f"[Monitor] {name} invalid {value}; skipped.")
    
    def register_model_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on model components for monitoring.
        
        Args:
            model: The MedicalSSLModel to monitor
        """
        # Clear existing hooks
        self.remove_hooks()
        
        # BiCrossAttLayer attention + fusion monitoring
        for i, layer in enumerate(model.cross_attention_layers):
            attention_handle = layer.register_forward_hook(
                self._create_attention_hook(f"cross_attention_{i}")
            )
            fusion_handle = layer.register_forward_hook(
                self._create_cross_fusion_hook(f"cross_attention_{i}")
            )
            self.hook_handles.extend([attention_handle, fusion_handle])
        
        # ImportanceWeightedConcat monitoring
        handle = model.importance_concat.register_forward_hook(
            self._create_fusion_weight_hook()
        )
        self.hook_handles.append(handle)
        
        # TextCompressor monitoring
        handle = model.text_compressor.register_forward_hook(
            self._create_compression_hook()
        )
        self.hook_handles.append(handle)
    
    def _create_attention_hook(self, layer_name: str) -> Callable:
        """Create hook for monitoring attention patterns."""
        def hook(module, input, output):
            if hasattr(module, '_last_attn_weights'):
                attn_weights = module._last_attn_weights
                
                # Calculate attention entropy (measure of focus)
                if attn_weights is not None:
                    # Validate tensor shape before processing
                    if not (isinstance(attn_weights, torch.Tensor) and attn_weights.ndim == 4):
                        return # skip, no monitoring
                    # Average over batch and heads
                    avg_weights = attn_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
                    
                    # Calculate entropy for each query position
                    entropy = -(avg_weights * torch.log(avg_weights + 1e-9)).sum(dim=-1)
                    avg_entropy = entropy.mean().item()
                    
                    self._safe_append(self.metrics[f"{layer_name}_entropy"], avg_entropy, f"{layer_name}_entropy")
                    
                    # Store full attention for detailed analysis (limited)
                    attention_freq = self.config.get('attention_save_freq', 100)
                    if self.global_step % attention_freq == 0:
                        self.attention_history.append({
                            'step': self.global_step,
                            'layer': layer_name,
                            'weights': attn_weights.detach().cpu()
                        })
        
        return hook
    
    def _create_fusion_weight_hook(self) -> Callable:
        """Create hook for monitoring fusion weights."""
        def hook(module, input, output):
            # Get current importance weights
            tab_weight, text_weight = module.get_importance_weights()

            self._safe_append(self.metrics['fusion_tab_weight'], tab_weight, 'fusion_tab_weight')
            self._safe_append(self.metrics['fusion_text_weight'], text_weight, 'fusion_text_weight')
            self._safe_append(self.metrics['fusion_ratio'], tab_weight / (text_weight + 1e-6), 'fusion_ratio')
        
        return hook

    def _create_cross_fusion_hook(self, layer_name: str) -> Callable:
        """Log BiCrossAttLayer sample-wise fusion stats when available."""
        def hook(module, input, output):
            stats = getattr(module, '_last_fusion_stats', None)
            if not stats:
                return
            tab_cross = stats.get('tab_cross_mean')
            text_cross = stats.get('text_cross_mean')
            if tab_cross is not None:
                self._safe_append(self.metrics[f"{layer_name}_tab_cross"], tab_cross, f"{layer_name}_tab_cross")
            if text_cross is not None:
                self._safe_append(self.metrics[f"{layer_name}_text_cross"], text_cross, f"{layer_name}_text_cross")

        return hook
    
    def _create_compression_hook(self) -> Callable:
        """Create hook for monitoring text compression."""
        def hook(module, input, output):
            # Monitor residual weight
            if hasattr(module, 'residual_logit'):
                residual_weight = torch.sigmoid(module.residual_logit).item()
                self._safe_append(self.metrics['compression_residual_weight'], residual_weight, 'compression_residual_weight')
        
        return hook
    
    def log_loss_weights(self, loss_precisions: Dict[str, float]) -> None:
        """
        Log current loss combination precisions.
        
        Args:
            loss_precisions: Dictionary of loss precisions (task importance weights)
        """
        for name, precision in loss_precisions.items():
            self._safe_append(self.metrics[f'loss_precision_{name}'], precision, f'loss_precision_{name}')
        
        # Calculate relative importance
        total_precision = sum(loss_precisions.values())
        for name, precision in loss_precisions.items():
            self._safe_append(self.metrics[f'loss_precision_{name}_relative'], precision / total_precision, f'loss_precision_{name}_relative')
    
    def log_gradient_stats(self, model: nn.Module) -> None:
        """
        Log gradient statistics for different model components.
        
        Args:
            model: Model to analyze gradients
        """
        grad_stats = self._compute_gradient_stats(model)
        
        for component, stats in grad_stats.items():
            for stat_name, value in stats.items():
                self._safe_append(self.metrics[f'grad_{component}_{stat_name}'], (self.global_step, value), f'grad_{component}_{stat_name}')
    
    def _compute_gradient_stats(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Compute gradient statistics for model components."""
        components = {
            'cross_attention': model.cross_attention_layers,
            'unified_layers': model.unified_layers,
            'ind_causal': model.ind_causal_transformer,
            'text_compressor': model.text_compressor,
            'importance_concat': model.importance_concat
        }
        
        stats = {}
        
        for name, component in components.items():
            grads = []
            for param in component.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            
            if grads:
                all_grads = torch.cat(grads)
                stats[name] = {
                    'norm': all_grads.norm().item(),
                    'mean': all_grads.mean().item(),
                    'std': all_grads.std().item(),
                    'max': all_grads.abs().max().item()
                }
            else:
                stats[name] = {
                    'norm': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0
                }
        
        return stats
    
    def log_individual_losses(self, losses: Dict[str, float]) -> None:
        """
        Log individual loss values.
        
        Args:
            losses: Dictionary of loss values
        """
        for name, value in losses.items():
            self._safe_append(self.metrics[f'loss_{name}'], value, f'loss_{name}')

    def log_summary(self, summary_txt: str, step: int, experiment_logger=None) -> None:
        """
        Record uncertainty summary with optional console logging.
        
        Args:
            summary_txt: String from loss_combiner.summary()
            step: Current global step
            experiment_logger: Optional experiment logger for console output
        """
        # Store for analysis
        self._summary_logs.append((step, summary_txt))
        
        # Log to console if enabled
        if self.uncertainty_config.get('log_to_console', True) and experiment_logger:
            experiment_logger.log_message(f"Uncertainty Summary @ Step {step}:\n{summary_txt}")
    
    def log_learning_rates(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Log learning rates for different parameter groups.
        
        Args:
            optimizer: Optimizer with parameter groups
        """
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            self._safe_append(self.metrics[f'lr_{group_name}'], group['lr'], f'lr_{group_name}')
    
    def step(self) -> None:
        """Increment global step counter."""
        self.global_step += 1
    
    def save_visualizations(self) -> None:
        """Save monitoring visualizations."""
        if not self.output_dir:
            return
        
        viz_dir = self.output_dir / f'step_{self.global_step}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot loss precision evolution
        self._plot_loss_precisions(viz_dir)
        
        # Plot fusion weight evolution
        self._plot_fusion_weights(viz_dir)
        
        # Plot gradient statistics
        self._plot_gradient_stats(viz_dir)
        
        # Save attention patterns
        self._save_attention_patterns(viz_dir)
    
    def _plot_loss_precisions(self, output_dir: Path) -> None:
        """Plot loss precision evolution."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Absolute precisions
        for loss_name in ['mlm', 'mcm', 'cvr', 'mcc', 'cpc']:
            if f'loss_precision_{loss_name}' in self.metrics:
                values = list(self.metrics[f'loss_precision_{loss_name}'])
                ax1.plot(values, label=loss_name)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Precision (σ⁻²)')
        ax1.set_title('Task Precision Evolution (Absolute)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative precisions
        for loss_name in ['mlm', 'mcm', 'cvr', 'mcc', 'cpc']:
            if f'loss_precision_{loss_name}_relative' in self.metrics:
                values = list(self.metrics[f'loss_precision_{loss_name}_relative'])
                ax2.plot(values, label=loss_name)
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Relative Precision')
        ax2.set_title('Task Precision Evolution (Relative)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_precisions.png', dpi=150)
        plt.close()
    
    def _plot_fusion_weights(self, output_dir: Path) -> None:
        """Plot fusion weight evolution."""
        if 'fusion_tab_weight' not in self.metrics:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Absolute weights
        tab_weights = list(self.metrics['fusion_tab_weight'])
        text_weights = list(self.metrics['fusion_text_weight'])
        
        ax1.plot(tab_weights, label='Tabular', color='blue')
        ax1.plot(text_weights, label='Text', color='red')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Importance Weight')
        ax1.set_title('Fusion Weight Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ratio
        ratios = list(self.metrics['fusion_ratio'])
        ax2.plot(ratios, color='green')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Tab/Text Ratio')
        ax2.set_title('Tabular to Text Importance Ratio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fusion_weights.png', dpi=150)
        plt.close()
    
    def _plot_gradient_stats(self, output_dir: Path) -> None:
        """Plot gradient statistics."""
        components = ['cross_attention', 'unified_layers', 'ind_causal', 
                     'text_compressor', 'importance_concat']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, stat in enumerate(['norm', 'mean', 'std', 'max']):
            ax = axes[i]
            
            for component in components:
                metric_name = f'grad_{component}_{stat}'
                if metric_name in self.metrics:
                    data = list(self.metrics[metric_name])
                    # If the data is in (step, val) format, unpack it; otherwise fall back to index-based x-axis
                    if data and isinstance(data[0], (tuple, list)):
                        xs, ys = zip(*data)
                    else:
                        xs = range(len(data))
                        ys = data
                    ax.plot(xs, ys, label=component)
            
            ax.set_xlabel('Step')
            ax.set_ylabel(stat.capitalize())
            ax.set_title(f'Gradient {stat.capitalize()} by Component')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if stat == 'mean':
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gradient_stats.png', dpi=150)
        plt.close()
    
    def _save_attention_patterns(self, output_dir: Path) -> None:
        """Save attention pattern visualizations."""
        if not self.attention_history:
            return
        
        # Save last few attention patterns
        attn_dir = output_dir / 'attention_patterns'
        attn_dir.mkdir(exist_ok=True)
        
        for i, attn_data in enumerate(list(self.attention_history)[-3:]):
            step = attn_data['step']
            layer = attn_data['layer']
            weights = attn_data['weights']
            
            # Average over batch and heads for visualization
            avg_weights = weights.mean(dim=(0, 1)).numpy()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_weights, cmap='Blues', cbar=True)
            plt.title(f'{layer} - Step {step}')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.tight_layout()
            plt.savefig(attn_dir / f'{layer}_step_{step}.png', dpi=150)
            plt.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of monitored metrics.
        
        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        
        # Current values
        current = {}
        for name, values in self.metrics.items():
            if values:
                last = values[-1]
                # If (step, val) => take val; else keep original scalar
                current[name] = last[1] if isinstance(last, (tuple, list)) else last
        summary['current'] = current
        
        # Window averages (last 1000 steps)
        window_averages = {}
        window_size = 1000
        for name, values in self.metrics.items():
            if len(values) >= window_size:
                # If values are (step, val) tuples, extract val only; otherwise leave as-is
                if values and isinstance(values[0], (tuple, list)):
                    recent = np.asarray([v for _, v in values])[-window_size:]
                else:
                    recent = np.asarray(values)[-window_size:]
                if np.isfinite(recent).any():
                    window_averages[name] = float(np.nanmean(recent))
        summary['window_averages'] = window_averages
        
        # Trends (last 100 steps)
        trends = {}
        for name, values in self.metrics.items():
            if len(values) >= 100:
                if values and isinstance(values[0], (tuple, list)):
                    recent = np.asarray([v for _, v in values])[-100:]
                else:
                    recent = np.asarray(values)[-100:]
                finite = recent[np.isfinite(recent)]
                if finite.size >= 2:
                    trends[name] = {
                        'mean': float(np.nanmean(finite)),
                        'std': float(np.nanstd(finite)),
                        'trend': 'increasing' if finite[-1] > finite[0] else 'decreasing'
                    }
        summary['trends'] = trends
        
        return summary
    
    def save_summary(self, path: Optional[str] = None) -> None:
        """Save monitoring summary to JSON."""
        if path is None and self.output_dir:
            path = self.output_dir / f'monitoring_summary_step_{self.global_step}.json'
        
        if path:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            summary = self.get_summary()
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

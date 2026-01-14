# src/downstream/lab_test/model/heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import math


class LinearHead(nn.Module):
    """
    Pure linear head for independent task prediction (Option A).
    
    Simple linear transformation without task interference.
    Minimal parameters, strictly adheres to linear probe definition.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize pure linear head.
        
        Args:
            in_dim: Input embedding dimension (e.g., 768)
            out_dim: Number of output tasks
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear layer.
        
        Args:
            x: Input embeddings [B, in_dim]
            
        Returns:
            Output predictions [B, out_dim]
        """
        return self.fc(x)
    
    def get_config(self) -> Dict:
        """Get configuration for saving/loading."""
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim
        }
    
    def __repr__(self) -> str:
        return f"LinearHead(in_dim={self.in_dim}, out_dim={self.out_dim})"


class PanelHead(nn.Module):
    """
    Shared linear layer + task-specific output layers (Option B).
    
    Enables intra-panel feature sharing while maintaining task-specific outputs.
    Still qualifies as linear probing since encoder remains frozen.
    """
    
    def __init__(self, in_dim: int, shared_dim: int, tasks: List[str]):
        """
        Initialize panel head with shared projection.
        
        Args:
            in_dim: Input embedding dimension (e.g., 768)
            shared_dim: Shared projection dimension (e.g., 128)
            tasks: List of task names for this panel
        """
        super().__init__()
        self.in_dim = in_dim
        self.shared_dim = shared_dim
        self.tasks = tasks
        
        # Shared projection layer (no bias for cleaner learned representation)
        self.shared = nn.Linear(in_dim, shared_dim, bias=False)
        
        # Task-specific output heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(shared_dim, 1, bias=True) for name in tasks
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared projection and task heads.
        
        Args:
            x: Input embeddings [B, in_dim]
            
        Returns:
            Dictionary mapping task names to predictions [B]
        """
        # Shared projection
        z = self.shared(x)  # [B, shared_dim]
        
        # Task-specific predictions
        outputs = {}
        for name, layer in self.heads.items():
            outputs[name] = layer(z).squeeze(-1)  # [B, 1] -> [B]
        
        return outputs
    
    def get_config(self) -> Dict:
        """Get configuration for saving/loading."""
        return {
            "in_dim": self.in_dim,
            "shared_dim": self.shared_dim,
            "tasks": self.tasks
        }
    
    def __repr__(self) -> str:
        return f"PanelHead(in_dim={self.in_dim}, shared_dim={self.shared_dim}, tasks={len(self.tasks)})"


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weighting.
    
    Implements the uncertainty-based weighting approach:
    L_total = Σ σ_i^{-2} * L_i + log σ_i^2
    
    This automatically balances tasks based on learned uncertainty estimates.
    Supports pos_weights for imbalanced binary classification tasks.
    """
    
    def __init__(self, task_names: List[str], task_types: Dict[str, str], 
                 loss_config: Dict[str, Any], task_metadata: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-task loss with uncertainty weighting.
        
        Args:
            task_names: List of all task names
            task_types: Dict mapping task names to types ('regression' or 'binary')
            loss_config: Loss configuration from training config
            task_metadata: Task metadata containing pos_weights and other settings
        """
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types
        self.loss_config = loss_config
        self.task_metadata = task_metadata
        
        # Extract loss function types from config
        self.regression_loss_type = loss_config.get('regression_loss', 'smooth_l1')
        self.binary_loss_type = loss_config.get('binary_loss', 'bce_with_logits')
        
        # Learnable log-variance parameters (one per task)
        initial_log_var = loss_config.get('initial_log_var', 0.0)
        self.log_vars = nn.Parameter(torch.full((len(task_names),), initial_log_var))
        
        # Create task name to index mapping
        self.task_to_idx = {name: i for i, name in enumerate(task_names)}
        
        # Prepare pos_weights for binary tasks
        self.pos_weights = {}
        for task_name in task_names:
            if (task_name in task_metadata and 
                task_metadata[task_name].get('pos_weight') is not None):
                self.pos_weights[task_name] = task_metadata[task_name]['pos_weight']
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss with uncertainty weighting.
        
        Args:
            predictions: Dict mapping task names to predictions
            targets: Dict mapping task names to target values
            masks: Dict mapping task names to valid sample masks
            
        Returns:
            total_loss: Weighted sum of task losses
            task_losses: Individual task loss values for logging
        """
        task_losses = {}
        total_loss = 0.0
        
        for task_name in self.task_names:
            if task_name not in predictions:
                continue
                
            pred = predictions[task_name]
            target = targets[task_name]
            mask = masks[task_name]
            
            # Skip if no valid samples
            if mask.sum() == 0:
                task_losses[task_name] = 0.0
                continue
            
            # Compute task-specific loss based on configuration
            task_type = self.task_types[task_name]
            if task_type == 'regression':
                if self.regression_loss_type == 'smooth_l1':
                    loss = F.smooth_l1_loss(pred[mask], target[mask], reduction='mean')
                elif self.regression_loss_type == 'mse':
                    loss = F.mse_loss(pred[mask], target[mask], reduction='mean')
                elif self.regression_loss_type == 'mae':
                    loss = F.l1_loss(pred[mask], target[mask], reduction='mean')
                else:
                    raise ValueError(f"Unknown regression loss: {self.regression_loss_type}")
                    
            elif task_type == 'binary':
                if self.binary_loss_type == 'bce_with_logits':
                    # Use pos_weight if available for this task
                    pos_weight = None
                    if task_name in self.pos_weights:
                        pos_weight = torch.tensor(self.pos_weights[task_name], 
                                                device=pred.device, dtype=pred.dtype)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        pred[mask], target[mask], 
                        pos_weight=pos_weight, 
                        reduction='mean'
                    )
                elif self.binary_loss_type == 'bce':
                    # Note: pos_weight not supported in standard BCE, only in BCE with logits
                    loss = F.binary_cross_entropy(torch.sigmoid(pred[mask]), target[mask], reduction='mean')
                else:
                    raise ValueError(f"Unknown binary loss: {self.binary_loss_type}")
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Apply uncertainty weighting
            task_idx = self.task_to_idx[task_name]
            log_var = self.log_vars[task_idx]
            
            # Uncertainty-weighted loss: σ^{-2} * L + log σ^2
            weighted_loss = torch.exp(-log_var) * loss + log_var
            
            total_loss = total_loss + weighted_loss
            task_losses[task_name] = loss.item()  # Store unweighted loss for logging
        
        return total_loss, task_losses
    
    def get_uncertainty_weights(self) -> Dict[str, float]:
        """
        Get current uncertainty weights for each task.
        
        Returns:
            Dict mapping task names to current σ^{-2} weights
        """
        weights = {}
        for i, task_name in enumerate(self.task_names):
            weights[task_name] = torch.exp(-self.log_vars[i]).item()
        return weights
    
    def __repr__(self) -> str:
        return f"MultiTaskLoss(tasks={len(self.task_names)})"


class LabTestPredictionModel(nn.Module):
    """
    Complete lab test prediction model with 5 clinical panels.
    
    Combines Option A (pure linear) and Option B (shared+specific) heads
    based on panel characteristics from configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize lab test prediction model from configuration.
        
        Args:
            config: Complete configuration dictionary
        """
        super().__init__()
        
        # Extract configuration
        model_config = config['model']
        ssl_config = config['ssl_backbone']
        training_config = config['training']
        
        self.encoder_dim = ssl_config['encoder_dim']
        self.panel_configs = model_config['panels']
        self.task_types = self._build_task_types_from_config(model_config['task_types'])
        
        # Create panel heads from configuration
        self.heads = nn.ModuleDict()
        for panel_name, panel_config in self.panel_configs.items():
            if panel_config['head_type'] == 'shared':
                self.heads[panel_name] = PanelHead(
                    in_dim=self.encoder_dim,
                    shared_dim=panel_config['shared_dim'],
                    tasks=panel_config['tasks']
                )
            elif panel_config['head_type'] == 'linear':
                self.heads[panel_name] = LinearHead(
                    in_dim=self.encoder_dim,
                    out_dim=len(panel_config['tasks'])
                )
            else:
                raise ValueError(f"Unknown head_type: {panel_config['head_type']}")
        
        # Get all task names from configuration
        all_tasks = []
        for panel_config in self.panel_configs.values():
            all_tasks.extend(panel_config['tasks'])
        
        # Multi-task loss with uncertainty weighting
        loss_config = training_config['loss']
        task_metadata = config.get('task_metadata', {})
        self.loss_fn = MultiTaskLoss(all_tasks, self.task_types, loss_config, task_metadata)
    
    def _build_task_types_from_config(self, task_types_config: Dict[str, List[str]]) -> Dict[str, str]:
        """Build task type mapping from configuration."""
        task_types = {}
        
        # Map regression tasks
        for task in task_types_config.get('regression', []):
            task_types[task] = 'regression'
            
        # Map binary tasks  
        for task in task_types_config.get('binary', []):
            task_types[task] = 'binary'
            
        return task_types
    
    def forward(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all prediction heads.
        
        Args:
            encoder_output: SSL encoder output [B, encoder_dim]
            
        Returns:
            Dictionary mapping task names to predictions
        """
        all_predictions = {}
        
        for panel_name, head in self.heads.items():
            config = self.panel_configs[panel_name]
            
            if config['head_type'] == 'shared':
                # PanelHead returns dict of task predictions
                panel_preds = head(encoder_output)
                all_predictions.update(panel_preds)
            else:
                # LinearHead returns tensor, need to split by task
                panel_pred = head(encoder_output)  # [B, num_tasks]
                for i, task_name in enumerate(config['tasks']):
                    all_predictions[task_name] = panel_pred[:, i]
        
        return all_predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss with uncertainty weighting.
        
        Args:
            predictions: Model predictions from forward()
            targets: Target values for each task
            masks: Valid sample masks for each task
            
        Returns:
            total_loss: Weighted total loss
            task_losses: Individual task losses for monitoring
        """
        return self.loss_fn(predictions, targets, masks)
    
    def get_uncertainty_weights(self) -> Dict[str, float]:
        """Get current uncertainty weights for monitoring."""
        return self.loss_fn.get_uncertainty_weights()
    
    def get_config(self) -> Dict:
        """Get model configuration for saving/loading."""
        return {
            "encoder_dim": self.encoder_dim,
            "panel_configs": self.panel_configs,
            "task_types": self.task_types
        }
    
    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"LabTestPredictionModel(encoder_dim={self.encoder_dim}, "
                f"total_params={total_params:,}, trainable_params={trainable_params:,})")
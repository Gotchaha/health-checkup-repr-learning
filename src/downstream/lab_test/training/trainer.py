# src/downstream/lab_test/training/trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from src.models import MedicalSSLModel, create_embedders_from_config
from ..model.heads import LabTestPredictionModel
from .utils import Logger, WandbLogger, DualMetricEarlyStopping, create_experiment_dirs, setup_reproducibility, unpack_individual_sequences
from .metrics import compute_task_metrics, compute_aggregate_metrics


class LabTestTrainer:
    """
    Downstream trainer for lab test prediction tasks.
    
    Loads pretrained SSL backbone, freezes it, and trains prediction heads
    for multi-task clinical lab value prediction using linear probing.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """
        Initialize LabTestTrainer.
        
        Args:
            config: Experiment configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            test_dataloader: Test data loader
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup experiment directories and logging
        self.dirs = create_experiment_dirs(config)
        self.logger = Logger(self.dirs['log_dir'], config['experiment_name'])
        
        # Setup wandb with proper config reading
        wandb_config = config.get('wandb', {})
        self.wandb_logger = WandbLogger(config, enabled=wandb_config.get('enabled', False))
        
        # Setup reproducibility
        setup_reproducibility(
            seed=config.get('seed', 42),
            deterministic=config.get('deterministic', False)
        )
        
        # Load SSL backbone and freeze
        self._load_ssl_backbone()
        
        # Create prediction heads
        self.prediction_model = LabTestPredictionModel(config).to(self.device)
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_early_stopping()
        
        # Setup mixed precision
        training_config = config.get('training', {})
        self.use_amp = training_config.get('mixed_precision', {}).get('enabled', True)
        self.scaler = GradScaler() if self.use_amp else None
        self._last_optimizer_stepped = True  # Track if optimizer ran in last step
        logging_config = training_config.get('logging', {})
        self.step_log_freq = logging_config.get('step_log_freq')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_macro_auroc = float('-inf')  # Track best AUROC (higher is better)
        self.best_mean_mae = float('inf')      # Track best MAE (lower is better)
        
        # Load checkpoint if resuming
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.log_message("LabTestTrainer initialized successfully")
    
    def _load_ssl_backbone(self) -> None:
        """Load pretrained SSL model and freeze parameters."""
        ssl_config = self.config['ssl_backbone']
        ssl_checkpoint_path = ssl_config['checkpoint_path']
        
        self.logger.log_message(f"Loading SSL backbone from: {ssl_checkpoint_path}")
        
        # Load SSL checkpoint
        checkpoint = torch.load(ssl_checkpoint_path, map_location=self.device)
        
        # Create embedders (needed for SSL model)
        self.embedders = create_embedders_from_config(
            checkpoint['config']['model']['embedders'],
            device=self.device
        )
        
        # Load embedder states
        embedders_state = checkpoint['embedders_state_dict']
        self.embedders.text.load_state_dict(embedders_state['text'])
        self.embedders.categorical.load_state_dict(embedders_state['categorical'])
        self.embedders.numerical.load_state_dict(embedders_state['numerical'])
        
        # Get vocabulary sizes
        vocab_sizes = self.embedders.get_vocab_sizes()
        
        # Create SSL model
        self.ssl_model = MedicalSSLModel(
            config=checkpoint['config']['model'],
            text_vocab_size=vocab_sizes['text'],
            cat_vocab_size=vocab_sizes['categorical'],
            device=self.device
        ).to(self.device)
        
        # Load SSL model state
        self.ssl_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze SSL backbone parameters
        if ssl_config.get('freeze_encoder', True):
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            for param in self.embedders.parameters():
                param.requires_grad = False
            
            self.logger.log_message("SSL backbone frozen for linear probing")
        else:
            self.logger.log_message("SSL backbone kept trainable (fine-tuning mode)")
        
        # Set to evaluation mode
        self.ssl_model.eval()
        self.embedders.eval()
        
        self.logger.log_message(f"SSL backbone loaded successfully (step {checkpoint.get('global_step', 'unknown')})")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer for prediction heads only."""
        optimizer_config = self.config['training']['optimizer']
        
        # Only optimize prediction head parameters
        params = list(self.prediction_model.parameters())
        
        if optimizer_config['type'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=optimizer_config['learning_rate'],
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                eps=optimizer_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
        
        self.logger.log_message(f"Optimizer created: {optimizer_config['type']}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            num_epochs = self.config['training']['num_epochs']
            self.warmup_epochs = scheduler_config.get('warmup_epochs', 0) or 0
            cosine_epochs = max(num_epochs - self.warmup_epochs, 1)
            base_lr = self.optimizer.param_groups[0]['lr']
            min_lr_ratio = scheduler_config.get('min_lr_ratio', 0.01)
            eta_min = base_lr * min_lr_ratio
            
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=eta_min
            )
            self.base_lr = base_lr
            self.logger.log_message(
                f"CosineAnnealingLR scheduler created (cosine_epochs={cosine_epochs}, eta_min={eta_min:.2e}, warmup_epochs={self.warmup_epochs})"
            )
        else:
            self.scheduler = None
            self.warmup_epochs = 0
            self.base_lr = self.config['training']['optimizer']['learning_rate']
            self.logger.log_message("No scheduler configured")
    
    def _setup_early_stopping(self) -> None:
        """Setup dual-metric early stopping."""
        es_config = self.config['training'].get('early_stopping', {})
        
        if es_config.get('enabled', True):
            self.early_stopping = DualMetricEarlyStopping(
                patience=es_config.get('patience', 15),
                min_delta=es_config.get('min_delta', 0.001),
                verbose=True
            )
            self.logger.log_message(f"Dual-metric early stopping enabled (patience={es_config.get('patience', 15)})")
        else:
            self.early_stopping = None
            self.logger.log_message("Early stopping disabled")

    def _apply_epoch_scheduler(self, epoch: int) -> None:
        """Apply epoch-level scheduler with optional warmup."""
        if not self.scheduler:
            return
        
        if self.warmup_epochs and epoch < self.warmup_epochs:
            scale = float(epoch + 1) / float(max(1, self.warmup_epochs))
            for group in self.optimizer.param_groups:
                group['lr'] = self.base_lr * scale
        else:
            self.scheduler.step()
    
    def forward_ssl_backbone(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through SSL backbone to get encoder outputs.
        
        Args:
            batch_data: Batch from downstream data loader
            
        Returns:
            Encoder outputs for prediction heads [B, encoder_dim]
        """
        with torch.no_grad():  # SSL backbone is frozen
            # Forward through SSL model
            outputs = self.ssl_model(batch_data, self.embedders.categorical, 
                                   self.embedders.numerical, self.embedders.text)
            
            # Extract final encoder embeddings (post IndCausalTransformer)
            # Unpack from individual sequences [B', E_max, D] back to exam level [B, D]
            encoder_output = unpack_individual_sequences(
                outputs.post_causal_emb,  # [B', E_max, D] 
                outputs.batch_metadata["segment_lengths"]  # List[int]
            )  # Returns [B, D]
            
        return encoder_output
    
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.prediction_model.train()
        
        # Move batch to device
        batch_data = self._move_batch_to_device(batch_data)
        
        # Extract labels and masks from processed batch
        lab_targets = batch_data['lab_targets']  # [B_total, 28]
        lab_masks = batch_data['lab_masks']      # [B_total, 28] 
        
        # Split targets and masks by task
        targets_dict, masks_dict = self._split_labels_by_task(lab_targets, lab_masks)
        
        if self.use_amp:
            with autocast('cuda'):
                # Forward through SSL backbone
                encoder_output = self.forward_ssl_backbone(batch_data)
                
                # Forward through prediction heads
                predictions = self.prediction_model(encoder_output)
                
                # Compute loss
                total_loss, task_losses = self.prediction_model.compute_loss(
                    predictions, targets_dict, masks_dict
                )
            
            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping - per panel to prevent coupling
            clip_config = self.config['training'].get('gradient_clipping', {})
            if clip_config.get('enabled', True):
                self.scaler.unscale_(self.optimizer)
                max_norm = clip_config.get('max_norm', 1.0)
                
                # Clip each panel's gradients separately
                for panel_name, panel_head in self.prediction_model.heads.items():
                    torch.nn.utils.clip_grad_norm_(panel_head.parameters(), max_norm)
                
                # If encoder is not frozen, clip it separately
                if not self.config['ssl_backbone'].get('freeze_encoder', True):
                    torch.nn.utils.clip_grad_norm_(self.ssl_model.parameters(), max_norm)
            
            # Optimizer step
            # Check if optimizer will actually step (AMP overflow detection)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            
            # Store as trainer attribute: if scale unchanged, optimizer ran
            self._last_optimizer_stepped = scale_after == scale_before
        else:
            # Forward through SSL backbone
            encoder_output = self.forward_ssl_backbone(batch_data)
            
            # Forward through prediction heads
            predictions = self.prediction_model(encoder_output)
            
            # Compute loss
            total_loss, task_losses = self.prediction_model.compute_loss(
                predictions, targets_dict, masks_dict
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping - per panel to prevent coupling
            clip_config = self.config['training'].get('gradient_clipping', {})
            if clip_config.get('enabled', True):
                max_norm = clip_config.get('max_norm', 1.0)
                
                # Clip each panel's gradients separately
                for panel_name, panel_head in self.prediction_model.heads.items():
                    torch.nn.utils.clip_grad_norm_(panel_head.parameters(), max_norm)
                
                # If encoder is not frozen, clip it separately
                if not self.config['ssl_backbone'].get('freeze_encoder', True):
                    torch.nn.utils.clip_grad_norm_(self.ssl_model.parameters(), max_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # For non-AMP, optimizer always runs
            self._last_optimizer_stepped = True
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Prepare metrics
        metrics = {
            'total_loss': total_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        metrics.update({f'{task}_loss': loss for task, loss in task_losses.items()})
        
        # Add uncertainty weights
        uncertainty_weights = self.prediction_model.get_uncertainty_weights()
        metrics.update({f'{task}_weight': weight for task, weight in uncertainty_weights.items()})
        
        return metrics
    
    def validate(self, full_eval: bool = False) -> Dict[str, float]:
        """Run validation loop with proper metrics computation."""
        if self.val_dataloader is None:
            return {}
        
        self.prediction_model.eval()
        
        # Collect all predictions and targets
        label_order = self.config['datamodule']['label_processing']['label_order']
        all_predictions = {task: [] for task in label_order}
        all_targets = {task: [] for task in label_order}
        all_masks = {task: [] for task in label_order}
        
        eval_cfg = self.config.get('evaluation', {})
        max_val_batches = eval_cfg.get('max_val_batches_per_epoch')
        if isinstance(max_val_batches, int) and max_val_batches <= 0:
            max_val_batches = None
        
        val_losses = {'val_total_loss': 0.0, 'val_num_batches': 0}
        mode_str = "full" if full_eval else "partial"
        self.logger.log_message(f"Starting validation pass ({mode_str})")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_dataloader, start=1):
                batch_data = self._move_batch_to_device(batch_data)
                
                # Extract labels and masks
                lab_targets = batch_data['lab_targets']  
                lab_masks = batch_data['lab_masks']
                targets_dict, masks_dict = self._split_labels_by_task(lab_targets, lab_masks)
                
                # Forward pass
                encoder_output = self.forward_ssl_backbone(batch_data)
                predictions = self.prediction_model(encoder_output)
                
                # Compute loss (for logging only)
                total_loss, task_losses = self.prediction_model.compute_loss(
                    predictions, targets_dict, masks_dict
                )
                val_losses['val_total_loss'] += total_loss.item()
                val_losses['val_num_batches'] += 1
                
                # Collect predictions/targets for metrics
                for task in label_order:
                    if task in predictions:
                        all_predictions[task].append(predictions[task].cpu())
                        all_targets[task].append(targets_dict[task].cpu())
                        all_masks[task].append(masks_dict[task].cpu())
                
                if (not full_eval) and max_val_batches is not None:
                    if batch_idx >= max_val_batches:
                        self.logger.log_message(f"Validation batch limit reached ({batch_idx})")
                        break
        
        self.logger.log_message("Validation pass completed")
        # Concatenate all batches
        for task in label_order:
            if all_predictions[task]:  # Only if we have data for this task
                all_predictions[task] = torch.cat(all_predictions[task], dim=0)
                all_targets[task] = torch.cat(all_targets[task], dim=0)
                all_masks[task] = torch.cat(all_masks[task], dim=0)
        
        # Compute proper metrics
        task_types = self.prediction_model.task_types
        task_metrics = compute_task_metrics(all_predictions, all_targets, all_masks, task_types)
        aggregate_metrics = compute_aggregate_metrics(task_metrics, task_types)
        
        # Average losses
        if val_losses['val_num_batches'] > 0:
            val_losses['val_total_loss'] /= val_losses['val_num_batches']
        
        # Combine all metrics
        all_metrics = {**val_losses}
        # Add val_ prefix to task and aggregate metrics
        for key, value in task_metrics.items():
            all_metrics[f'val_{key}'] = value
        for key, value in aggregate_metrics.items():
            all_metrics[f'val_{key}'] = value
        
        return all_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {'total_loss': 0.0, 'num_batches': 0}
        epoch_display = self.current_epoch + 1
        total_batches = None
        
        for batch_idx, batch_data in enumerate(self.train_dataloader, start=1):
            step_metrics = self.train_step(batch_data)
            self.global_step += 1
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            epoch_metrics['num_batches'] += 1
            
            # Optional step logging
            if self.step_log_freq and self.step_log_freq > 0:
                if batch_idx % self.step_log_freq == 0:
                    self.logger.log_step(epoch_display, batch_idx, total_batches, step_metrics)
            
            # Log to wandb periodically
            if self.wandb_logger.is_enabled() and self.global_step % 50 == 0:
                log_dict = {f'train/{k}': v for k, v in step_metrics.items()}
                log_dict['train/epoch'] = self.current_epoch
                log_dict['train/global_step'] = self.global_step
                self.wandb_logger.log(log_dict, step=self.global_step)
        
        # Average epoch metrics
        num_batches = epoch_metrics['num_batches']
        if num_batches > 0:
            for key in epoch_metrics:
                if key != 'num_batches':
                    epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.log_message("Starting downstream training...")
        
        num_epochs = self.config['training']['num_epochs']
        eval_cfg = self.config.get('evaluation', {})
        full_every = eval_cfg.get('full_val_every_n_epochs')
        if isinstance(full_every, int) and full_every <= 0:
            full_every = None
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_display = epoch + 1
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            is_full_eval = False
            if full_every is not None and full_every > 0:
                if (epoch_display % full_every) == 0:
                    is_full_eval = True
            val_metrics = self.validate(full_eval=is_full_eval)
            
            # Combined metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['val_full_eval'] = float(is_full_eval)
            
            # Log epoch metrics
            self.logger.log_metrics(epoch_metrics, epoch_display)

            # Scheduler step (epoch-based)
            self._apply_epoch_scheduler(epoch)
            
            # Model selection based on dual metrics
            macro_auroc = val_metrics.get('val_macro_auroc', 0.0) 
            mean_mae = val_metrics.get('val_mean_mae', float('inf'))
            
            # Combined improvement check (either metric improves significantly)
            auroc_improved = macro_auroc > self.best_macro_auroc + 0.001
            mae_improved = mean_mae < self.best_mean_mae - 0.001
            
            if auroc_improved or mae_improved:
                # Update bests
                if auroc_improved:
                    self.best_macro_auroc = macro_auroc
                if mae_improved:
                    self.best_mean_mae = mean_mae
                    
                # Save best model
                best_path = self.save_checkpoint(is_best=True)
                self.logger.log_message(f"New best model: AUROC={macro_auroc:.4f}, MAE={mean_mae:.4f}")
            
            # Early stopping with dual metrics
            if self.early_stopping and self.early_stopping(macro_auroc, mean_mae):
                self.logger.log_message(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Save regular checkpoint
            checkpoint_config = self.config['training']['checkpointing']
            if (epoch + 1) % checkpoint_config['save_every_n_epochs'] == 0:
                self.save_checkpoint()
            
            # Log to wandb
            if self.wandb_logger.is_enabled():
                wandb_dict = {f'epoch/{k}': v for k, v in epoch_metrics.items()}
                wandb_dict['epoch/epoch'] = epoch
                self.wandb_logger.log(wandb_dict, step=self.global_step)
        
        self.logger.log_message("Training completed!")
        self.wandb_logger.finish()
    
    def test(self) -> Dict[str, float]:
        """Run test evaluation with proper metrics computation."""
        if self.test_dataloader is None:
            self.logger.log_message("No test dataloader provided")
            return {}
        
        self.logger.log_message("Running test evaluation...")
        self.prediction_model.eval()
        
        # Collect all predictions and targets
        label_order = self.config['datamodule']['label_processing']['label_order']
        all_predictions = {task: [] for task in label_order}
        all_targets = {task: [] for task in label_order}
        all_masks = {task: [] for task in label_order}
        
        test_losses = {'test_total_loss': 0.0, 'test_num_batches': 0}
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_dataloader, start=1):
                batch_data = self._move_batch_to_device(batch_data)
                
                # Extract labels and masks
                lab_targets = batch_data['lab_targets']
                lab_masks = batch_data['lab_masks']
                targets_dict, masks_dict = self._split_labels_by_task(lab_targets, lab_masks)
                
                # Forward pass
                encoder_output = self.forward_ssl_backbone(batch_data)
                predictions = self.prediction_model(encoder_output)
                
                # Compute loss (for logging only)
                total_loss, task_losses = self.prediction_model.compute_loss(
                    predictions, targets_dict, masks_dict
                )
                test_losses['test_total_loss'] += total_loss.item()
                test_losses['test_num_batches'] += 1
                
                # Collect predictions/targets for metrics
                for task in label_order:
                    if task in predictions:
                        all_predictions[task].append(predictions[task].cpu())
                        all_targets[task].append(targets_dict[task].cpu())
                        all_masks[task].append(masks_dict[task].cpu())
        
        # Concatenate all batches
        for task in label_order:
            if all_predictions[task]:  # Only if we have data for this task
                all_predictions[task] = torch.cat(all_predictions[task], dim=0)
                all_targets[task] = torch.cat(all_targets[task], dim=0)
                all_masks[task] = torch.cat(all_masks[task], dim=0)
        
        # Compute proper metrics
        task_types = self.prediction_model.task_types
        task_metrics = compute_task_metrics(all_predictions, all_targets, all_masks, task_types)
        aggregate_metrics = compute_aggregate_metrics(task_metrics, task_types)
        
        # Average losses
        if test_losses['test_num_batches'] > 0:
            test_losses['test_total_loss'] /= test_losses['test_num_batches']
        
        # Combine all metrics
        all_metrics = {**test_losses}
        # Add test_ prefix to task and aggregate metrics
        for key, value in task_metrics.items():
            all_metrics[f'test_{key}'] = value
        for key, value in aggregate_metrics.items():
            all_metrics[f'test_{key}'] = value
        
        self.logger.log_message("Test evaluation completed")
        return all_metrics
    
    def _split_labels_by_task(self, targets: torch.Tensor, masks: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Split label tensors by task name for loss computation."""
        label_order = self.config['datamodule']['label_processing']['label_order']
        
        targets_dict = {}
        masks_dict = {}
        
        for i, task_name in enumerate(label_order):
            targets_dict[task_name] = targets[:, i]
            masks_dict[task_name] = masks[:, i].bool()
        
        return targets_dict, masks_dict
    
    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device."""
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        value[sub_key] = sub_value.to(self.device)
        return batch_data
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False) -> str:
        """Save checkpoint."""
        if path is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt" 
            if is_best:
                filename = "best_model.pt"
            path = os.path.join(self.dirs['checkpoint_dir'], filename)
        
        checkpoint = {
            # Model state
            'prediction_model_state_dict': self.prediction_model.state_dict(),
            
            # Training components
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            
            # Training state
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_macro_auroc': self.best_macro_auroc,
            'best_mean_mae': self.best_mean_mae,
            
            # Metadata
            'config': self.config,
            'ssl_checkpoint_path': self.config['ssl_backbone']['checkpoint_path']
        }
        
        if self.use_amp and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        # Log checkpoint
        if self.wandb_logger.is_enabled():
            self.wandb_logger.log_checkpoint(path, self.global_step, is_best=is_best)
        
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path."""
        self.logger.log_message(f"Loading checkpoint from: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.prediction_model.load_state_dict(checkpoint['prediction_model_state_dict'])
        
        # Load training components
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_macro_auroc = checkpoint.get('best_macro_auroc', float('-inf'))
        self.best_mean_mae = checkpoint.get('best_mean_mae', float('inf'))
        
        # Load scaler if using AMP
        if self.use_amp and self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.log_message(f"Resumed from checkpoint at epoch {self.current_epoch}, step {self.global_step}")

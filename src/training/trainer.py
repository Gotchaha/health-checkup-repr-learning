# src/training/trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple, List
import time
from pathlib import Path
from tqdm import tqdm

from ..models import (
    HealthExamDataset, PersonBatchSampler, InfinitePersonBatchSampler, collate_exams,
    create_embedders_from_config, MedicalSSLModel, ModelOutputs
)
from .multi_task_loss import MultiTaskLoss
from .heads import MLMHead, MCMHead, CVRHead, MCCHead, CPCHead
from .monitoring import ArchitectureMonitor
from .callbacks import (
    CallbackList, EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, GradientMonitor, NaNDetector,
    CriticalTrainingError
)
from ..utils import (
    ExperimentLogger, setup_reproducibility, 
    create_experiment_dirs, WandbLogger
)


class SSLTrainer:
    """
    Self-supervised learning trainer for medical examination data.
    
    Handles training with all five objectives (MLM, MCM, CVR, MCC, CPC),
    differential learning rates, monitoring, and checkpointing.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """
        Initialize SSLTrainer.
        
        Args:
            config: Experiment configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup logging
        paths_cfg = config.get('paths', {})
        outputs_root = Path(paths_cfg.get('outputs_root', 'outputs'))
        logs_subdir = paths_cfg.get('logs_subdir', 'logs')
        logs_root = outputs_root / logs_subdir

        self.logger = ExperimentLogger(logs_root=logs_root)
        self.experiment = self.logger.start_experiment(config)
        self.dirs = create_experiment_dirs(config)
        
        # Setup wandb if enabled
        self.wandb_logger = WandbLogger(config, enabled=config.get('use_wandb', False))
        
        # Initialize embedders
        self.embedders = create_embedders_from_config(
            config['model']['embedders'], 
            device=self.device
        )
        
        # Get vocabulary sizes for model
        vocab_sizes = self.embedders.get_vocab_sizes()
        
        # Initialize model
        self.model = MedicalSSLModel(
            config=config['model'],
            text_vocab_size=vocab_sizes['text'],
            cat_vocab_size=vocab_sizes['categorical'],
            device=self.device
        ).to(self.device)
        
        # Initialize loss heads
        self.mlm_head = MLMHead(
            d_model=config['model']['d_model'],
            text_vocab_size=vocab_sizes['text'],
            dropout=config['model'].get('head_dropout', 0.1)
        ).to(self.device)
        
        self.mcm_head = MCMHead(
            d_model=config['model']['d_model'],
            cat_vocab_size=vocab_sizes['categorical'],
            dropout=config['model'].get('head_dropout', 0.1)
        ).to(self.device)
        
        self.cvr_head = CVRHead(
            d_model=config['model']['d_model'],
            temperature=config['model'].get('cvr_temperature', 0.1),
            normalize=config['model'].get('cvr_normalize', True)
        ).to(self.device)

        self.mcc_head = MCCHead(
            d_model=config['model']['d_model'],
            temperature=config['model'].get('mcc_temperature', 1.0),
            normalize=config['model'].get('mcc_normalize', True)
        ).to(self.device)
        
        self.cpc_head = CPCHead(
            d_model=config['model']['d_model'],
            proj_dim=config['model'].get('cpc_proj_dim', 128),
            temperature=config['model'].get('cpc_temperature', 0.1),
            min_negatives=config['model'].get('cpc_min_negatives', 1)
        ).to(self.device)
        
        # Initialize multi-task loss
        self.loss_combiner = MultiTaskLoss(
            initial_weights=config['training'].get('initial_loss_weights', None)
        ).to(self.device)
        
        # Setup optimizer with differential learning rates
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup gradient scaler for mixed precision
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.should_stop = False
        self._last_optimizer_stepped = True  # Track if optimizer ran in last step 
        
        # Initialize monitoring
        monitoring_config = config.get('monitoring', {})
        monitoring_enabled = monitoring_config.get('enabled', True)
        if monitoring_enabled:
            try:
                self.monitor = ArchitectureMonitor(
                    config=monitoring_config,
                    output_dir=os.path.join(self.dirs['log_dir'], 'monitoring')
                )
                self.monitor.register_model_hooks(self.model)
                self.monitoring_enabled = True
            except Exception as e:
                self.experiment.log_message(f"Warning: Monitoring initialization failed: {e}")
                self.monitor = None
                self.monitoring_enabled = False
        else:
            self.monitor = None
            self.monitoring_enabled = False
        
        # Initialize callbacks
        self.callbacks = self._setup_callbacks()
        
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
    
    def _setup_callbacks(self) -> CallbackList:
        """Setup training callbacks with error handling."""
        callbacks = []
        
        try:
            # Early stopping (step-based)
            if self.config['training'].get('early_stopping', {}).get('enabled', True):
                es_config = self.config['training'].get('early_stopping', {})
                callbacks.append(EarlyStopping(
                    monitor=es_config.get('monitor', 'val_raw_total_loss'),
                    min_delta=es_config.get('min_delta', 0.001),
                    patience=es_config.get('patience', 30000),  # Step-based default
                    mode=es_config.get('mode', 'min'),
                    verbose=True
                ))
            
            # Model checkpoint (simplified step-based)
            checkpoint_config = self.config['training'].get('checkpoint', {})
            if checkpoint_config.get('enabled', True):
                callbacks.append(ModelCheckpoint(
                    monitor=checkpoint_config.get('monitor', 'val_raw_total_loss'),
                    mode=checkpoint_config.get('mode', 'min'),
                    save_best_only=checkpoint_config.get('save_best_only', True),  # Changed default to True
                    verbose=True
                ))
            
            # Learning rate scheduler callback (step-based)
            if self.scheduler:
                step_on = self.config['training'].get('scheduler', {}).get('step_on', 'step')
                callbacks.append(LearningRateScheduler(
                    self.scheduler,
                    step_on=step_on,  # Use config value, default 'step'
                    verbose=True
                ))
            
            # Gradient monitor
            if self.config['training'].get('monitor_gradients', True):
                callbacks.append(GradientMonitor(
                    monitor_freq=100,
                    verbose=True
                ))
            
            # NaN detector
            callbacks.append(NaNDetector(
                check_freq=10,
                raise_on_nan=self.config['training'].get('raise_on_nan', True)
            ))
            
        except Exception as e:
            self.experiment.log_message(f"Warning: Callback setup partially failed: {e}")
        
        return CallbackList(callbacks)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with differential learning rates."""
        # Get parameter groups
        model_params = self.model.get_trainable_params()
        
        # Build optimizer groups
        optimizer_groups = [
            # Embedders - lower LR
            {
                'params': self.embedders.text.parameters(),
                'lr': self.config['training']['embedder_lr'],
                'name': 'text_embedder'
            },
            {
                'params': self.embedders.categorical.parameters(),
                'lr': self.config['training']['embedder_lr'],
                'name': 'categorical_embedder'
            },
            {
                'params': self.embedders.numerical.parameters(),
                'lr': self.config['training']['embedder_lr'],
                'name': 'numerical_embedder'
            },
            
            # Model components - main LR
            {
                'params': model_params['tab_embedder'],
                'lr': self.config['training']['learning_rate'],
                'name': 'tab_embedder'
            },
            {
                'params': model_params['cross_attention'],
                'lr': self.config['training']['learning_rate'],
                'name': 'cross_attention'
            },
            {
                'params': model_params['text_compressor'],
                'lr': self.config['training']['learning_rate'],
                'name': 'text_compressor'
            },
            {
                'params': model_params['importance_concat'],
                'lr': self.config['training']['learning_rate'],
                'name': 'importance_concat'
            },
            {
                'params': model_params['unified_layers'],
                'lr': self.config['training']['learning_rate'],
                'name': 'unified_layers'
            },
            {
                'params': model_params['time_embedding'],
                'lr': self.config['training']['learning_rate'],
                'name': 'time_embedding'
            },
            {
                'params': model_params['ind_causal'],
                'lr': self.config['training']['learning_rate'],
                'name': 'ind_causal'
            },
            
            # Loss heads - main LR
            {
                'params': self.mlm_head.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'mlm_head'
            },
            {
                'params': self.mcm_head.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'mcm_head'
            },
            {
                'params': self.cvr_head.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'cvr_head'
            },
            {
                'params': self.mcc_head.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'mcc_head'
            },
            {
                'params': self.cpc_head.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'cpc_head'
            },
            
            # Loss weights - higher LR
            {
                'params': self.loss_combiner.parameters(),
                'lr': self.config['training'].get('loss_weight_lr', 1e-3),
                'weight_decay': 0.0,  # CRITICAL: Exclude log_vars from L2 regularization
                'name': 'loss_weights'
            }
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=self.config['training'].get('weight_decay', 0.01),
            betas=self.config['training'].get('betas', (0.9, 0.999))
        )
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', {})

        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine')

        #--- no min_lr implementation ---#
        # if scheduler_type == 'cosine':
        #     from transformers import get_cosine_schedule_with_warmup
            
        #     # Use step-based training parameters
        #     num_training_steps = self.config['training']['max_steps']
        #     num_warmup_steps = scheduler_config.get('warmup_steps', 8000)  # Step-based default
            
        #     scheduler = get_cosine_schedule_with_warmup(
        #         self.optimizer,
        #         num_warmup_steps=num_warmup_steps,
        #         num_training_steps=num_training_steps
        #     )

        if scheduler_type == 'cosine':
            import math
            from torch.optim.lr_scheduler import LambdaLR
            
            # Use step-based training parameters
            num_training_steps = self.config['training']['max_steps']
            num_warmup_steps = scheduler_config.get('warmup_steps', 8000)  # Step-based default
            min_lr_ratio = scheduler_config.get('min_lr_ratio', 0.01)  # Default 1% floor
            
            # Record base LRs for each parameter group
            base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            
            def lr_lambda_for_group(base_lr):
                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    
                    progress = (current_step - num_warmup_steps) / float(
                        max(1, num_training_steps - num_warmup_steps)
                    )
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    
                    # Apply floor: min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                return lr_lambda
            
            # Create one lambda function per parameter group
            lambda_functions = [lr_lambda_for_group(base_lr) for base_lr in base_lrs]
            scheduler = LambdaLR(self.optimizer, lambda_functions)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return scheduler
    
    def _train_step(self, batch_data: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        """
        Execute a single training step (one batch).
        
        Args:
            batch_data: Batch data from dataloader
            batch_idx: Batch index
            
        Returns:
            Dictionary of training metrics for this step
        """
        self.model.train()
        self.embedders.train()
        
        try:
            self.callbacks.on_batch_begin(self, batch_idx)
        except Exception as e:
            self.experiment.log_message(f"Callback error at batch begin: {e}")

        if batch_idx % 100 == 0 and 'held_out_cells_count' in batch_data:
            total_held_out = batch_data.get('held_out_cells_count', 0)
            if total_held_out > 0:
                codes = batch_data.get('held_out_codes_in_batch', [])
                self.experiment.log_message(
                    f"Batch {batch_idx}: {total_held_out} held-out cells "
                    f"(codes: {codes[:5]}{'...' if len(codes) > 5 else ''})"
                )
        
        # Move batch to device
        batch_data = self._move_batch_to_device(batch_data)
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=self.use_amp):
            # Get model outputs
            outputs = self.model(batch_data, self.embedders.categorical, 
                               self.embedders.numerical, self.embedders.text)
            
            # Compute individual losses (using existing methods)
            mlm_loss = self._compute_mlm_loss(outputs, batch_data)
            mcm_loss = self._compute_mcm_loss(outputs, batch_data)
            cvr_loss = self._compute_cvr_loss(outputs, batch_data)
            mcc_loss = self._compute_mcc_loss(outputs, batch_data)
            cpc_loss = self._compute_cpc_loss(outputs, batch_data)
            
            # Combine losses (using correct interface)
            loss_dict = self.loss_combiner(
                mlm_loss, mcm_loss, cvr_loss, mcc_loss, cpc_loss,
                return_dict=True
            )
            total_loss = loss_dict['total_loss']
        
        # Monitoring config reused in backward paths
        monitoring_cfg = self.config.get('monitoring', {})
        grad_stats_freq = monitoring_cfg.get('grad_stats_freq', 500)

        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping (using existing parameter collection pattern)
            if self.config['training'].get('gradient_clip'):
                self.scaler.unscale_(self.optimizer)
                all_params = (
                    list(self.model.parameters()) +
                    list(self.embedders.text.parameters()) +
                    list(self.embedders.categorical.parameters()) +
                    list(self.embedders.numerical.parameters()) +
                    list(self.mlm_head.parameters()) +
                    list(self.mcm_head.parameters()) +
                    list(self.cvr_head.parameters()) +
                    list(self.mcc_head.parameters()) +
                    list(self.cpc_head.parameters()) +
                    list(self.loss_combiner.parameters())
                )
                torch.nn.utils.clip_grad_norm_(
                    all_params, 
                    self.config['training'].get('gradient_clip', 1.0)
                )

            # Gradient statistics monitoring (AMP branch)
            if (self.monitoring_enabled and self.monitor and 
                self.global_step % grad_stats_freq == 0):
                try:
                    self.monitor.log_gradient_stats(self.model)
                except Exception as e:
                    self.experiment.log_message(f"Gradient monitoring error: {e}")

            # Check if optimizer will actually step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            
            # Store as trainer attribute: if scale unchanged, optimizer ran
            self._last_optimizer_stepped = scale_after == scale_before
        else:
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                all_params = (
                    list(self.model.parameters()) +
                    list(self.embedders.text.parameters()) +
                    list(self.embedders.categorical.parameters()) +
                    list(self.embedders.numerical.parameters()) +
                    list(self.mlm_head.parameters()) +
                    list(self.mcm_head.parameters()) +
                    list(self.cvr_head.parameters()) +
                    list(self.mcc_head.parameters()) +
                    list(self.cpc_head.parameters()) +
                    list(self.loss_combiner.parameters())
                )
                torch.nn.utils.clip_grad_norm_(
                    all_params,
                    self.config['training'].get('gradient_clip', 1.0)
                )

            # Gradient statistics monitoring (non-AMP branch)
            if (self.monitoring_enabled and self.monitor and 
                self.global_step % grad_stats_freq == 0):
                try:
                    self.monitor.log_gradient_stats(self.model)
                except Exception as e:
                    self.experiment.log_message(f"Gradient monitoring error: {e}")
            
            self.optimizer.step()
            # For non-AMP, optimizer always runs
            self._last_optimizer_stepped = True
        
        self.optimizer.zero_grad()
        
        # Increment global step
        self.global_step += 1
        
        # Prepare metrics (using loss_dict values)
        step_metrics = {
            'total_loss': total_loss.item(),
            'mlm_loss': mlm_loss.item(),
            'mcm_loss': mcm_loss.item(),
            'cvr_loss': cvr_loss.item(),
            'mcc_loss': mcc_loss.item(),
            'cpc_loss': cpc_loss.item(),
        }
        
        # Add precision metrics from loss_dict
        if 'mlm_precision' in loss_dict:
            step_metrics.update({
                'mlm_precision': loss_dict['mlm_precision'].item(),
                'mcm_precision': loss_dict['mcm_precision'].item(),
                'cvr_precision': loss_dict['cvr_precision'].item(),
                'mcc_precision': loss_dict['mcc_precision'].item(),
                'cpc_precision': loss_dict['cpc_precision'].item(),
            })
        
        # Callback: batch end
        try:
            self.callbacks.on_batch_end(self, batch_idx, step_metrics)
        except CriticalTrainingError as e:
            # Save an emergency checkpoint first, then log a fatal message
            try:
                emergency_path = f"critical_error_step{self.global_step}.pt"
                self.save_checkpoint(os.path.join(self.dirs['checkpoint_dir'], emergency_path))
                self.experiment.log_message("Emergency checkpoint saved before termination")
            except:
                pass  # Don't let checkpoint saving prevent error propagation
            
            self.experiment.log_message(f"CRITICAL ERROR at step {self.global_step}: {e}")
            raise  # Let the outer watchdog or Slurm job catch this
        except Exception as e:
            self.experiment.log_message(f"Non-critical callback error (ignored) at step {self.global_step}: {e}")
        
        # Monitoring integration
        if self.monitoring_enabled and self.monitor:
            try:
                # Log loss precisions every step
                self.monitor.log_loss_weights(self.loss_combiner.get_current_precisions())
                self.monitor.log_individual_losses({
                    'mlm': mlm_loss.item(),
                    'mcm': mcm_loss.item(),
                    'cvr': cvr_loss.item(),
                    'mcc': mcc_loss.item(),
                    'cpc': cpc_loss.item()
                })
                self.monitor.log_learning_rates(self.optimizer)
                    
                # Uncertainty summary logging
                if self.should_log_uncertainty_summary():
                    self.monitor.log_summary(
                        self.loss_combiner.summary(),
                        step=self.global_step,
                        experiment_logger=self.experiment
                    )

                # Sync monitoring step counter
                self.monitor.step()

            except Exception as e:
                self.experiment.log_message(f"Monitoring error: {e}")
        
        return step_metrics
    
    def validate(self, subset: bool = False) -> Dict[str, float]:
        """
        Run validation on subset or full validation set.
        
        Args:
            subset: If True, use val_subset_batches for quick validation
                    If False, use entire validation set
        
        Returns:
            Dictionary of validation metrics
        """
        if not self.val_dataloader:
            return {}
            
        self.model.eval()
        self.embedders.eval()
        
        val_metrics = {
            'mlm_loss': 0.0,
            'mcm_loss': 0.0,
            'cvr_loss': 0.0,
            'mcc_loss': 0.0,
            'cpc_loss': 0.0,
            # Evaluation metric for early stopping: unweighted mean of task losses
            'raw_total_loss': 0.0
        }
        
        max_batches = self.config['training'].get('val_subset_batches', 400) if subset else None
        batch_count = 0
        val_type = "quick" if subset else "full"
        
        val_start = time.time()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch_data = self._move_batch_to_device(batch_data)
                
                # Forward pass
                outputs = self.model(batch_data, self.embedders.categorical,
                                   self.embedders.numerical, self.embedders.text)
                
                # Compute losses (using existing methods)
                mlm_loss = self._compute_mlm_loss(outputs, batch_data)
                mcm_loss = self._compute_mcm_loss(outputs, batch_data)
                cvr_loss = self._compute_cvr_loss(outputs, batch_data)
                mcc_loss = self._compute_mcc_loss(outputs, batch_data)
                cpc_loss = self._compute_cpc_loss(outputs, batch_data)
                
                # Accumulate metrics (validation uses unweighted losses only)
                val_metrics['mlm_loss'] += mlm_loss.item()
                val_metrics['mcm_loss'] += mcm_loss.item()
                val_metrics['cvr_loss'] += cvr_loss.item()
                val_metrics['mcc_loss'] += mcc_loss.item()
                val_metrics['cpc_loss'] += cpc_loss.item()
                raw_total = (
                    mlm_loss + mcm_loss + cvr_loss + mcc_loss + cpc_loss
                ) / 5.0
                val_metrics['raw_total_loss'] += raw_total.item()
                
                batch_count += 1
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= batch_count
        
        elapsed = time.time() - val_start

        # Log validation type and batch count
        total_batches = len(self.val_dataloader) if hasattr(self.val_dataloader, '__len__') else "unknown"
        if batch_count > 0:
            console_freq = self.config['training']['logging']['console_freq']
            self.experiment.log_validation_metrics(
                step=self.global_step,
                val_metrics=val_metrics,
                val_type=val_type,
                processed_batches=batch_count,
                total_batches=total_batches,
                elapsed_time_s=elapsed,
                console_freq=console_freq
            )
        else:
            self.experiment.log_message(
                f"Validation ({val_type}) skipped: 0/{total_batches} batches processed"
            )
        
        # Log summary at validation time (for monitoring integration)
        if not subset and self.monitoring_enabled and self.monitor:
            try:
                self.monitor.log_summary(
                    self.loss_combiner.summary(),
                    step=self.global_step,
                    experiment_logger=self.experiment
                )
            except Exception as e:
                self.experiment.log_message(f"Validation summary error: {e}")
        
        return val_metrics
    
    def _compute_mlm_loss(self, outputs: ModelOutputs, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Compute MLM loss."""
        _, mlm_loss = self.mlm_head(
            outputs.mlm_embeddings,
            batch_data["result_mlm_labels"]
        )
        return mlm_loss
    
    def _compute_mcm_loss(self, outputs: ModelOutputs, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Compute MCM loss."""
        _, mcm_loss = self.mcm_head(
            outputs.mcm_embeddings,
            batch_data["expanded_labels"]["mcm_labels"]
        )
        return mcm_loss
    
    def _compute_cvr_loss(self, outputs: ModelOutputs, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Compute CVR loss."""
        _, cvr_loss = self.cvr_head(
            outputs.cvr_embeddings,
            batch_data["expanded_labels"]["cvr_mask"],
            batch_data["expanded_labels"]["cvr_candidates"],
            batch_data["expanded_labels"]["cvr_labels"]
        )
        return cvr_loss

    def _compute_mcc_loss(self, outputs: ModelOutputs, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Compute MCC loss."""
        _, mcc_loss = self.mcc_head(
            outputs.mcc_embeddings,
            batch_data["expanded_labels"]["mcc_mask"],
            batch_data["expanded_labels"]["mcc_candidates"],
            batch_data["expanded_labels"]["mcc_labels"]
        )
        return mcc_loss
    
    def _compute_cpc_loss(self, outputs: ModelOutputs, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Compute CPC loss."""
        cpc_loss = self.cpc_head(
            outputs.pre_causal_emb,
            outputs.post_causal_emb,
            outputs.individual_attention_mask,
            outputs.segment_lengths
        )
        return cpc_loss

    def should_log_uncertainty_summary(self) -> bool:
        """Check if should log uncertainty summary based on config."""
        if not self.monitoring_enabled or not self.monitor:
            return False
            
        uncertainty_config = self.config.get('monitoring', {}).get('uncertainty_summary', {})
        if not uncertainty_config.get('enabled', True):
            return False
        
        early_steps = uncertainty_config.get('early_steps', 100)
        early_freq = uncertainty_config.get('early_frequency', 10) 
        late_freq = uncertainty_config.get('late_frequency', 200)
        
        if self.global_step <= early_steps:
            return self.global_step % early_freq == 0
        else:
            return self.global_step % late_freq == 0
    
    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device (PRESERVED ORIGINAL NESTED DICT HANDLING)."""
        # Move tensors to device
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Handle nested dictionaries (like expanded_labels)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        value[sub_key] = sub_value.to(self.device)
        
        return batch_data
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            path: Custom path for checkpoint
            is_best: Whether this is the best model
            
        Returns:
            Path to saved checkpoint
        """
        if path is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
            if is_best:
                filename = "best_model.pt"
            path = os.path.join(self.dirs['checkpoint_dir'], filename)

        checkpoint_cfg = self.config['training'].get('checkpoint', {})
        if not checkpoint_cfg.get('enabled', True):
            return path
        
        checkpoint = {
            # Model and embedders
            'model_state_dict': self.model.state_dict(),
            'embedders_state_dict': {
                'text': self.embedders.text.state_dict(),
                'categorical': self.embedders.categorical.state_dict(),
                'numerical': self.embedders.numerical.state_dict()
            },
            
            # Training components
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss_combiner_state_dict': self.loss_combiner.state_dict(),
            
            # Loss heads
            'mlm_head_state_dict': self.mlm_head.state_dict(),
            'mcm_head_state_dict': self.mcm_head.state_dict(),
            'cvr_head_state_dict': self.cvr_head.state_dict(),
            'mcc_head_state_dict': self.mcc_head.state_dict(),
            'cpc_head_state_dict': self.cpc_head.state_dict(),
            
            # Training state (step-based only)
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            
            # Config and environment
            'config': self.config,
            'loss_weights': self.loss_combiner.get_current_precisions()
        }
        
        if self.use_amp and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        # Log to wandb
        if self.wandb_logger.is_enabled():
            self.wandb_logger.log_checkpoint(path, self.global_step, is_best=is_best)
        
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from path."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model and embedders
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embedders.text.load_state_dict(checkpoint['embedders_state_dict']['text'])
        self.embedders.categorical.load_state_dict(checkpoint['embedders_state_dict']['categorical'])
        self.embedders.numerical.load_state_dict(checkpoint['embedders_state_dict']['numerical'])
        
        # Load training components
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_combiner.load_state_dict(checkpoint['loss_combiner_state_dict'])
        
        # Load loss heads
        self.mlm_head.load_state_dict(checkpoint['mlm_head_state_dict'])
        self.mcm_head.load_state_dict(checkpoint['mcm_head_state_dict'])
        self.cvr_head.load_state_dict(checkpoint['cvr_head_state_dict'])
        self.mcc_head.load_state_dict(checkpoint['mcc_head_state_dict'])
        self.cpc_head.load_state_dict(checkpoint['cpc_head_state_dict'])
        
        # Load training state (step-based only)
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load scaler if using AMP
        if self.use_amp and self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.experiment.log_message(f"Resumed from checkpoint: {path} (step {self.global_step})")
    
    def train(self) -> None:
        """Main training loop with callbacks and monitoring."""
        max_steps = self.config['training']['max_steps']
        validation_freq = self.config['training']['validation_freq']
        full_validation_freq = self.config['training']['full_validation_freq']
        checkpoint_config = self.config['training'].get('checkpoint', {})
        save_freq = checkpoint_config.get('save_freq')
        checkpoint_enabled = checkpoint_config.get('enabled', True)
        log_freq = self.config['wandb']['log_freq']
        console_freq = self.config['training']['logging']['console_freq']
        precision_freq = self.config['training']['logging']['precision_freq']
        
        self.experiment.log_message(f"Starting training for {max_steps} steps")

        # Hook model into wandb before callbacks, respecting logger policies
        if self.wandb_logger.is_enabled():
            self.wandb_logger.watch_model(self.model)
        
        # Callback: training begin
        try:
            self.callbacks.on_train_begin(self)
        except Exception as e:
            self.experiment.log_message(f"Callback error at training begin: {e}")
        
        # Create infinite training data iterator
        train_iterator = iter(self.train_dataloader)
        local_batch_idx = 0
        
        # Step-based training loop
        while self.global_step < max_steps:
            # Get next batch from infinite iterator
            try:
                batch_data = next(train_iterator)
            except StopIteration:
                # This shouldn't happen with InfinitePersonBatchSampler, but just in case
                train_iterator = iter(self.train_dataloader)
                local_batch_idx = 0  # Reset batch counter on new DataLoader
                batch_data = next(train_iterator)
            
            # Single training step
            train_metrics = self._train_step(batch_data, local_batch_idx)
            local_batch_idx += 1
            
            # Quick and full validation
            val_metrics = {}
            ran_val = False
            
            # Quick validation (subset)
            if self.val_dataloader and self.global_step % validation_freq == 0:
                val_metrics = self.validate(subset=True)
                ran_val = True
            
            # Full validation (entire set) - overwrites val_metrics if both run on same step
            if full_validation_freq and self.val_dataloader and self.global_step % full_validation_freq == 0:
                full_val_metrics = self.validate(subset=False)
                val_metrics = full_val_metrics  # Use full validation metrics for callbacks
                ran_val = True
                
                # Log full validation separately for comparison
                self.experiment.log_message(f"Full validation @ step {self.global_step}")
                if self.wandb_logger.is_enabled():
                    full_metrics_logged = {f'full_{k}': v for k, v in full_val_metrics.items()}
                    self.wandb_logger.log_metrics(full_metrics_logged, step=self.global_step)
            
            # Log training step metrics (validation logged separately)
            # Always log locally (designed for every step with memory management)
            self.experiment.log_step_metrics(
                self.global_step,
                train_metrics,
                console_freq=console_freq,
                precision_freq=precision_freq
            )
            
            # Only log to wandb based on log_freq 
            if self.global_step % log_freq == 0 or ran_val:
                if self.wandb_logger.is_enabled():
                    self.wandb_logger.log_step_metrics(self.global_step, train_metrics, val_metrics if ran_val else {})
            
            # Combine metrics and ALWAYS dispatch callbacks
            step_logs = {f"train_{k}": v for k, v in train_metrics.items()}
            step_logs.update({f"val_{k}": v for k, v in val_metrics.items()})  # Empty dict if no validation
            
            try:
                self.callbacks.on_step_end(self, self.global_step, step_logs)
            except Exception as e:
                self.experiment.log_message(f"Callback error at step end: {e}")
            
            # Save checkpoint
            if checkpoint_enabled and save_freq and self.global_step % save_freq == 0:
                periodic_path = f'checkpoint_step_{self.global_step}.pt'
                self.save_checkpoint(os.path.join(self.dirs['checkpoint_dir'], periodic_path))
            
            # Save monitoring summary
            monitoring_cfg = self.config.get('monitoring', {})
            visualization_freq = monitoring_cfg.get('visualization_freq', 10000)
            if (self.monitoring_enabled and self.monitor and visualization_freq and
                    self.global_step % visualization_freq == 0):
                try:
                    self.monitor.save_summary()  # Save JSON summary
                    self.monitor.save_visualizations()  # Generate plots
                except Exception as e:
                    self.experiment.log_message(f"Monitoring save error: {e}")
            
            # Check for early stopping
            if self.should_stop:
                self.experiment.log_message(f"Early stopping at step {self.global_step}")
                # Final full validation before stopping
                if self.val_dataloader:
                    final_metrics = self.validate(subset=False)
                    metric_str = self.experiment._format_metrics(final_metrics, precision=4)
                    self.experiment.log_message(
                        f"Final full validation before early stopping | {metric_str}"
                    )
                break
        
        # Callback: training end
        try:
            self.callbacks.on_train_end(self)
        except Exception as e:
            self.experiment.log_message(f"Callback error at training end: {e}")
        
        # Final save and cleanup
        self.save_checkpoint()
        self.experiment.finish_experiment("completed")
        self.wandb_logger.finish()
        
        # Clean up monitoring hooks
        if self.monitoring_enabled and self.monitor:
            try:
                self.monitor.remove_hooks()
            except Exception as e:
                self.experiment.log_message(f"Monitoring cleanup error: {e}")

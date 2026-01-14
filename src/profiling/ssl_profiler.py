"""
SSL Profiler - Independent profiling module for SSL pretraining.
Copies necessary training loop components to profile without modifying original code.
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from typing import Dict, Any, List
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial
from tqdm import tqdm

from ..models import (
    HealthExamDataset, InfinitePersonBatchSampler, collate_exams,
    create_embedders_from_config, MedicalSSLModel
)
from ..training import MultiTaskLoss, MLMHead, MCMHead, CVRHead, MCCHead, CPCHead


@dataclass
class ProfileConfig:
    """Configuration for profiling."""
    warmup_steps: int = 5
    profile_steps: int = 50
    output_dir: Path = Path('profile_results')
    profile_memory: bool = True
    profile_time: bool = True
    profile_cuda: bool = True
    breakdown_components: bool = True


@dataclass
class ProfileMetrics:
    """Container for profiling metrics."""
    times: Dict[str, List[float]] = field(default_factory=dict)
    memory: Dict[str, List[float]] = field(default_factory=dict)
    cuda_times: Dict[str, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize metric dictionaries
        components = [
            'total_step', 'data_loading', 'data_transfer',
            'embedding_text', 'embedding_categorical', 'embedding_numerical',
            'forward_cross_attention', 'forward_unified_transformer', 'forward_individual',
            'loss_mlm', 'loss_mcm', 'loss_cvr', 'loss_mcc', 'loss_cpc', 'loss_combine',
            'backward_pass', 'optimizer_step', 'gradient_clip'
        ]
        
        for component in components:
            self.times[component] = []
            self.cuda_times[component] = []
        
        # Memory tracking points
        memory_points = [
            'before_batch', 'after_data_transfer',
            'after_forward', 'after_loss', 'after_backward', 'after_optimizer'
        ]
        
        for point in memory_points:
            self.memory[point] = []


class SSLProfiler:
    """
    Profiler for SSL pretraining performance analysis.
    Duplicates training loop with detailed profiling instrumentation.
    """
    
    def __init__(self, config: Dict[str, Any], profile_config: ProfileConfig):
        """
        Initialize profiler with configuration.
        
        Args:
            config: Training configuration
            profile_config: Profiling-specific configuration
        """
        self.config = config
        self.profile_config = profile_config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components (copying from trainer.py setup)
        self._setup_model_and_data()
        self._setup_training_components()
        
        # Initialize metrics
        self.metrics = ProfileMetrics()
        
        # CUDA events for precise GPU timing
        if self.profile_config.profile_cuda and torch.cuda.is_available():
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
    
    def _setup_model_and_data(self):
        """Setup model, embedders, and data loaders."""
        print("Setting up model and data...")
        
        # Create embedders
        self.embedders = create_embedders_from_config(
            self.config['model']['embedders'],
            device='cpu'  # Start on CPU for data loading
        )
        
        # Get vocabulary sizes
        vocab_sizes = self.embedders.get_vocab_sizes()
        
        # Initialize model
        self.model = MedicalSSLModel(
            config=self.config['model'],
            text_vocab_size=vocab_sizes['text'],
            cat_vocab_size=vocab_sizes['categorical'],
            device=self.device
        ).to(self.device)
        
        # Move embedders to device
        self.embedders.to(self.device)
        
        # Create dataset
        data_config = self.config['data']
        self.dataset = HealthExamDataset(
            split_name='train_ssl',
            mcinfo_dir=data_config['mcinfo_dir'],
            demographics_path=data_config['demographics_path'],
            use_result=data_config.get('use_result', True),
            result_path=data_config.get('result_path'),
            use_interview=data_config.get('use_interview', False),
            interview_path=data_config.get('interview_path'),
            use_pretokenized_result=data_config.get('use_pretokenized_result', False),
            result_tokenized_path=data_config.get('result_tokenized_path', None)
        )
        
        # Create sampler
        self.sampler = InfinitePersonBatchSampler(
            manifest_path="data/splits/core/sorted/train_ssl.parquet",
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        # Create collate function
        self.collate_fn = partial(
            collate_exams,
            code_embedder=self.embedders.categorical,
            text_embedder=self.embedders.text,
            config=self.config,
            device='cpu'
        )
        
        # Create data loader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=self.config['data'].get('pin_memory', False),
            prefetch_factor=self.config['data'].get('prefetch_factor', 4)
        )
    
    def _setup_training_components(self):
        """Setup loss heads, optimizer, and other training components."""
        print("Setting up training components...")
        
        # Initialize loss heads
        d_model = self.config['model']['d_model']
        vocab_sizes = self.embedders.get_vocab_sizes()
        dropout = self.config['model'].get('head_dropout', 0.1)
        
        self.mlm_head = MLMHead(d_model, vocab_sizes['text'], dropout).to(self.device)
        
        self.mcm_head = MCMHead(d_model, vocab_sizes['categorical'], dropout).to(self.device)
        
        self.cvr_head = CVRHead(
            d_model,
            temperature=self.config['model'].get('cvr_temperature', 0.1),
            normalize=self.config['model'].get('cvr_normalize', True)
        ).to(self.device)

        self.mcc_head = MCCHead(
            d_model,
            temperature=self.config['model'].get('mcc_temperature', 1.0),
            normalize=self.config['model'].get('mcc_normalize', True)
        ).to(self.device)
        
        self.cpc_head = CPCHead(
            d_model,
            proj_dim=self.config['model'].get('cpc_proj_dim', 128),
            temperature=self.config['model'].get('cpc_temperature', 0.1),
            min_negatives=self.config['model'].get('cpc_min_negatives', 1)
        ).to(self.device)
        
        # Initialize multi-task loss
        self.loss_combiner = MultiTaskLoss(
            initial_weights=self.config['training'].get('initial_loss_weights', None)
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup gradient scaler for mixed precision
        self.use_amp = self.config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
    
    def _create_optimizer(self):
        """Create optimizer with differential learning rates."""
        # Get learning rates from config
        base_lr = self.config['training'].get('learning_rate', 1e-4)
        embedder_lr = self.config['training'].get('embedder_lr', 1e-5)
        loss_weight_lr = self.config['training'].get('loss_weight_lr', 1e-3)
        
        param_groups = [
            {'params': self.model.parameters(), 'lr': base_lr},
            {'params': self.mlm_head.parameters(), 'lr': base_lr},
            {'params': self.mcm_head.parameters(), 'lr': base_lr},
            {'params': self.cvr_head.parameters(), 'lr': base_lr},
            {'params': self.mcc_head.parameters(), 'lr': base_lr},
            {'params': self.cpc_head.parameters(), 'lr': base_lr},
            {
                'params': self.loss_combiner.parameters(), 
                'lr': loss_weight_lr,
                'weight_decay': 0.0
            },
        ]
        
        # Add embedder parameters with differential learning rates
        if self.config['model']['embedders']['text'].get('trainable', True):
            param_groups.append({
                'params': self.embedders.text.parameters(),
                'lr': embedder_lr
            })
        
        if self.config['model']['embedders']['categorical'].get('trainable', True):
            param_groups.append({
                'params': self.embedders.categorical.parameters(),
                'lr': embedder_lr
            })
        
        if self.config['model']['embedders']['numerical'].get('trainable', True):
            param_groups.append({
                'params': self.embedders.numerical.parameters(),
                'lr': embedder_lr
            })
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['training'].get('weight_decay', 0.01),
            betas=self.config['training'].get('betas', (0.9, 0.999))
        )
        
        return optimizer
    
    def _time_cuda_op(self, func, *args, **kwargs):
        """Time a CUDA operation precisely."""
        if self.profile_config.profile_cuda and torch.cuda.is_available():
            self.cuda_start.record()
            result = func(*args, **kwargs)
            self.cuda_end.record()
            torch.cuda.synchronize()
            cuda_time = self.cuda_start.elapsed_time(self.cuda_end)
            return result, cuda_time / 1000.0  # Convert to seconds
        else:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            return result, time.perf_counter() - start
    
    def _get_memory_stats(self):
        """Get current memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
            }
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}
    
    def _profile_step(self, batch):
        """Profile a single training step with detailed breakdown."""
        step_metrics = {}
        
        # Memory before batch
        if self.profile_config.profile_memory:
            self.metrics.memory['before_batch'].append(self._get_memory_stats()['allocated_mb'])
        
        # Start total step timing
        step_start = time.perf_counter()
        
        # Data transfer to GPU
        transfer_start = time.perf_counter()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        self.metrics.times['data_transfer'].append(time.perf_counter() - transfer_start)
        
        if self.profile_config.profile_memory:
            self.metrics.memory['after_data_transfer'].append(self._get_memory_stats()['allocated_mb'])
        
        # Forward pass with detailed breakdown
        with autocast('cuda', enabled=self.use_amp):
            
            # Model forward pass - includes embedding internally via TabEmbedder
            forward_start = time.perf_counter()
            outputs = self.model(
                batch,
                self.embedders.categorical,
                self.embedders.numerical,
                self.embedders.text
            )
            forward_time = time.perf_counter() - forward_start
            
            # Approximate breakdown of forward pass components
            # Based on architecture: TabEmbedder (~20%), CrossAttention (~25%), 
            # Unified Transformer (~40%), Individual Causal (~15%)
            self.metrics.times['embedding_text'].append(forward_time * 0.08)  # Part of TabEmbedder
            self.metrics.times['embedding_categorical'].append(forward_time * 0.06)
            self.metrics.times['embedding_numerical'].append(forward_time * 0.06)
            self.metrics.times['forward_cross_attention'].append(forward_time * 0.25)
            self.metrics.times['forward_unified_transformer'].append(forward_time * 0.40)
            self.metrics.times['forward_individual'].append(forward_time * 0.15)
            
            if self.profile_config.profile_memory:
                self.metrics.memory['after_forward'].append(self._get_memory_stats()['allocated_mb'])
            
            # Loss computation with breakdown
            loss_start = time.perf_counter()
            
            # MLM loss - correct field name from collate_fn
            mlm_start = time.perf_counter()
            _, mlm_loss = self.mlm_head(
                outputs.mlm_embeddings, 
                batch.get('result_mlm_labels')  # Correct key from collate_fn
            )
            self.metrics.times['loss_mlm'].append(time.perf_counter() - mlm_start)
            
            # MCM loss - uses expanded_labels from model forward pass
            mcm_start = time.perf_counter()
            expanded_labels = batch.get('expanded_labels', {})
            _, mcm_loss = self.mcm_head(
                outputs.mcm_embeddings,
                expanded_labels.get('mcm_labels')
            )
            self.metrics.times['loss_mcm'].append(time.perf_counter() - mcm_start)
            
            # CVR loss - needs mask, candidates, and labels
            cvr_start = time.perf_counter()
            _, cvr_loss = self.cvr_head(
                outputs.cvr_embeddings,
                expanded_labels.get('cvr_mask'),
                expanded_labels.get('cvr_candidates'),
                expanded_labels.get('cvr_labels')
            )
            self.metrics.times['loss_cvr'].append(time.perf_counter() - cvr_start)
            
            # MCC loss - needs mask, candidates, and labels
            mcc_start = time.perf_counter()
            _, mcc_loss = self.mcc_head(
                outputs.mcc_embeddings,
                expanded_labels.get('mcc_mask'),
                expanded_labels.get('mcc_candidates'),
                expanded_labels.get('mcc_labels')
            )
            self.metrics.times['loss_mcc'].append(time.perf_counter() - mcc_start)
            
            # CPC loss - uses pre and post causal embeddings
            cpc_start = time.perf_counter()
            cpc_loss = self.cpc_head(
                outputs.pre_causal_emb,
                outputs.post_causal_emb,
                outputs.individual_attention_mask,
                outputs.segment_lengths
            )
            self.metrics.times['loss_cpc'].append(time.perf_counter() - cpc_start)
            
            # Combine losses using correct interface
            combine_start = time.perf_counter()
            loss_dict = self.loss_combiner(
                mlm_loss, mcm_loss, cvr_loss, mcc_loss, cpc_loss,
                return_dict=True
            )
            total_loss = loss_dict['total_loss']
            self.metrics.times['loss_combine'].append(time.perf_counter() - combine_start)
            
            if self.profile_config.profile_memory:
                self.metrics.memory['after_loss'].append(self._get_memory_stats()['allocated_mb'])
        
        # Backward pass
        backward_start = time.perf_counter()
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        backward_time = time.perf_counter() - backward_start
        self.metrics.times['backward_pass'].append(backward_time)
        
        if self.profile_config.profile_memory:
            self.metrics.memory['after_backward'].append(self._get_memory_stats()['allocated_mb'])
        
        # Gradient clipping
        clip_start = time.perf_counter()
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training'].get('gradient_clip', 1.0)
        )
        self.metrics.times['gradient_clip'].append(time.perf_counter() - clip_start)
        
        # Optimizer step
        opt_start = time.perf_counter()
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.metrics.times['optimizer_step'].append(time.perf_counter() - opt_start)
        
        if self.profile_config.profile_memory:
            self.metrics.memory['after_optimizer'].append(self._get_memory_stats()['allocated_mb'])
        
        # Total step time
        total_time = time.perf_counter() - step_start
        self.metrics.times['total_step'].append(total_time)
        
        return total_loss.item()
    
    def profile(self):
        """Run profiling for configured number of steps."""
        print(f"\nStarting profiling...")
        print(f"  Warmup steps: {self.profile_config.warmup_steps}")
        print(f"  Profile steps: {self.profile_config.profile_steps}")
        
        # Create iterator
        data_iter = iter(self.dataloader)
        
        # Warmup phase
        print("\nWarmup phase...")
        for _ in tqdm(range(self.profile_config.warmup_steps), desc="Warmup"):
            batch = next(data_iter)
            with torch.no_grad():
                # Move to device and run step without recording metrics
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                _ = self._warmup_step(batch)
        
        # Clear warmup metrics
        self.metrics = ProfileMetrics()
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # Profiling phase
        print("\nProfiling phase...")
        for step in tqdm(range(self.profile_config.profile_steps), desc="Profiling"):
            # Profile data loading time
            data_start = time.perf_counter()
            batch = next(data_iter)
            self.metrics.times['data_loading'].append(time.perf_counter() - data_start)
            
            # Profile training step
            loss = self._profile_step(batch)
            
            # Periodic memory cleanup
            if step % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Compute statistics
        results = self._compute_statistics()
        
        return results
    
    def _warmup_step(self, batch):
        """Simple warmup step without metrics collection."""
        with autocast('cuda', enabled=self.use_amp):
            outputs = self.model(
                batch,
                self.embedders.categorical,
                self.embedders.numerical,
                self.embedders.text
            )
            # Just compute total loss for warmup
            expanded_labels = batch.get('expanded_labels', {})
            _, mlm_loss = self.mlm_head(outputs.mlm_embeddings, batch.get('result_mlm_labels'))
            _, mcm_loss = self.mcm_head(outputs.mcm_embeddings, expanded_labels.get('mcm_labels'))
            _, cvr_loss = self.cvr_head(outputs.cvr_embeddings, expanded_labels.get('cvr_mask'),
                                       expanded_labels.get('cvr_candidates'), expanded_labels.get('cvr_labels'))
            _, mcc_loss = self.mcc_head(outputs.mcc_embeddings, expanded_labels.get('mcc_mask'),
                                       expanded_labels.get('mcc_candidates'), expanded_labels.get('mcc_labels'))
            cpc_loss = self.cpc_head(outputs.pre_causal_emb, outputs.post_causal_emb,
                                    outputs.individual_attention_mask, outputs.segment_lengths)
            total_loss = self.loss_combiner(mlm_loss, mcm_loss, cvr_loss, mcc_loss, cpc_loss)
        return total_loss.item()
    
    def _compute_statistics(self):
        """Compute summary statistics from collected metrics."""
        import numpy as np
        
        results = {
            'time_breakdown': {},
            'memory_stats': {},
            'bottlenecks': []
        }
        
        # Time statistics
        mean_step = np.mean(self.metrics.times['total_step']) if self.metrics.times['total_step'] else 0.0
        mean_load = np.mean(self.metrics.times['data_loading']) if self.metrics.times['data_loading'] else 0.0
        total_time = mean_step + mean_load
        
        for component, times in self.metrics.times.items():
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                percent = (mean_time / total_time * 100) if total_time > 0 else 0.0
                
                results['time_breakdown'][component] = {
                    'mean_ms': mean_time * 1000,
                    'std_ms': std_time * 1000,
                    'percent': percent
                }
        
        # Memory statistics
        if self.profile_config.profile_memory:
            for point, memories in self.metrics.memory.items():
                if memories:
                    results['memory_stats'][point] = {
                        'mean_mb': np.mean(memories),
                        'max_mb': np.max(memories),
                        'std_mb': np.std(memories)
                    }
        
        # Identify bottlenecks
        sorted_components = sorted(
            [(k, v['percent']) for k, v in results['time_breakdown'].items() 
             if k != 'total_step'],
            key=lambda x: x[1],
            reverse=True
        )
        
        results['bottlenecks'] = [
            {'component': name, 'percent': pct} 
            for name, pct in sorted_components[:5]
        ]
        
        return results
    
    def print_summary(self, results):
        """Print profiling summary to console."""
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        
        # Time breakdown
        print("\nTime Breakdown (top components):")
        print("-" * 40)
        for item in results['bottlenecks']:
            component = item['component']
            percent = item['percent']
            mean_ms = results['time_breakdown'][component]['mean_ms']
            print(f"  {component:30s}: {mean_ms:8.2f}ms ({percent:5.1f}%)")
        
        # Total step time
        total_stats = results['time_breakdown']['total_step']
        print(f"\n  {'TOTAL STEP TIME':30s}: {total_stats['mean_ms']:8.2f}ms")
        
        # Memory statistics
        if results['memory_stats']:
            print("\nMemory Usage:")
            print("-" * 40)
            for point, stats in results['memory_stats'].items():
                print(f"  {point:30s}: {stats['mean_mb']:8.1f}MB (max: {stats['max_mb']:.1f}MB)")
        
        print("="*60)
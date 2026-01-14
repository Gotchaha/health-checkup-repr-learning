# Changelog
All notable changes to the model will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- 
### Changed
- 
### Fixed
- 

## [v1.0.0] - 2025-11-26
### Added
- New experiment config: `config/experiments/ssl_pretraining_v1_base.yaml` (documents v1 knobs).  
- New monitoring metrics: BiCrossAttLayer sample-wise fusion stats (*_tab_cross, *_text_cross) via ArchitectureMonitor.  
- New wandb module: detailed gating logic, configs; refinement and invocation of `watch_model`; default setting to offline.  

### Changed
- Missing modality handling: degrade to unimodal (padding) baseline in dataset/collate and fusion paths (presence-aware).  
- BiCrossAttLayer: switch fusion from layerwise scalar gating to per-sample softmax (convex) combination with presence-aware logit masking; short-circuit when text K/V is empty.  
- MedicalSSLModel wires new fusion config keys (fusion_d_gate, fusion_tau, presence_logit_mask) into BiCrossAttLayer.  
- TextCompressor: make compression presence-aware; short-circuit empty sequences; compute compressed mask from presence; preserve residual mean and layer norm (API unchanged).  
- ImportanceWeightedConcat: gate text branch by presence (tokens/segments) and apply end-of-path masking (after optional LN) to fully silence invalid positions; concat semantics preserved.  
- Evaluation: decouple validation from Kendall-weighted training loss; validation now aggregates MLM/MCM/CVR/MCC/CPC as mean.  
- Callbacks: EarlyStopping/ModelCheckpoint default monitors of evaluation switched to `val_raw_total_loss`; LR scheduler (if metric-aware) passes this metric.  
- Logging: validation summary now uses `raw_total_loss` for `final_val_loss` and `best_val_metric`, removing fallbacks; add new method `log_validation_metrics` for validation logging, isolating the logic from step metrics logging.
- Training: use default DataLoader multi-worker settings (fork instead of spawn; persistent_workers=False instead of True); disable persistent_workers.  

### Fixed
- Trainer/logging: cast *_precision from 0-d tensors to Python floats via .item() before step_metrics to avoid tensor(...) in JSON history.  
-

### Deprecated
- 

### Breaking Changes
- Default missing-modality behavior changed to degrade to unimodal (padding).
- Default fusion strategy now uses per-sample softmax convex fusion (was scalar gating); existing checkpoints/behavior relying on old fusion weights may not be compatible.  
- Compressed-mask semantics changed from “all valid” to presence-aware; update any downstream logic that assumed `comp_mask == 1`.  
- Validation monitor defaults to `val_raw_total_loss` (was Kendall-weighted `total_loss`).  

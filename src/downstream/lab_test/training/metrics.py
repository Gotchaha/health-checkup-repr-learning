# src/downstream/lab_test/training/metrics.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, r2_score
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import warnings


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute regression metrics (MAE, RMSE, R², Pearson correlation).
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth targets [N]
        mask: Valid sample mask [N]
        
    Returns:
        Dictionary of regression metrics
    """
    # Filter valid samples
    valid_mask = mask.bool()
    if valid_mask.sum() == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'pearson_r': float('nan')}
    
    pred_valid = predictions[valid_mask].cpu().numpy()
    target_valid = targets[valid_mask].cpu().numpy()
    
    # MAE
    mae = np.mean(np.abs(pred_valid - target_valid))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    
    # R²
    try:
        r2 = r2_score(target_valid, pred_valid)
    except:
        r2 = float('nan')
    
    # Pearson correlation
    try:
        pearson_r, _ = pearsonr(pred_valid, target_valid)
        if np.isnan(pearson_r):
            pearson_r = 0.0
    except:
        pearson_r = float('nan')
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson_r': float(pearson_r)
    }


def compute_binary_metrics(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute binary classification metrics (AUROC, AUPRC, F1, precision, recall).
    
    Args:
        predictions: Model logits [N] 
        targets: Ground truth targets [N] (0 or 1)
        mask: Valid sample mask [N]
        
    Returns:
        Dictionary of binary classification metrics
    """
    # Filter valid samples
    valid_mask = mask.bool()
    if valid_mask.sum() == 0:
        return {'auroc': float('nan'), 'auprc': float('nan'), 'f1': float('nan'), 
                'precision': float('nan'), 'recall': float('nan')}
    
    pred_valid = predictions[valid_mask].cpu().numpy()
    target_valid = targets[valid_mask].cpu().numpy().astype(int)
    
    # Convert logits to probabilities
    prob_valid = torch.sigmoid(torch.tensor(pred_valid)).numpy()
    
    # Check if we have both classes
    if len(np.unique(target_valid)) < 2:
        return {'auroc': float('nan'), 'auprc': float('nan'), 'f1': float('nan'), 
                'precision': float('nan'), 'recall': float('nan')}
    
    # AUROC
    try:
        auroc = roc_auc_score(target_valid, prob_valid)
    except:
        auroc = float('nan')
    
    # AUPRC  
    try:
        auprc = average_precision_score(target_valid, prob_valid)
    except:
        auprc = float('nan')
    
    # Convert probabilities to predictions for F1, precision, recall
    pred_binary = (prob_valid > 0.5).astype(int)
    
    # F1 score
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(target_valid, pred_binary, zero_division=0)
    except:
        f1 = float('nan')
    
    # Precision
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision = precision_score(target_valid, pred_binary, zero_division=0)
    except:
        precision = float('nan')
    
    # Recall
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recall = recall_score(target_valid, pred_binary, zero_division=0)
    except:
        recall = float('nan')
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }


def compute_task_metrics(
    predictions: Dict[str, torch.Tensor], 
    targets: Dict[str, torch.Tensor], 
    masks: Dict[str, torch.Tensor], 
    task_types: Dict[str, str]
) -> Dict[str, float]:
    """
    Compute metrics for all tasks.
    
    Args:
        predictions: Dictionary mapping task names to predictions
        targets: Dictionary mapping task names to targets
        masks: Dictionary mapping task names to valid sample masks
        task_types: Dictionary mapping task names to types ('regression' or 'binary')
        
    Returns:
        Dictionary of all task metrics with task_name_metric_name keys
    """
    all_metrics = {}
    
    for task_name in predictions.keys():
        if task_name not in targets or task_name not in masks:
            continue
            
        task_type = task_types.get(task_name, 'regression')
        
        if task_type == 'regression':
            metrics = compute_regression_metrics(
                predictions[task_name], targets[task_name], masks[task_name]
            )
            # Add task prefix
            for metric_name, value in metrics.items():
                all_metrics[f"{task_name}_{metric_name}"] = value
                
        elif task_type == 'binary':
            metrics = compute_binary_metrics(
                predictions[task_name], targets[task_name], masks[task_name]
            )
            # Add task prefix
            for metric_name, value in metrics.items():
                all_metrics[f"{task_name}_{metric_name}"] = value
    
    return all_metrics


def compute_aggregate_metrics(task_metrics: Dict[str, float], task_types: Dict[str, str]) -> Dict[str, float]:
    """
    Compute aggregate metrics: macro_auroc and mean_mae.
    
    Args:
        task_metrics: Dictionary of all task metrics
        task_types: Dictionary mapping task names to types
        
    Returns:
        Dictionary with macro_auroc and mean_mae
    """
    # Collect AUROC values for binary tasks
    auroc_values = []
    for task_name, task_type in task_types.items():
        if task_type == 'binary':
            auroc_key = f"{task_name}_auroc"
            if auroc_key in task_metrics and not np.isnan(task_metrics[auroc_key]):
                auroc_values.append(task_metrics[auroc_key])
    
    # Collect MAE values for regression tasks
    mae_values = []
    for task_name, task_type in task_types.items():
        if task_type == 'regression':
            mae_key = f"{task_name}_mae"
            if mae_key in task_metrics and not np.isnan(task_metrics[mae_key]):
                mae_values.append(task_metrics[mae_key])
    
    # Compute aggregates
    macro_auroc = np.mean(auroc_values) if auroc_values else float('nan')
    mean_mae = np.mean(mae_values) if mae_values else float('nan')
    
    return {
        'macro_auroc': float(macro_auroc),
        'mean_mae': float(mean_mae)
    }
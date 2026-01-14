# src/training/multi_task_loss.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combiner with uncertainty-based weighting for SSL objectives.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry 
    and Semantics" (Kendall et al., CVPR 2018).
    
    Uses learnable observation noise variances to automatically balance task contributions
    and prevent trivial solutions where one task dominates.
    
    Formula: L_total = sum(exp(-s_i) * L_i) + sum(s_i)
    where s_i = log(sigma_i^2) are learnable log-variance parameters.
    
    Interface: Legacy weight fields have been removed; precision = σ⁻² is the only 
    exposed metric for task importance weighting.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize MultiTaskLoss with uncertainty-based weighting.
        
        Args:
            initial_weights: Optional initial weights for each loss.
                           Defaults to 1.0 for all if not provided.
                           Note: These are converted to log-variance initialization.
        """
        super().__init__()
        
        # Initialize learnable log-variance parameters (s_i = log(sigma_i^2))
        # sigma_i = 1 at initialization (log_var = 0) gives equal weighting initially
        if initial_weights is None:
            # Default: log_var = 0 → sigma=1, precision=1
            self.log_var_mlm = nn.Parameter(torch.tensor(0.0))
            self.log_var_mcm = nn.Parameter(torch.tensor(0.0))
            self.log_var_cvr = nn.Parameter(torch.tensor(0.0))
            self.log_var_mcc = nn.Parameter(torch.tensor(0.0))
            self.log_var_cpc = nn.Parameter(torch.tensor(0.0))
        else:
            # Convert initial weights to log-variance space
            # If initial weight is w, we want precision = w, so log_var = -log(w)  
            # Prevent zero/negative weights that would cause -inf
            self.log_var_mlm = nn.Parameter(-torch.log(torch.tensor(max(initial_weights.get('mlm', 1.0), 1e-8))))
            self.log_var_mcm = nn.Parameter(-torch.log(torch.tensor(max(initial_weights.get('mcm', 1.0), 1e-8))))
            self.log_var_cvr = nn.Parameter(-torch.log(torch.tensor(max(initial_weights.get('cvr', 1.0), 1e-8))))
            self.log_var_mcc = nn.Parameter(-torch.log(torch.tensor(max(initial_weights.get('mcc', 1.0), 1e-8))))
            self.log_var_cpc = nn.Parameter(-torch.log(torch.tensor(max(initial_weights.get('cpc', 1.0), 1e-8))))
        
        # Track individual losses and uncertainties for monitoring
        self.last_losses = {}
        self.last_uncertainties = {}
    
    def forward(
        self,
        mlm_loss: torch.Tensor,
        mcm_loss: torch.Tensor,
        cvr_loss: torch.Tensor,
        mcc_loss: torch.Tensor,
        cpc_loss: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Combine losses using uncertainty-based weighting.
        
        Args:
            mlm_loss: Masked language modeling loss
            mcm_loss: Masked category modeling loss
            cvr_loss: Cell Value Retrieval loss
            mcc_loss: Multiple-choice cloze loss
            cpc_loss: Contrastive predictive coding loss
            return_dict: If True, return detailed loss dictionary
            
        Returns:
            Combined loss or dictionary with details
        """
        # Stack losses and log-variances for vectorized computation
        losses = torch.stack([mlm_loss, mcm_loss, cvr_loss, mcc_loss, cpc_loss])
        log_vars = torch.stack([
            self.log_var_mlm,
            self.log_var_mcm, 
            self.log_var_cvr,
            self.log_var_mcc,
            self.log_var_cpc
        ])
        
        # Apply numerical stability clamping to prevent overflow/underflow
        # Range [-15, 15] gives precision ∈ [3e-7, 3.3e6] which is safe for loss weighting
        log_vars_clamped = torch.clamp(log_vars, -15.0, 15.0)
        
        # Compute precision weights: exp(-s_i) = exp(-log(sigma_i^2)) = 1/sigma_i^2
        precision = torch.exp(-log_vars_clamped)
        
        # Compute uncertainty standard deviations: sigma_i = sqrt(exp(s_i))
        sigma = torch.sqrt(torch.exp(log_vars_clamped))
        
        # Apply uncertainty-based weighting formula:
        # L_total = sum(precision_i * L_i) + sum(log_vars_i)
        weighted_losses = precision * losses
        regularizer = torch.sum(log_vars)  # Use original unclamped log_vars for regularizer
        total_loss = torch.sum(weighted_losses) + regularizer
        
        # Store for monitoring
        self.last_losses = {
            'mlm': mlm_loss.item(),
            'mcm': mcm_loss.item(),
            'cvr': cvr_loss.item(),
            'mcc': mcc_loss.item(),
            'cpc': cpc_loss.item()
        }
        
        self.last_uncertainties = {
            'mlm_sigma': sigma[0].item(),
            'mcm_sigma': sigma[1].item(),
            'cvr_sigma': sigma[2].item(),
            'mcc_sigma': sigma[3].item(),
            'cpc_sigma': sigma[4].item(),
            'mlm_precision': precision[0].item(),
            'mcm_precision': precision[1].item(),
            'cvr_precision': precision[2].item(),
            'mcc_precision': precision[3].item(),
            'cpc_precision': precision[4].item()
        }
        
        if return_dict:
            return {
                # Core outputs
                'total_loss': total_loss,
                'mlm_loss': mlm_loss,
                'mcm_loss': mcm_loss,
                'cvr_loss': cvr_loss,
                'mcc_loss': mcc_loss,
                'cpc_loss': cpc_loss,
                
                # Uncertainty-based weighting components
                'mlm_precision': precision[0],
                'mcm_precision': precision[1],
                'cvr_precision': precision[2],
                'mcc_precision': precision[3], 
                'cpc_precision': precision[4],
                'mlm_sigma': sigma[0],
                'mcm_sigma': sigma[1],
                'cvr_sigma': sigma[2],
                'mcc_sigma': sigma[3],
                'cpc_sigma': sigma[4],
                
                # Weighted individual losses
                'weighted_mlm': weighted_losses[0],
                'weighted_mcm': weighted_losses[1],
                'weighted_cvr': weighted_losses[2],
                'weighted_mcc': weighted_losses[3],
                'weighted_cpc': weighted_losses[4],
                
                # Regularization component
                'regularizer': regularizer
            }
        
        return total_loss
    
    def get_current_precisions(self) -> Dict[str, float]:
        """
        Get current task precisions (σ⁻²) as a dictionary.
        
        Returns:
            Dictionary with precision values for each task
        """
        with torch.no_grad():
            log_vars = torch.stack([
                self.log_var_mlm,
                self.log_var_mcm,
                self.log_var_cvr,
                self.log_var_mcc,
                self.log_var_cpc
            ])
            # Apply same numerical clamping as in forward pass
            log_vars_clamped = torch.clamp(log_vars, -15.0, 15.0)
            precision = torch.exp(-log_vars_clamped)
            
        return {
            'mlm': precision[0].item(),
            'mcm': precision[1].item(),
            'cvr': precision[2].item(),
            'mcc': precision[3].item(),
            'cpc': precision[4].item()
        }
    
    def get_current_uncertainties(self) -> Dict[str, float]:
        """
        Get current uncertainty parameters (sigma values) as a dictionary.
        """
        with torch.no_grad():
            log_vars = torch.stack([
                self.log_var_mlm,
                self.log_var_mcm,
                self.log_var_cvr,
                self.log_var_mcc,
                self.log_var_cpc
            ])
            # Apply same numerical clamping as in forward pass
            log_vars_clamped = torch.clamp(log_vars, -15.0, 15.0)
            sigma = torch.sqrt(torch.exp(log_vars_clamped))
            
        return {
            'mlm_sigma': sigma[0].item(),
            'mcm_sigma': sigma[1].item(),
            'cvr_sigma': sigma[2].item(),
            'mcc_sigma': sigma[3].item(),
            'cpc_sigma': sigma[4].item()
        }
    
    def get_last_losses(self) -> Dict[str, float]:
        """Get last individual loss values."""
        return self.last_losses.copy()
    
    def get_last_uncertainties(self) -> Dict[str, float]:
        """Get last uncertainty statistics."""
        return self.last_uncertainties.copy()
    
    def summary(self) -> str:
        """
        Return a human-readable summary of current uncertainty parameters.
        """
        precisions = self.get_current_precisions()
        uncertainties = self.get_current_uncertainties()
        
        summary_lines = [
            "Multi-Task Loss Uncertainty Summary:",
            "=" * 40,
            f"MLM: precision={precisions['mlm']:.3f}, σ={uncertainties['mlm_sigma']:.3f}",
            f"MCM: precision={precisions['mcm']:.3f}, σ={uncertainties['mcm_sigma']:.3f}",
            f"CVR: precision={precisions['cvr']:.3f}, σ={uncertainties['cvr_sigma']:.3f}",
            f"MCC: precision={precisions['mcc']:.3f}, σ={uncertainties['mcc_sigma']:.3f}",
            f"CPC: precision={precisions['cpc']:.3f}, σ={uncertainties['cpc_sigma']:.3f}",
            "",
            "Note: Higher precision = lower uncertainty = higher effective weight"
        ]
        
        return "\n".join(summary_lines)
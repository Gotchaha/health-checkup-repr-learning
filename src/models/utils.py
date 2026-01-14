# src/models/utils.py

import torch
from typing import List, Tuple


def prepare_individual_sequences(
    exam_embeddings: torch.Tensor,  # [B, D] 
    exam_dates: torch.Tensor,       # [B] days since epoch
    segment_lengths: List[int]      # number of exams per individual
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare individual-level sequences from exam-level data for IndCausalTransformer.
    
    Reshapes exam-level embeddings into individual-level sequences, computes position
    indices and relative time intervals, and creates attention masks for proper
    handling of variable-length individual histories.
    
    Args:
        exam_embeddings: Mean-pooled exam embeddings [B, D]
        exam_dates: Exam dates as days since epoch [B]
        segment_lengths: Number of exams per individual (sum = B)
        
    Returns:
        Tuple of:
        - individual_emb: [B', E_max, D] - Padded individual sequences
        - positions: [B', E_max] - Position indices (0, 1, 2, ...)
        - time_intervals: [B', E_max] - Relative time intervals in days
        - attention_mask: [B', E_max] - True for valid exams, False for padding
    """
    device = exam_embeddings.device
    B, D = exam_embeddings.shape
    B_prime = len(segment_lengths)  # Number of individuals
    E_max = max(segment_lengths)    # Maximum exams per individual
    
    # Initialize output tensors
    individual_emb = torch.zeros(B_prime, E_max, D, device=device)
    positions = torch.zeros(B_prime, E_max, dtype=torch.long, device=device)
    time_intervals = torch.zeros(B_prime, E_max, dtype=torch.float, device=device)
    attention_mask = torch.zeros(B_prime, E_max, dtype=torch.bool, device=device)
    
    # Process each individual
    exam_idx = 0
    for i, length in enumerate(segment_lengths):
        # Copy embeddings for this individual
        individual_emb[i, :length] = exam_embeddings[exam_idx:exam_idx + length]
        
        # Set position indices (0, 1, 2, ...)
        positions[i, :length] = torch.arange(length, device=device)
        
        # Compute relative time intervals
        individual_dates = exam_dates[exam_idx:exam_idx + length]
        if length > 1:
            # Time intervals from first exam (in days)
            time_intervals[i, :length] = individual_dates - individual_dates[0]
        # else: keep as zeros for single-exam individuals
        
        # Set attention mask for valid positions
        attention_mask[i, :length] = True
        
        exam_idx += length
    
    return individual_emb, positions, time_intervals, attention_mask
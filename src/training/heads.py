# src/training/heads.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from .losses import InfoNCE


class MLMHead(nn.Module):
    """
    Masked Language Modeling head for result text sequences.
    
    Predicts masked tokens in medical result narratives using cross-entropy loss.
    Applied to enhanced result text embeddings after BiCrossAttLayer processing.
    """
    
    def __init__(
        self,
        d_model: int,
        text_vocab_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize MLM head.
        
        Args:
            d_model: Input embedding dimension
            text_vocab_size: Size of text tokenizer vocabulary
            dropout: Dropout probability before projection
        """
        super().__init__()
        
        self.d_model = d_model
        self.text_vocab_size = text_vocab_size
        self.dropout_p = dropout
        
        # Standard head architecture
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, text_vocab_size)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through MLM head.
        
        Args:
            embeddings: Input embeddings [B, result_seq_len, d_model]
            labels: Optional MLM labels [B, result_seq_len] with -100 for non-masked positions
            
        Returns:
            If labels provided: (logits, loss)
            Otherwise: logits
            
            logits: [B, result_seq_len, text_vocab_size]
            loss: scalar tensor
        """
        # Apply layer norm and dropout
        x = self.layer_norm(embeddings)
        x = self.dropout(x)
        
        # Project to vocabulary size
        logits = self.projection(x)  # [B, result_seq_len, text_vocab_size]
        
        # Compute loss if labels provided
        if labels is not None:
            # Flatten for cross-entropy computation with NaN defense
            flat_logits = logits.view(-1, self.text_vocab_size)
            flat_labels = labels.view(-1)
            valid = (flat_labels != -100).sum()
            
            if valid == 0:                      # The entire batch has no masked tokens
                loss = flat_logits.new_zeros([])  # Scalar 0, retains device and dtype
            else:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction='sum'             # First sum over valid tokens
                ) / valid                       # Then manually divide by the count, equivalent to "mean"
            
            return logits, loss
        
        return logits
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"MLMHead(d_model={self.d_model}, "
                f"text_vocab_size={self.text_vocab_size}, "
                f"dropout={self.dropout_p})")


class MCMHead(nn.Module):
    """
    Masked Category Modeling head for categorical test values.
    
    Predicts masked categorical test values using cross-entropy loss.
    Applied to enhanced tabular embeddings after UniTransformerLayer processing.
    """
    
    def __init__(
        self,
        d_model: int,
        cat_vocab_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize MCM head.
        
        Args:
            d_model: Input embedding dimension
            cat_vocab_size: Size of categorical embedder vocabulary
            dropout: Dropout probability before projection
        """
        super().__init__()
        
        self.d_model = d_model
        self.cat_vocab_size = cat_vocab_size
        self.dropout_p = dropout
        
        # Standard head architecture
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, cat_vocab_size)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through MCM head.
        
        Args:
            embeddings: Input embeddings [B, new_seq_len, d_model]
            labels: Optional MCM labels [B, new_seq_len] with -100 for non-masked positions
            
        Returns:
            If labels provided: (logits, loss)
            Otherwise: logits
            
            logits: [B, new_seq_len, cat_vocab_size]
            loss: scalar tensor
        """
        # Apply layer norm and dropout
        x = self.layer_norm(embeddings)
        x = self.dropout(x)
        
        # Project to vocabulary size
        logits = self.projection(x)  # [B, new_seq_len, cat_vocab_size]
        
        # Compute loss if labels provided
        if labels is not None:
            # Flatten for cross-entropy computation with NaN defense
            flat_logits = logits.view(-1, self.cat_vocab_size)
            flat_labels = labels.view(-1)
            valid = (flat_labels != -100).sum()
            
            if valid == 0:                      # The entire batch has no masked tokens
                loss = flat_logits.new_zeros([])  # Scalar 0, retains device and dtype
            else:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction='sum'             # First sum over valid tokens
                ) / valid                       # Then manually divide by the count, equivalent to "mean"
            
            return logits, loss
        
        return logits
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"MCMHead(d_model={self.d_model}, "
                f"cat_vocab_size={self.cat_vocab_size}, "
                f"dropout={self.dropout_p})")


class CVRHead(nn.Module):
    """
    Cell Value Retrieval head for text test values.
    
    Retrieval task requiring models to select correct text content from a candidate
    pool using InfoNCE loss. Applied to enhanced tabular embeddings after
    UniTransformerLayer processing.
    
    For each masked text cell, the model must identify the correct original content
    from a pool of shuffled candidates using semantic similarity.
    """
    
    def __init__(
        self,
        d_model: int,
        temperature: float = 0.1,
        normalize: bool = True
    ):
        """
        Initialize CVR head.
        
        Args:
            d_model: Input embedding dimension
            temperature: Temperature scaling for InfoNCE (lower = sharper)
            normalize: Whether to L2-normalize embeddings before similarity
        """
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        cvr_mask: torch.Tensor,
        cvr_candidates: torch.Tensor,
        cvr_labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through CVR head.
        
        Args:
            embeddings: Input embeddings [B, new_seq_len, d_model]
            cvr_mask: CVR position mask [B, new_seq_len] with 1 at query positions
            cvr_candidates: Candidate embeddings [n_cvr, d_model] (frozen)
            cvr_labels: InfoNCE target labels [n_cvr] for correct candidate indices
            
        Returns:
            If labels provided: (similarity_matrix, loss)
            Otherwise: similarity_matrix
            
            similarity_matrix: [n_cvr, n_cvr] - query-candidate similarities
            loss: scalar InfoNCE loss
        """
        # Handle edge case: no CVR samples in batch
        if cvr_candidates.size(0) == 0:
            device = embeddings.device
            empty_sim = torch.empty(0, 0, device=device)
            if cvr_labels is not None:
                empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
                return empty_sim, empty_loss
            return empty_sim
        
        # Extract query embeddings using CVR mask
        query_positions = (cvr_mask == 1)  # Boolean mask [B, new_seq_len]
        query_embeddings = embeddings[query_positions]  # [n_cvr, d_model]
        
        # Optional L2 normalization for better InfoNCE performance
        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            cvr_candidates = F.normalize(cvr_candidates.detach(), p=2, dim=-1)
        else:
            cvr_candidates = cvr_candidates.detach()
        
        # Compute similarity matrix with temperature scaling
        # similarity[i, j] = sim(query_i, candidate_j)
        similarity_matrix = torch.matmul(query_embeddings, cvr_candidates.T) / self.temperature
        # Shape: [n_cvr, n_cvr]
        
        # Compute InfoNCE loss if labels provided
        if cvr_labels is not None:
            # Each query should match its corresponding candidate
            # cvr_labels[i] gives the candidate index for query i
            loss = F.cross_entropy(similarity_matrix, cvr_labels)
            return similarity_matrix, loss
        
        return similarity_matrix
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"CVRHead(d_model={self.d_model}, "
                f"temperature={self.temperature}, "
                f"normalize={self.normalize})")


class MCCHead(nn.Module):
    """
    Multiple-Choice Cloze head for numerical test values.
    
    Contrastive learning task where models select the correct numerical value from 
    K candidates (1 true + K-1 distractors) using InfoNCE loss. Applied to enhanced 
    tabular embeddings after UniTransformerLayer processing.
    
    For each masked numerical cell, the model must identify the correct value from
    K candidates using semantic similarity in the learned embedding space.
    """
    
    def __init__(
        self,
        d_model: int,
        temperature: float = 1.0,
        normalize: bool = True
    ):
        """
        Initialize MCC head.
        
        Args:
            d_model: Input embedding dimension
            temperature: Temperature scaling for InfoNCE (higher = softer, easier task)
            normalize: Whether to L2-normalize embeddings before similarity
        """
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        mcc_mask: torch.Tensor,
        mcc_candidates: torch.Tensor,
        mcc_labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through MCC head.
        
        Args:
            embeddings: Input embeddings [B, new_seq_len, d_model]
            mcc_mask: MCC position mask [B, new_seq_len] with True at query positions
            mcc_candidates: Candidate embeddings [n_mcc, K, d_model] (frozen)
            mcc_labels: InfoNCE target labels [n_mcc] for correct choice indices [0, K-1]
            
        Returns:
            If labels provided: (similarity_matrix, loss)
            Otherwise: similarity_matrix
            
            similarity_matrix: [n_mcc, K] - query-candidate similarities
            loss: scalar InfoNCE loss
        """
        # Handle edge case: no MCC samples in batch
        if mcc_candidates.size(0) == 0:
            device = embeddings.device
            # Return empty similarity matrix with correct shape [0, K]
            K = mcc_candidates.shape[1] if mcc_candidates.ndim == 3 else 5  # fallback
            empty_sim = torch.empty(0, K, device=device)
            if mcc_labels is not None:
                empty_loss = torch.tensor(0.0, device=device, requires_grad=True)
                return empty_sim, empty_loss
            return empty_sim
        
        # Extract query embeddings using MCC mask
        query_positions = (mcc_mask == True)  # Boolean mask [B, new_seq_len]
        query_embeddings = embeddings[query_positions]  # [n_mcc, d_model]
        
        n_mcc, K, d_model = mcc_candidates.shape
        
        # Ensure we have the expected number of queries
        assert query_embeddings.shape[0] == n_mcc, (
            f"Query count mismatch: got {query_embeddings.shape[0]} queries, "
            f"expected {n_mcc} from candidates"
        )
        
        # Optional L2 normalization for better InfoNCE performance
        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)  # [n_mcc, d_model]
            mcc_candidates = F.normalize(mcc_candidates.detach(), p=2, dim=-1)  # [n_mcc, K, d_model]
        else:
            mcc_candidates = mcc_candidates.detach()  # Prevent gradient leakage
        
        # Compute similarity matrix with temperature scaling
        # For each query i, compute similarity with its K candidates
        # similarity[i, j] = sim(query_i, candidate_i[j])
        
        # Vectorized computation: einsum for batch dot product
        similarity_matrix = torch.einsum('nd,nkd->nk', query_embeddings, mcc_candidates)
        # Shape: [n_mcc, K]
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute InfoNCE loss if labels provided
        if mcc_labels is not None:
            # Each query should match its corresponding correct candidate
            # mcc_labels[i] gives the correct choice index [0, K-1] for query i
            loss = F.cross_entropy(similarity_matrix, mcc_labels)
            return similarity_matrix, loss
        
        return similarity_matrix
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"MCCHead(d_model={self.d_model}, "
                f"temperature={self.temperature}, "
                f"normalize={self.normalize})")


class CPCHead(nn.Module):
    """
    Contrastive Predictive Coding head for individual-level exam sequence modeling.
    
    Implements CPC objective where contextual representations predict future exam
    embeddings using InfoNCE loss with cross-patient negative sampling.
    
    For each individual with E_i exams, sample position t ∈ [2, E_i]:
    - Query: c_(t-1) from IndCausalTransformer (contextual prediction)
    - Positive: x_t from pre-causal embeddings (actual future exam)  
    - Negatives: All exam embeddings from other individuals (cross-patient contrasts)
    
    Uses learned projection heads to map embeddings into a dedicated contrastive space
    for stable optimization and task-specific representations.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        proj_dim: int = 128,
        temperature: float = 0.1,
        min_negatives: int = 1
    ):
        """
        Initialize CPC head.
        
        Args:
            d_model: Input embedding dimension
            proj_dim: Projection dimension for contrastive space
            temperature: Temperature scaling for InfoNCE loss
            min_negatives: Minimum guaranteed number of negatives per query
        """
        super().__init__()
        
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.min_negatives = min_negatives
        
        # Learned projection heads for contrastive space
        self.q_proj = nn.Linear(d_model, proj_dim, bias=False)  # Query projection
        self.k_proj = nn.Linear(d_model, proj_dim, bias=False)  # Key projection
        
        # InfoNCE loss with paired negatives (each query has personalized negative set)
        self.info_nce = InfoNCE(
            temperature=temperature,
            reduction='mean',
            negative_mode='paired'
        )
    
    def forward(
        self,
        pre_causal_emb: torch.Tensor,    # [B', E_max, D] - x embeddings before causal transformer
        post_causal_emb: torch.Tensor,   # [B', E_max, D] - c embeddings after causal transformer  
        attention_mask: torch.Tensor,    # [B', E_max] - valid position mask
        segment_lengths: List[int]       # actual exam counts per individual
    ) -> torch.Tensor:
        """
        Compute CPC loss using InfoNCE.
        
        Args:
            pre_causal_emb: Individual exam embeddings before causal modeling
            post_causal_emb: Individual exam embeddings after causal modeling
            attention_mask: Mask indicating valid exam positions
            segment_lengths: Number of actual exams per individual
            
        Returns:
            CPC loss tensor (scalar)
        """
        # Compute safe number of negatives for this batch
        num_negatives = self._compute_safe_num_negatives(segment_lengths)
        
        if num_negatives == 0:
            # Not enough data for CPC, return zero loss with gradient
            return torch.tensor(0.0, device=pre_causal_emb.device, requires_grad=True)
        
        # Prepare CPC samples (queries, positives, negatives)
        queries, positives, negatives = self._prepare_cpc_samples(
            pre_causal_emb, post_causal_emb, attention_mask, segment_lengths, num_negatives
        )
        
        if queries.size(0) == 0:
            # No valid CPC pairs in this batch
            return torch.tensor(0.0, device=pre_causal_emb.device, requires_grad=True)
        
        # Compute InfoNCE loss
        cpc_loss = self.info_nce(queries, positives, negatives)
        
        return cpc_loss
    
    def _compute_safe_num_negatives(self, segment_lengths: List[int]) -> int:
        """
        Compute safe number of negatives that all individuals can provide.
        
        Formula: M_min = total_exams - max_individual_exams
                 Safe M = max(min_negatives, M_min - 1)
        
        Args:
            segment_lengths: Number of exams per individual
            
        Returns:
            Safe number of negatives per query
        """
        if len(segment_lengths) <= 1:
            return 0  # Need at least 2 individuals for cross-patient contrast
        
        total_exams = sum(segment_lengths)
        max_individual_exams = max(segment_lengths)
        M_min = total_exams - max_individual_exams
        
        # Apply safety margin and ensure minimum
        safe_M = max(self.min_negatives, M_min - 1)
        return safe_M
    
    def _prepare_cpc_samples(
        self,
        pre_causal_emb: torch.Tensor,
        post_causal_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_lengths: List[int],
        num_negatives: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample CPC training pairs and construct InfoNCE inputs.
        
        Args:
            pre_causal_emb: [B', E_max, D] - exam embeddings before causal modeling
            post_causal_emb: [B', E_max, D] - exam embeddings after causal modeling
            attention_mask: [B', E_max] - valid position mask
            segment_lengths: Number of actual exams per individual  
            num_negatives: Fixed number of negatives per query
            
        Returns:
            Tuple of:
            - queries: [N_valid, proj_dim] - projected c_(t-1) contextual embeddings
            - positives: [N_valid, proj_dim] - projected x_t target exam embeddings
            - negatives: [N_valid, num_negatives, proj_dim] - projected negative embeddings
        """
        device = pre_causal_emb.device
        queries, positives = [], []
        
        # Track original batch indices for individuals with valid exams
        valid_idxs = [i for i, length in enumerate(segment_lengths) if length > 0]
        individual_exams = [pre_causal_emb[i, :length] for i, length in enumerate(segment_lengths) if length > 0]
        
        # Sample queries and positives from each individual
        valid_individuals = []
        for i, length in enumerate(segment_lengths):
            if length > 1:  # Need at least 2 exams for CPC (t ∈ [2, E_i])
                # Uniform sampling of t from [2, length] (1-indexed)
                t = torch.randint(2, length + 1, (1,)).item()
                
                # Query: c_(t-1) from post-causal (0-indexed: t-2)
                query = post_causal_emb[i, t - 2]  # c_(t-1)
                
                # Positive: x_t from pre-causal (0-indexed: t-1)  
                positive = pre_causal_emb[i, t - 1]  # x_t
                
                queries.append(query)
                positives.append(positive)
                valid_individuals.append(i)
        
        if not queries:
            # No valid individuals for CPC
            empty_tensor = torch.empty(0, self.proj_dim, device=device)
            return empty_tensor, empty_tensor, torch.empty(0, num_negatives, self.proj_dim, device=device)
        
        # Construct personalized negatives for each query (FIXED negative sampling)
        negatives_list = []
        for idx, individual_idx in enumerate(valid_individuals):
            # Collect negatives from all OTHER individuals (fixed index mapping)
            negatives_for_i = []
            for j_idx, j in enumerate(valid_idxs):
                if j != individual_idx:  # Now comparing original batch indices correctly
                    negatives_for_i.append(individual_exams[j_idx])
            
            if negatives_for_i:
                all_negatives = torch.cat(negatives_for_i, dim=0)  # [M_available, D]
                
                # Sample exactly num_negatives
                if all_negatives.size(0) >= num_negatives:
                    # Sample without replacement
                    indices = torch.randperm(all_negatives.size(0), device=device)[:num_negatives]
                    sampled_negatives = all_negatives[indices]  # [num_negatives, D]
                else:
                    # Sample with replacement if not enough negatives
                    indices = torch.randint(
                        0, all_negatives.size(0), (num_negatives,), device=device
                    )
                    sampled_negatives = all_negatives[indices]  # [num_negatives, D]
                
                negatives_list.append(sampled_negatives)
            else:
                # Edge case: no other individuals (shouldn't happen if num_negatives > 0)
                zero_negatives = torch.zeros(num_negatives, self.d_model, device=device)
                negatives_list.append(zero_negatives)
        
        # Stack raw embeddings
        queries_raw = torch.stack(queries)      # [N_valid, D]
        positives_raw = torch.stack(positives)  # [N_valid, D]  
        negatives_raw = torch.stack(negatives_list)  # [N_valid, num_negatives, D]
        
        # Apply learned projections to contrastive space
        queries_proj = self.q_proj(queries_raw)  # [N_valid, proj_dim]
        positives_proj = self.k_proj(positives_raw)  # [N_valid, proj_dim]
        
        # Project negatives: [N_valid, num_negatives, D] -> [N_valid, num_negatives, proj_dim]
        N_valid = negatives_raw.size(0)
        negatives_proj = self.k_proj(negatives_raw.view(-1, self.d_model))  # [N_valid * num_negatives, proj_dim]
        negatives_proj = negatives_proj.view(N_valid, num_negatives, self.proj_dim)  # [N_valid, num_negatives, proj_dim]
        
        return queries_proj, positives_proj, negatives_proj
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"CPCHead(d_model={self.d_model}, "
                f"proj_dim={self.proj_dim}, "
                f"temperature={self.temperature}, "
                f"min_negatives={self.min_negatives})")
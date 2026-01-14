# src/models/medical_ssl_model.py

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .embedders import TabEmbedder, SinusoidalPositionalEncoding, TimeAwarePositionalEmbedding
from .transformers import BiCrossAttLayer, UniTransformerLayer, IndCausalTransformer
from .fusion import TextCompressor, ImportanceWeightedConcat
from .utils import prepare_individual_sequences


@dataclass
class ModelOutputs:
    """Container for model outputs needed for losses and downstream tasks."""
    # For loss computation
    mlm_embeddings: torch.Tensor           # Enhanced result text [B, result_seq_len, D]
    mcm_embeddings: torch.Tensor           # Unified embeddings [B, new_seq_len, D]
    cvr_embeddings: torch.Tensor           # Same as mcm_embeddings
    mcc_embeddings: torch.Tensor           # Same as mcm_embeddings
    pre_causal_emb: torch.Tensor           # Before IndCausal [B', E_max, D]
    post_causal_emb: torch.Tensor          # After IndCausal [B', E_max, D]
    
    # For downstream tasks
    next_prediction: torch.Tensor          # Last embedding per individual [B', D]
    general_representation: torch.Tensor   # Mean pooled per individual [B', D]
    
    # Auxiliary data for losses
    result_attention_mask: torch.Tensor    # For MLM [B, result_seq_len]
    unified_attention_mask: torch.Tensor   # For MCM [B, new_seq_len]
    individual_attention_mask: torch.Tensor # For CPC [B', E_max]
    segment_lengths: List[int]             # For CPC negative sampling
    
    # For monitoring (from model components)
    fusion_weights: Optional[Tuple[float, float]] = None
    
    # Original batch metadata
    batch_metadata: Optional[Dict[str, Any]] = None


class MedicalSSLModel(nn.Module):
    """
    Multi-modal SSL model for medical examination data.
    
    Orchestrates all components in the correct order and returns
    outputs needed for all five training objectives.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        text_vocab_size: int,
        cat_vocab_size: int,
        device: str = "cpu"
    ):
        """
        Initialize MedicalSSLModel.
        
        Args:
            config: Model configuration
            text_vocab_size: Size of text vocabulary (for embeddings)
            cat_vocab_size: Size of categorical vocabulary
            device: Device for model
        """
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Extract key dimensions
        self.d_model = config.get('d_model', 768)
        self.n_cross_layers = config.get('n_cross_layers', 2)
        self.n_uni_layers = config.get('n_uni_layers', 10)
        self.n_ind_layers = config.get('n_ind_layers', 2)
        
        # TabEmbedder
        self.tab_embedder = TabEmbedder(
            D=self.d_model,
            tiny_text_config=config.get('tiny_text_config', None),
            device=device
        )
        
        # Positional encoding for result text
        self.result_pos_encoding = SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_len=config.get('max_result_len', 1024),
            dropout=config.get('pos_dropout', 0.1),
            device=device
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            BiCrossAttLayer(
                d_model=self.d_model,
                n_heads=config.get('cross_n_heads', 8),
                mlp_ratio=config.get('cross_mlp_ratio', 2.0),
                attn_dropout=config.get('attn_dropout', 0.1),
                mlp_dropout=config.get('mlp_dropout', 0.1),
                proj_dropout=config.get('proj_dropout', 0.1),
                learnable_fusion=config.get('learnable_fusion', True),
                asymmetric_fusion=config.get('asymmetric_fusion', False),
                fusion_d_gate=config.get('fusion_d_gate', 64),
                fusion_tau=config.get('fusion_tau', 1.0),
                presence_logit_mask=config.get('presence_logit_mask', True)
            )
            for _ in range(self.n_cross_layers)
        ])
        
        # Text compressor
        self.text_compressor = TextCompressor(
            d_model=self.d_model,
            n_out=config.get('text_compress_tokens', 50),
            n_heads=config.get('compress_n_heads', 8),
            dropout=config.get('compress_dropout', 0.1)
        )
        
        # Importance-weighted concatenation
        self.importance_concat = ImportanceWeightedConcat(
            d_model=self.d_model,
            initial_tab_weight=config.get('initial_tab_weight', 2.0),
            initial_text_weight=config.get('initial_text_weight', 1.0),
            use_layernorm=True,
            use_segment_emb=True
        )
        
        # Unified transformer layers
        self.unified_layers = nn.ModuleList([
            UniTransformerLayer(
                d_model=self.d_model,
                n_heads=config.get('uni_n_heads', 8),
                mlp_ratio=config.get('uni_mlp_ratio', 4.0),
                dropout=config.get('uni_dropout', 0.1)
            )
            for _ in range(self.n_uni_layers)
        ])
        
        # Time-aware positional embedding for individuals
        self.time_pos_embedding = TimeAwarePositionalEmbedding(
            d_model=self.d_model,
            time_scale=config.get('time_scale', 365.0),
            dropout=config.get('time_pos_dropout', 0.1),
            device=device
        )
        
        # Individual causal transformer
        self.ind_causal_transformer = IndCausalTransformer(
            d_model=self.d_model,
            n_heads=config.get('ind_n_heads', 4),
            mlp_ratio=config.get('ind_mlp_ratio', 2.0),
            n_layers=self.n_ind_layers,
            dropout=config.get('ind_dropout', 0.1)
        )
    
    def forward(
        self,
        batch_data: Dict[str, Any],
        code_embedder,
        num_embedder,
        text_embedder
    ) -> ModelOutputs:
        """
        Forward pass through the entire model.
        
        Args:
            batch_data: Output from collate_exams
            code_embedder: Categorical embedder
            num_embedder: Numerical embedder  
            text_embedder: Text embedder
            
        Returns:
            ModelOutputs with all necessary tensors for losses and downstream tasks
        """
        # Step 1: TabEmbedder
        final_emb, final_mask, expanded_labels, result_emb = self.tab_embedder(
            code_embedder, num_embedder, text_embedder, batch_data
        )

        # Write expanded_labels back to batch_data for trainer access
        batch_data["expanded_labels"] = expanded_labels
        
        # Step 2: Add positional encoding to result text
        result_emb = self.result_pos_encoding(result_emb)
        
        # Step 3: Cross-attention layers
        tab_enhanced = final_emb
        text_enhanced = result_emb
        
        for i, cross_layer in enumerate(self.cross_attention_layers):
            tab_enhanced, text_enhanced = cross_layer(
                tab_enhanced, text_enhanced,
                final_mask, batch_data["result_attention_mask"]
            )
        
        # MLM embeddings are the enhanced result text
        mlm_embeddings = text_enhanced
        
        # Step 4: Text compression
        compressed_text, compressed_mask = self.text_compressor(
            text_enhanced, batch_data["result_attention_mask"]
        )
        
        # Step 5: Importance-weighted concatenation  
        unified_tokens, unified_mask = self.importance_concat(
            tab_enhanced, compressed_text,
            final_mask, compressed_mask
        )
        
        # Get fusion weights for monitoring
        fusion_weights = self.importance_concat.get_importance_weights()
        
        # Step 6: Unified transformer layers
        for i, uni_layer in enumerate(self.unified_layers):
            unified_tokens = uni_layer(unified_tokens, unified_mask)
        
        # MCM, CVR and MCC embeddings are from unified tokens
        # Extract only tabular portion
        L_tab = final_mask.size(1)  # Tabular sequence length
        mcm_embeddings = unified_tokens[:, :L_tab]  # [B, L_tab, D]
        cvr_embeddings = unified_tokens[:, :L_tab]  # [B, L_tab, D]
        mcc_embeddings = unified_tokens[:, :L_tab]  # [B, L_tab, D]
        
        # Step 7: Mean pooling for individual sequences
        # Mask out padding positions
        masked_unified = unified_tokens * unified_mask.unsqueeze(-1).float()
        denom = unified_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled_embeddings = masked_unified.sum(dim=1) / denom  # [B, D]
        
        # Step 8: Prepare individual sequences
        individual_emb, positions, time_intervals, ind_attention_mask = prepare_individual_sequences(
            pooled_embeddings,
            batch_data["exam_dates"],
            batch_data["segment_lengths"]
        )
        
        # Step 9: Add time-aware positional embeddings
        time_embeddings = self.time_pos_embedding(positions, time_intervals, ind_attention_mask)
        individual_emb = individual_emb + time_embeddings
        
        # Store pre-causal embeddings for CPC
        pre_causal_emb = individual_emb.clone()
        
        # Step 10: Individual causal transformer
        post_causal_emb = self.ind_causal_transformer(individual_emb, ind_attention_mask)
        
        # Step 11: Extract downstream representations
        # Next prediction: last valid embedding per individual
        next_prediction = []
        for i, length in enumerate(batch_data["segment_lengths"]):
            if length > 0:
                next_prediction.append(post_causal_emb[i, length - 1])
            else:
                # Shouldn't happen, but safety check
                next_prediction.append(torch.zeros(self.d_model, device=post_causal_emb.device))
        next_prediction = torch.stack(next_prediction) if next_prediction else torch.empty(0, self.d_model, device=post_causal_emb.device)
        
        # General representation: mean pooling over valid positions
        masked_post_causal = post_causal_emb * ind_attention_mask.unsqueeze(-1).float()
        denom = ind_attention_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        general_representation = masked_post_causal.sum(dim=1) / denom
        
        # Build outputs
        outputs = ModelOutputs(
            # For loss computation
            mlm_embeddings=mlm_embeddings,
            mcm_embeddings=mcm_embeddings,
            cvr_embeddings=cvr_embeddings,
            mcc_embeddings=mcc_embeddings,
            pre_causal_emb=pre_causal_emb,
            post_causal_emb=post_causal_emb,
            
            # For downstream tasks
            next_prediction=next_prediction,
            general_representation=general_representation,
            
            # Auxiliary data for losses
            result_attention_mask=batch_data["result_attention_mask"],
            unified_attention_mask=unified_mask,
            individual_attention_mask=ind_attention_mask,
            segment_lengths=batch_data["segment_lengths"],
            
            # For monitoring
            fusion_weights=fusion_weights,
            
            # Original batch metadata
            batch_metadata=batch_data
        )
        
        return outputs
    
    def get_trainable_params(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get trainable parameters grouped by component for differential learning rates.
        
        Returns:
            Dictionary mapping component names to parameter lists
        """
        params = {
            'tab_embedder': list(self.tab_embedder.parameters()),
            'cross_attention': list(self.cross_attention_layers.parameters()),
            'text_compressor': list(self.text_compressor.parameters()),
            'importance_concat': list(self.importance_concat.parameters()),
            'unified_layers': list(self.unified_layers.parameters()),
            'time_embedding': list(self.time_pos_embedding.parameters()),
            'ind_causal': list(self.ind_causal_transformer.parameters()),
        }
        
        return params

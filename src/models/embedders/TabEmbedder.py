# src/models/embedders/TabEmbedder.py

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
from src.models.transformers import TinyTextTransformer
from src.models.embedders.CategoricalEmbedder import CategoricalEmbedder
from src.models.embedders.NumericalEmbedder import NumericalEmbedder
from src.models.embedders.TextEmbedder import TextEmbedder


class TabEmbedder(nn.Module):
    """
    Tabular data embedder that processes multi-modal health examination data.
    
    Combines categorical, numerical, and text embeddings into a unified sequence
    representation with special tokens for downstream transformer processing.
    
    Output sequence structure: [CLS] gender age [SEP] test1 [SEP] test2 [SEP] ...
    """
    
    def __init__(
        self,
        D: int = 768,
        tiny_text_config: Dict = None,
        device: str = "cpu"
    ):
        """
        Initialize TabEmbedder.
        
        Args:
            D: Target embedding dimension for all modalities
            tiny_text_config: Configuration for TinyTextTransformer
            device: Device for computations
        """
        super().__init__()
        
        self.D = D
        self.device = device
        
        # Embedding tables for demographics and types (sized for actual ID ranges)
        self.age_embedding = nn.Embedding(9, D)      # 0-8 (0=missing, 1=<20, ..., 8=>=80)
        self.gender_embedding = nn.Embedding(3, D)   # 0-2 (0=missing, 1=M, 2=F)
        self.type_embedding = nn.Embedding(4, D)     # 0-3 (0=pad, 1=PQ, 2=CD/CO, 3=ST)
        
        # Special token embeddings
        self.cls_embedding = nn.Parameter(torch.randn(D))
        self.sep_embedding = nn.Parameter(torch.randn(D))
        
        # MCC mask embedding for masked numerical values
        self.mcc_mask_embedding = nn.Parameter(torch.randn(D))
        
        # TinyTextTransformer for processing text test values
        if tiny_text_config is None:
            tiny_text_config = {
                'd_model': 768,
                'nhead': 4,
                'd_ff': 1536,
                'n_layers': 2,
                'D_out': D,
                'dropout': 0.1
            }
        self.tiny_text_transformer = TinyTextTransformer(**tiny_text_config)
        
        # Result text dimension projection (768 → D)
        self.result_proj = nn.Linear(768, D)
        
        # Move to device
        if device != "cpu":
            self.to(device)
    
    def _get_computation_dtype(self) -> torch.dtype:
        """
        Get appropriate dtype for current computation context.
        
        Returns:
            torch.float16 if under autocast, torch.float32 otherwise
        """
        try:
            return torch.float16 if torch.is_autocast_enabled() else torch.float32
        except Exception as e:
            # Fallback to float32 if autocast detection fails
            print(f"Warning: Autocast detection failed ({e}), using float32")
            return torch.float32
    
    def forward(
        self,
        code_embedder: CategoricalEmbedder,
        num_embedder: NumericalEmbedder,
        text_embedder: TextEmbedder,
        batch_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through TabEmbedder.
        
        Args:
            code_embedder: Categorical embedder for codes and categorical values
            num_embedder: Numerical embedder for numerical values
            text_embedder: Text embedder for text values and result text
            batch_data: Dictionary from collate_exams containing all batch data
            
        Returns:
            Tuple of:
            - final_emb: [B, new_seq_len, D] - Final sequence embeddings
            - final_mask: [B, new_seq_len] - Final attention mask
            - expanded_labels: Dict with expanded training objective tensors
            - result_emb: [B, result_seq_len, D] - Result text embeddings
        """
        device = next(self.parameters()).device
        
        # Extract batch information
        B = batch_data["B"]
        T_max = batch_data["T_max"]
        new_seq_len = 4 + 2 * T_max
        
        # 1. Get individual embeddings
        
        # Code embeddings
        code_emb = code_embedder.embed(batch_data["code_ids"])  # [B, T_max, D]
        
        # Categorical value embeddings (use masked inputs for MCM)
        cat_emb = code_embedder.embed(batch_data["mcm_inputs"])  # [B, T_max, D]
        
        # Numerical embeddings
        num_emb = num_embedder(batch_data["num_values"])  # [B, T_max, D]

        # Handle MCC masking - replace -999.0 positions with learned mask embedding
        mcc_mask_positions = (batch_data["num_values"] == -999.0)
        if mcc_mask_positions.any():
            num_emb = num_emb.clone()  # Avoid modifying leaf tensor in-place
            mask_emb = self.mcc_mask_embedding.to(dtype=num_emb.dtype, device=num_emb.device)
            num_emb[mcc_mask_positions] = mask_emb
        
        # 2. Process text test values
        text_emb_tab = self._process_text_values(
            text_embedder, 
            batch_data["text_token_ids"],
            batch_data["text_attention_mask"], 
            batch_data["text_locations"],
            B, T_max
        )  # [B, T_max, D]

        # 2.5. Process CVR candidates
        cvr_candidates = self._process_cvr_candidates(
            batch_data, text_embedder, code_embedder
        )  # [n_cvr, D]

        # 2.6. Process MCC candidates
        mcc_candidates = self._process_mcc_candidates(
            batch_data, num_embedder, code_embedder
        )  # [n_mcc, K, D]
        
        # 3. Combine embeddings with masks
        tab_emb = self._combine_embeddings(
            code_emb, cat_emb, num_emb, text_emb_tab,
            batch_data["mask_code"], batch_data["mask_cat"],
            batch_data["mask_num"], batch_data["mask_text"]
        )  # [B, T_max, D]
        
        # Add type embeddings
        type_emb = self.type_embedding(batch_data["type_ids"])  # [B, T_max, D]
        tab_emb = tab_emb + type_emb
        
        # 4. Build final sequence with special tokens
        final_emb, final_mask = self._build_final_sequence(
            tab_emb, 
            batch_data["mask_code"],
            batch_data["exam_ages"],
            batch_data["exam_genders"]
        )  # [B, new_seq_len, D], [B, new_seq_len]
        
        # 5. Expand training objective tensors
        expanded_labels = self._expand_training_tensors(
            batch_data["mcm_labels"], 
            batch_data["cvr_mask"],
            batch_data["mcc_mask"],
            T_max
        )
        
        # Add CVR data to expanded_labels
        expanded_labels["cvr_candidates"] = cvr_candidates
        expanded_labels["cvr_labels"] = batch_data["cvr_labels"]

        # Add MCC data to expanded_labels
        expanded_labels["mcc_candidates"] = mcc_candidates
        expanded_labels["mcc_labels"] = batch_data["mcc_labels"]
        
        # 6. Process result text
        result_emb = self._process_result_text(
            text_embedder,
            batch_data["result_input_ids"],
            batch_data["result_attention_mask"]
        )  # [B, result_seq_len, D]
        
        return final_emb, final_mask, expanded_labels, result_emb
    
    def _process_text_values(
        self,
        text_embedder: TextEmbedder,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_locations: torch.Tensor,
        B: int,
        T_max: int
    ) -> torch.Tensor:
        """Process text test values through TinyTextTransformer and scatter back."""
        # FIX: Handle edge case with proper dtype and device detection
        if text_token_ids.numel() == 0:
            # Scenario 2: No upstream tensor - use autocast detection
            dtype = self._get_computation_dtype()
            device = text_token_ids.device  # Use upstream tensor device even when empty
            return torch.zeros(B, T_max, self.D, device=device, dtype=dtype)
        
        # Get text embeddings from TextEmbedder
        text_emb = text_embedder.embed(text_token_ids)  # [N_seq, L, 768]
        
        # Process through TinyTextTransformer
        text_emb_agg = self.tiny_text_transformer(text_emb, text_attention_mask)  # [N_seq, D]
        
        # FIX: Create tensor with dtype and device matching the computed embeddings
        # Scenario 1: Have upstream tensor - use its dtype and device
        text_emb_tab = torch.zeros(B, T_max, self.D, device=text_token_ids.device, dtype=text_emb_agg.dtype)
        if text_locations.numel() > 0:
            batch_idxs = text_locations[:, 0]
            test_idxs = text_locations[:, 1]
            text_emb_tab[batch_idxs, test_idxs] = text_emb_agg
        
        return text_emb_tab

    def _process_cvr_candidates(
        self,
        batch_data: Dict[str, Any],
        text_embedder: TextEmbedder,
        code_embedder: CategoricalEmbedder
    ) -> torch.Tensor:
        """
        Process CVR candidates with no_grad and add type/code embeddings.
        
        Args:
            batch_data: Batch data containing CVR information
            text_embedder: Text embedder for processing tokens
            code_embedder: Code embedder for code embeddings
            
        Returns:
            Candidate embeddings [n_cvr, D] ready for InfoNCE
        """
        cvr_true_ids = batch_data["cvr_true_ids"]
        cvr_true_attention_masks = batch_data["cvr_true_attention_masks"]
        
        if len(cvr_true_ids) == 0:
            # Handle zero-sample case
            device = next(self.parameters()).device
            dtype = self._get_computation_dtype()
            return torch.empty(0, self.D, device=device, dtype=dtype)
        
        # Vectorized lookup for efficiency
        cvr_positions = torch.nonzero(batch_data["cvr_mask"] == 1)  # [n_cvr, 2]
        batch_indices = cvr_positions[:, 0]  # [n_cvr]
        cell_indices = cvr_positions[:, 1]   # [n_cvr]
        
        # Get type and code IDs for candidates
        candidate_types = batch_data["type_ids"][batch_indices, cell_indices]  # [n_cvr]
        candidate_codes = batch_data["code_ids"][batch_indices, cell_indices]  # [n_cvr]
        
        # Process candidates with full no_grad to prevent gradient leakage
        with torch.no_grad():
            # Get text embeddings from TextEmbedder
            candidate_text_emb = text_embedder.embed(cvr_true_ids)  # [n_cvr, L, 768]
            
            # Process through TinyTextTransformer
            candidate_emb = self.tiny_text_transformer(candidate_text_emb, cvr_true_attention_masks)  # [n_cvr, D]
            
            # Align dtype & device to avoid AMP FP16↔FP32 mismatch
            aligned = {"dtype": candidate_emb.dtype, "device": candidate_emb.device}
            type_emb = self.type_embedding(candidate_types).to(**aligned)  # [n_cvr, D]
            code_emb = code_embedder.embed(candidate_codes).to(**aligned)  # [n_cvr, D]
            candidate_emb = (candidate_emb + type_emb + code_emb).detach()
        
        return candidate_emb

    def _process_mcc_candidates(
        self,
        batch_data: Dict[str, Any],
        num_embedder: NumericalEmbedder,
        code_embedder: CategoricalEmbedder
    ) -> torch.Tensor:
        """
        Process MCC candidates through numerical embedder.
        
        Args:
            batch_data: Batch data containing MCC information
            num_embedder: Numerical embedder for processing values
            code_embedder: Categorical embedder for code embeddings
            
        Returns:
            Candidate embeddings [n_mcc, K, D] ready for InfoNCE
        """
        opts_raw = batch_data["opts_raw"]  # [n_mcc, K]
        
        if opts_raw.numel() == 0:
            # Handle empty case - get K from shape
            K = opts_raw.shape[1] if opts_raw.ndim > 1 else 5  # fallback
            device = next(self.parameters()).device
            dtype = self._get_computation_dtype()
            return torch.empty(0, K, self.D, device=device, dtype=dtype)
        
        # Reshape for batch processing: [n_mcc, K] -> [n_mcc*K]
        n_mcc, K = opts_raw.shape
        opts_flat = opts_raw.view(-1)  # [n_mcc*K]
        
        # Process through numerical embedder
        opts_emb_flat = num_embedder(opts_flat)  # [n_mcc*K, D]
        
        # Reshape back: [n_mcc*K, D] -> [n_mcc, K, D]
        opts_emb = opts_emb_flat.view(n_mcc, K, -1)  # [n_mcc, K, D]

        # Get context information for each masked position (same as CVR pattern)
        mcc_positions = torch.nonzero(batch_data["mcc_mask"] == 1)  # [n_mcc, 2]
        batch_indices = mcc_positions[:, 0]  # [n_mcc]
        cell_indices = mcc_positions[:, 1]   # [n_mcc]
        
        # Get type and code IDs for candidates
        candidate_types = batch_data["type_ids"][batch_indices, cell_indices]  # [n_mcc]
        candidate_codes = batch_data["code_ids"][batch_indices, cell_indices]  # [n_mcc]
        
        # Add context bias but freeze gradients
        with torch.no_grad():
            # Align dtype & device to opts_emb
            aligned = {"dtype": opts_emb.dtype, "device": opts_emb.device}
            type_emb = self.type_embedding(candidate_types).to(**aligned)  # [n_mcc, D]
            code_emb = code_embedder.embed(candidate_codes).to(**aligned)  # [n_mcc, D]
            context_bias = (type_emb + code_emb).unsqueeze(1)  # [n_mcc, 1, D]
        
        opts_emb = opts_emb + context_bias.detach()  # [n_mcc, K, D]
        
        return opts_emb
        
    
    def _combine_embeddings(
        self,
        code_emb: torch.Tensor,
        cat_emb: torch.Tensor, 
        num_emb: torch.Tensor,
        text_emb: torch.Tensor,
        mask_code: torch.Tensor,
        mask_cat: torch.Tensor,
        mask_num: torch.Tensor,
        mask_text: torch.Tensor
    ) -> torch.Tensor:
        """Combine different embedding types with appropriate masking."""
        # Apply masks and combine via element-wise addition
        tab_emb = (code_emb * mask_code.unsqueeze(-1) +
                   cat_emb * mask_cat.unsqueeze(-1) + 
                   num_emb * mask_num.unsqueeze(-1) +
                   text_emb * mask_text.unsqueeze(-1))
        
        return tab_emb
    
    def _build_final_sequence(
        self,
        tab_emb: torch.Tensor,
        mask_tab: torch.Tensor,
        exam_ages: torch.Tensor,
        exam_genders: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build final sequence: [CLS] gender age [SEP] test1 [SEP] test2 [SEP] ..."""
        B, T_max, D = tab_emb.shape
        new_seq_len = 4 + 2 * T_max
        device = tab_emb.device
        
        # FIX: Initialize output tensors with proper dtype
        # Scenario 1: Have upstream tensor - use its dtype
        final_emb = torch.zeros(B, new_seq_len, D, device=device, dtype=tab_emb.dtype)
        final_mask = torch.zeros(B, new_seq_len, dtype=torch.bool, device=device)
        
        # FIX: Use parameters directly - PyTorch handles AMP casting automatically
        # No manual .to(dtype) calls needed - PyTorch caches cast parameters under autocast
        final_emb[:, 0] = self.cls_embedding.unsqueeze(0).expand(B, -1)          # [CLS]
        final_emb[:, 1] = self.gender_embedding(exam_genders)                    # gender
        final_emb[:, 2] = self.age_embedding(exam_ages)                          # age  
        final_emb[:, 3] = self.sep_embedding.unsqueeze(0).expand(B, -1)          # [SEP]
        final_mask[:, :4] = True                                                 # prefix always valid
        
        # Vectorized interleaving of tests and separators
        test_positions = torch.arange(T_max, device=device) * 2 + 4     # 4, 6, 8, 10, ...
        sep_positions = test_positions + 1                              # 5, 7, 9, 11, ...
        
        # Set test embeddings and masks
        final_emb[:, test_positions] = tab_emb
        final_mask[:, test_positions] = mask_tab
        
        # Set separators (valid where tests are valid)
        # FIX: Use parameter directly - no manual casting needed
        final_emb[:, sep_positions] = self.sep_embedding.unsqueeze(0).unsqueeze(0).expand(B, T_max, -1)
        final_mask[:, sep_positions] = mask_tab  # SEP valid where test is valid
        
        return final_emb, final_mask
    
    def _expand_training_tensors(
        self,
        mcm_labels: torch.Tensor,
        cvr_mask: torch.Tensor,
        mcc_mask: torch.Tensor,
        T_max: int
    ) -> Dict[str, torch.Tensor]:
        """Expand training objective tensors to new sequence length."""
        B = mcm_labels.shape[0]
        new_seq_len = 4 + 2 * T_max
        device = mcm_labels.device
        
        # Test positions in new sequence
        test_positions = torch.arange(T_max, device=device) * 2 + 4
        
        # Expand mcm_labels (pad with -100 for ignored positions)
        new_mcm_labels = torch.full((B, new_seq_len), -100, dtype=mcm_labels.dtype, device=device)
        new_mcm_labels[:, test_positions] = mcm_labels
        
        # Expand cvr_mask (pad with -100 for ignored positions)
        new_cvr_mask = torch.full((B, new_seq_len), -100, dtype=cvr_mask.dtype, device=device)
        new_cvr_mask[:, test_positions] = cvr_mask

        # Expand mcc_mask (pad with False for ignored positions)
        new_mcc_mask = torch.full((B, new_seq_len), False, dtype=mcc_mask.dtype, device=device)
        new_mcc_mask[:, test_positions] = mcc_mask
        
        return {
            "mcm_labels": new_mcm_labels,
            "cvr_mask": new_cvr_mask,
            "mcc_mask": new_mcc_mask
        }
    
    def _process_result_text(
        self,
        text_embedder: TextEmbedder,
        result_input_ids: torch.Tensor,
        result_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Process result text and project to target dimension."""
        # Get embeddings from TextEmbedder
        result_emb = text_embedder.embed(result_input_ids)  # [B, result_seq_len, 768]
        
        # Project to target dimension
        # NOTE: Linear layer parameters handled automatically by AMP
        result_emb = self.result_proj(result_emb)  # [B, result_seq_len, D]
        
        return result_emb
    
    def get_config(self) -> Dict:
        """Get configuration dictionary for saving/loading."""
        return {
            "D": self.D,
            "tiny_text_config": self.tiny_text_transformer.get_config(),
            "device": self.device
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return f"TabEmbedder(D={self.D}, device='{self.device}')"
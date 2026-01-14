# src/models/embedders/TextEmbedder.py
import os
import yaml
import logging
import torch
import unicodedata
import re
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pathlib import Path

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Combined text tokenizer and embedder using JMedRoBERTa.
    
    Handles both tokenization and embedding to ensure alignment between
    the vocabulary and embedding matrix, particularly for PHI tokens.
    
    Optimized to extract only the embedding layer for efficiency while
    maintaining the ability to fine-tune the embeddings.
    
    ref: https://huggingface.co/alabnii/jmedroberta-base-sentencepiece
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "alabnii/jmedroberta-base-sentencepiece",
        max_length: int = 512,
        padding: str = "longest",
        truncation: bool = True,
        add_phi_tokens: bool = True,
        phi_patterns_path: str = "config/cleaning/v1/deidentification/phi_patterns.yaml",
        special_tokens: Optional[List[str]] = None,
        output_dir: str = "outputs/embedders/text_embedder",
        trainable: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the text embedder.
        
        Args:
            pretrained_model_name: HuggingFace model name
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            add_phi_tokens: Whether to add PHI replacement tokens
            phi_patterns_path: Path to PHI patterns config
            special_tokens: Additional special tokens beyond PHI tokens
            output_dir: Directory for saving model and metadata
            trainable: Whether embedding weights should be trainable
            device: Device for embedding computations ("cpu" or "cuda:x")
        """
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.output_dir = output_dir
        self.device = device
        self.trainable = trainable
        
        # Core model loading: use cache if available for default model
        if (pretrained_model_name == "alabnii/jmedroberta-base-sentencepiece" and 
            self._cache_exists()):
            # Load from cache (fast path)
            logger.info(f"Loading {pretrained_model_name} from cache")
            self.tokenizer, config, original_embedding_weights = self._load_from_cache()
            logger.info("✓ Model loaded from cache successfully")
        else:
            # Load from HuggingFace (current path)
            if pretrained_model_name == "alabnii/jmedroberta-base-sentencepiece":
                logger.info(f"Cache not found for {pretrained_model_name}, downloading from HuggingFace")
            else:
                logger.info(f"Loading {pretrained_model_name} from HuggingFace (non-default model)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            config = AutoConfig.from_pretrained(pretrained_model_name)
            temp_model = AutoModel.from_pretrained(pretrained_model_name)
            original_embedding_weights = temp_model.get_input_embeddings().weight.data.clone()
        
            # Clean up temporary model
            del temp_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✓ Model loaded from HuggingFace successfully")
        
        # Fix unrealistic model_max_length if needed
        if self.tokenizer.model_max_length > 1_000_000:
            self.tokenizer.model_max_length = max_length
        
        # Add PHI tokens and other special tokens if requested
        all_special_tokens = []
        if add_phi_tokens:
            phi_tokens = self._load_phi_tokens(phi_patterns_path)
            all_special_tokens.extend(phi_tokens)
            
        # Add custom special tokens if provided
        if special_tokens:
            all_special_tokens.extend(special_tokens)
            
        # Add all collected special tokens to the tokenizer
        num_added = 0
        if all_special_tokens:
            special_tokens_dict = {"additional_special_tokens": all_special_tokens}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added {num_added} special tokens. Vocabulary size: {len(self.tokenizer)}")
        self.added_special_tokens = all_special_tokens
        
        # Get vocabulary sizes
        original_vocab_size = original_embedding_weights.shape[0]
        new_vocab_size = len(self.tokenizer)  # After adding special tokens
        embedding_dim = config.hidden_size
        
        # Create a standalone embedding layer with the new size
        self.embedding_layer = torch.nn.Embedding(new_vocab_size, embedding_dim)
        
        # Initialize the embedding weights
        # Always check sizes to handle mismatches safely
        if original_vocab_size == new_vocab_size:
            # Direct copy if sizes match
            self.embedding_layer.weight.data.copy_(original_embedding_weights)
        else:
            # Copy weights for common tokens
            min_vocab_size = min(original_vocab_size, new_vocab_size)
            self.embedding_layer.weight.data[:min_vocab_size].copy_(
                original_embedding_weights[:min_vocab_size]
            )
            
            # Initialize any new token embeddings with random values
            if new_vocab_size > original_vocab_size:
                torch.nn.init.normal_(
                    self.embedding_layer.weight.data[original_vocab_size:],
                    mean=0.0, 
                    std=0.02
                )
                logger.info(f"Initialized {new_vocab_size - original_vocab_size} new token embeddings")
            elif new_vocab_size < original_vocab_size:
                logger.warning(
                    f"Tokenizer vocabulary size ({new_vocab_size}) is smaller than "
                    f"model embedding size ({original_vocab_size}). "
                    f"Some pretrained embeddings will be discarded."
                )
        
        # Set trainability
        self.embedding_layer.weight.requires_grad = trainable
        
        # Move to device if specified
        if device != "cpu":
            self.embedding_layer = self.embedding_layer.to(device)
        

    def _get_cache_path(self) -> Path:
        """Get cache directory path for current model."""
        # Convert model name to safe directory name  
        safe_name = self.pretrained_model_name.replace("/", "-")
        return Path("cache") / "embedders" / "text" / safe_name

    def _cache_exists(self) -> bool:
        """Check if cache exists for current model."""
        cache_dir = self._get_cache_path()
        required_files = [
            cache_dir / "tokenizer" / "tokenizer.json",
            cache_dir / "config.json", 
            cache_dir / "embedding_weights.pt"
        ]
        return all(f.exists() for f in required_files)

    def _load_from_cache(self) -> Tuple[object, object, torch.Tensor]:
        """Load tokenizer, config, and embedding weights from cache."""
        cache_dir = self._get_cache_path()
        
        # Load tokenizer from cache
        tokenizer = AutoTokenizer.from_pretrained(cache_dir / "tokenizer")
        
        # Load config from cache  
        config = AutoConfig.from_pretrained(cache_dir)
        
        # Load embedding weights from cache
        embedding_weights = torch.load(cache_dir / "embedding_weights.pt", map_location="cpu")
        
        return tokenizer, config, embedding_weights

    def _to_fullwidth(self, text: str) -> str:
        """
        Convert half-width characters to full-width for Japanese text.
        
        This is crucial for JMedRoBERTa which expects full-width (zenkaku) characters.
        
        Args:
            text: Input text possibly containing half-width characters
            
        Returns:
            Text with half-width characters converted to full-width
        """
        text = unicodedata.normalize('NFKC', text)
        
        out = []
        for c in text:
            code = ord(c)
            if c == ' ':
                out.append('\u3000')  # full-width space
            elif 33 <= code <= 126:
                out.append(chr(code + 0xFEE0))
            else:
                out.append(c)
        return ''.join(out)

    @classmethod
    def from_config(cls, config_path: str) -> "TextEmbedder":
        """
        Create an embedder from a YAML config file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Configured TextEmbedder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            pretrained_model_name=config.get('pretrained_model_name', "alabnii/jmedroberta-base-sentencepiece"),
            max_length=config.get('max_length', 512),
            padding=config.get('padding', "longest"),
            truncation=config.get('truncation', True),
            add_phi_tokens=config.get('add_phi_tokens', True),
            phi_patterns_path=config.get('phi_patterns_path', "config/cleaning/phi_patterns.yaml"),
            special_tokens=config.get('special_tokens', None),
            output_dir=config.get('output_dir', "outputs/embedders/text_embedder"),
            trainable=config.get('trainable', True),
            device=config.get('device', "cpu")
        )
            
    def _load_phi_tokens(self, phi_patterns_path: str) -> List[str]:
        """
        Extract PHI replacement tokens from a patterns YAML.
        
        Supports both legacy schema (categories at the YAML top-level) and
        v1 schema (categories nested under the top-level key "patterns").
        
        Args:
            phi_patterns_path: Path to PHI patterns YAML
            
        Returns:
            List of PHI replacement tokens such as ["<ADDRESS>", "<EMAIL>", ...]
        """
        if not os.path.exists(phi_patterns_path):
            logger.warning(f"PHI patterns file not found at {phi_patterns_path}")
            return []

        try:
            with open(phi_patterns_path, 'r') as f:
                phi_config = yaml.safe_load(f)

            categories_iterable = []
            detected_schema = "unknown"

            if isinstance(phi_config, dict) and 'patterns' in phi_config and isinstance(phi_config['patterns'], dict):
                # v1 schema with explicit meta + patterns sections
                detected_schema = "v1"
                src = phi_config['patterns']
                # Accept keys whose values look like a list of regex patterns
                categories_iterable = [
                    k for k, v in src.items()
                    if isinstance(k, str) and not k.startswith('#') and isinstance(v, (list, tuple)) and len(v) > 0
                ]
            elif isinstance(phi_config, dict):
                # Legacy schema: categories at top level; ignore pseudo-comment keys
                detected_schema = "legacy"
                categories_iterable = [
                    k for k, v in phi_config.items()
                    if isinstance(k, str) and not k.startswith('#') and isinstance(v, (list, tuple)) and len(v) > 0
                ]

            # Normalize and make ordering deterministic for reproducibility
            categories = sorted({str(k).upper() for k in categories_iterable})

            if detected_schema == "legacy":
                logger.info("Detected legacy PHI patterns schema; consider migrating to v1 'meta+patterns'.")
            elif detected_schema == "v1":
                logger.info("Detected v1 PHI patterns schema under 'patterns'.")
            else:
                logger.warning("Could not confidently detect PHI patterns schema; proceeding with empty token set.")

            logger.info(f"Extracted {len(categories)} PHI categories: {categories}")
            return [f"<{category}>" for category in categories]
        except Exception as e:
            logger.warning(f"Error loading PHI tokens: {e}")
            return []
    
    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text(s) to token IDs.
        
        Args:
            texts: Text string or list of text strings
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]

        # Convert to full-width before tokenization
        texts = [self._to_fullwidth(text) for text in texts]
            
        # Tokenize the input texts
        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return encoded
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to remove special tokens from the output
            
        Returns:
            Decoded text or list of texts
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Handle single sequence vs. batch
        is_single_sequence = not isinstance(token_ids[0], (list, torch.Tensor))
        if is_single_sequence:
            token_ids = [token_ids]
            
        # Decode each sequence
        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) 
                        for ids in token_ids]
        
        # Return single string if input was single sequence
        if is_single_sequence:
            return decoded_texts[0]
        return decoded_texts
    
    def embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, device: Optional[str] = None) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            device: Optional device override
            
        Returns:
            Embedded representation [batch_size, seq_len, embedding_dim]
        """
        device = device or self.device
        if device != input_ids.device:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        return self.embedding_layer(input_ids)
    
    def process(self, texts: Union[str, List[str]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Complete pipeline: tokenize and embed in one step.
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            (tokenized_outputs, embeddings)
        """
        tokenized = self.tokenize(texts)
        embeddings = self.embed(tokenized["input_ids"])
        return tokenized, embeddings
    
    def to(self, device: str) -> "TextEmbedder":
        """
        Move embedder to specified device.
        
        Args:
            device: Device to move to ("cpu" or "cuda:x")
            
        Returns:
            Self (for method chaining)
        """
        self.device = device
        self.embedding_layer = self.embedding_layer.to(device)
        return self
    
    def set_trainable(self, trainable: bool) -> "TextEmbedder":
        """
        Set whether embedding weights should be trainable.
        
        Args:
            trainable: Whether embedding weights should be trainable
            
        Returns:
            Self (for method chaining)
        """
        self.trainable = trainable
        self.embedding_layer.weight.requires_grad = trainable
        return self
    
    def is_trainable(self) -> bool:
        """
        Check if embedding weights are trainable.
        
        Returns:
            True if trainable, False if frozen
        """
        return self.trainable
    
    def get_vocab_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return len(self.tokenizer)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            int: Embedding dimension
        """
        return self.embedding_layer.embedding_dim
    
    def get_phi_token_ids(self) -> Dict[str, int]:
        """
        Get mapping from PHI token strings to token IDs.
        
        Returns:
            Dictionary mapping PHI token strings to their IDs
        """
        return {token: self.tokenizer.convert_tokens_to_ids(token) 
                for token in self.added_special_tokens 
                if token.startswith('<') and token.endswith('>')}
    
    def save_pretrained(self, directory: Optional[str] = None) -> None:
        """
        Save embedder state to disk.
        
        Args:
            directory: Directory to save embedder state, defaults to self.output_dir
        """
        directory = directory or self.output_dir
        os.makedirs(directory, exist_ok=True)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(directory)
        
        # Save just the embedding weights
        torch.save(self.embedding_layer.state_dict(), os.path.join(directory, "embedding_weights.pt"))
        
        # Save metadata
        metadata = {
            "pretrained_model_name": self.pretrained_model_name,
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "vocab_size": self.get_vocab_size(),
            "embedding_dim": self.get_embedding_dimension(),
            "special_tokens": self.added_special_tokens,
            "trainable": self.trainable,
            "device": self.device
        }
        
        with open(os.path.join(directory, "text_embedder_metadata.yaml"), "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
            
    @classmethod
    def from_pretrained(cls, directory: str) -> "TextEmbedder":
        """
        Load embedder from saved state.
        
        Args:
            directory: Directory with saved embedder state
            
        Returns:
            Loaded TextEmbedder instance
        """
        # Load metadata
        metadata_path = os.path.join(directory, "text_embedder_metadata.yaml")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            raise ValueError(f"No metadata file found at {metadata_path}")
        
        # Create instance with minimal settings (no token adding)
        instance = cls(
            pretrained_model_name=metadata.get('pretrained_model_name'),
            max_length=metadata.get('max_length', 512),
            padding=metadata.get('padding', "longest"),
            truncation=metadata.get('truncation', True),
            add_phi_tokens=False,  # Don't reload PHI tokens, they're already in the saved tokenizer
            output_dir=directory,
            trainable=metadata.get('trainable', True),
            device=metadata.get('device', "cpu")
        )
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(directory)
        
        # Create embedding layer with correct dimensions
        vocab_size = len(instance.tokenizer)
        embedding_dim = metadata.get('embedding_dim')
        instance.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        instance.embedding_layer.weight.requires_grad = instance.trainable
        
        # Load embedding weights
        weight_path = os.path.join(directory, "embedding_weights.pt")
        instance.embedding_layer.load_state_dict(torch.load(weight_path, map_location=instance.device))
        
        # Store special tokens
        instance.added_special_tokens = metadata.get('special_tokens', [])
        
        return instance

    def parameters(self):
        """Return parameters from the underlying embedding layer."""
        return self.embedding_layer.parameters()

    def state_dict(self):
        """Return state dict from the underlying embedding layer."""
        return self.embedding_layer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into the underlying embedding layer."""
        return self.embedding_layer.load_state_dict(state_dict)
    
    def __repr__(self) -> str:
        """String representation with key information."""
        return (f"TextEmbedder(model='{self.pretrained_model_name}', "
                f"vocab_size={self.get_vocab_size()}, "
                f"embedding_dim={self.get_embedding_dimension()}, "
                f"special_tokens={len(self.added_special_tokens)}, "
                f"trainable={self.trainable})")

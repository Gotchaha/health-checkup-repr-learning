# tests/models/embedders/test_TextEmbedder.py
import os
import torch
import pytest
from src.models.embedders.TextEmbedder import TextEmbedder

class TestTextEmbedder:
    """Test suite for TextEmbedder class."""
    
    @pytest.fixture
    def embedder(self):
        """Create a basic embedder instance for testing."""
        return TextEmbedder(
            pretrained_model_name="alabnii/jmedroberta-base-sentencepiece",
            max_length=128,  # Smaller for testing
            add_phi_tokens=True,
            phi_patterns_path="config/cleaning/phi_patterns.yaml"
        )
    
    @pytest.fixture
    def test_texts(self):
        """Sample Japanese medical texts for testing."""
        return [
            "患者は頭痛を訴えています。",  # Patient complains of headache
            "血圧は正常範囲内です。",      # Blood pressure is within normal range
        ]
    
    def test_initialization(self, embedder):
        """Test that embedder initializes correctly."""
        assert embedder.pretrained_model_name == "alabnii/jmedroberta-base-sentencepiece"
        assert embedder.max_length == 128
        assert embedder.device == "cpu"
        assert embedder.trainable == True
        assert hasattr(embedder, "tokenizer")
        assert hasattr(embedder, "embedding_layer")
    
    def test_tokenize(self, embedder, test_texts):
        """Test tokenization functionality."""
        tokens = embedder.tokenize(test_texts)
        
        # Check output format
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        
        # Check dimensions
        assert tokens["input_ids"].dim() == 2
        assert tokens["attention_mask"].dim() == 2
        
        # Check batch size
        assert tokens["input_ids"].size(0) == 2
        
        # Sequence should be shorter than max_length
        assert tokens["input_ids"].size(1) <= embedder.max_length
    
    def test_embed(self, embedder, test_texts):
        """Test embedding functionality."""
        tokens = embedder.tokenize(test_texts)
        embeddings = embedder.embed(tokens["input_ids"])
        
        # Check output dimensions
        assert embeddings.dim() == 3
        assert embeddings.size(0) == 2  # Batch size
        assert embeddings.size(1) <= embedder.max_length  # Sequence length
        assert embeddings.size(2) == embedder.get_embedding_dimension()  # Embedding dimension
    
    def test_process(self, embedder, test_texts):
        """Test complete process pipeline."""
        tokens, embeddings = embedder.process(test_texts)
        
        # Check tokens
        assert "input_ids" in tokens
        assert tokens["input_ids"].size(0) == 2
        
        # Check embeddings
        assert embeddings.dim() == 3
        assert embeddings.size(0) == 2
        assert embeddings.size(2) == embedder.get_embedding_dimension()
    
    def test_fullwidth_conversion(self, embedder):
        """Test half-width to full-width conversion."""
        half_width = "ﾃｽﾄ 123 ABC"
        full_width = embedder._to_fullwidth(half_width)
        
        # Check that characters are converted to full-width
        assert '1' not in full_width
        assert 'A' not in full_width
        assert ' ' not in full_width  # Space should be converted to ideographic space
        assert '１' in full_width
        assert 'Ａ' in full_width
        assert '　' in full_width  # Ideographic space
    
    def test_device_handling(self, embedder):
        """Test device handling functionality."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")
        
        # Move to CUDA
        embedder.to("cuda")
        assert embedder.device == "cuda"
        
        # Check that embedding layer is on CUDA
        assert embedder.embedding_layer.weight.device.type == "cuda"
        
        # Process text
        tokens, embeddings = embedder.process("テスト")
        assert embeddings.device.type == "cuda"
        
        # Move back to CPU for cleanup
        embedder.to("cpu")
    
    def test_trainability(self, embedder):
        """Test that trainability can be controlled."""
        # Default should be trainable
        assert embedder.is_trainable() == True
        assert embedder.embedding_layer.weight.requires_grad == True
        
        # Test setting to not trainable
        embedder.set_trainable(False)
        assert embedder.is_trainable() == False
        assert embedder.embedding_layer.weight.requires_grad == False
        
        # Test setting back to trainable
        embedder.set_trainable(True)
        assert embedder.is_trainable() == True
        assert embedder.embedding_layer.weight.requires_grad == True

    def test_trainability_from_init(self):
        """Test that trainability can be set from initialization."""
        frozen_embedder = TextEmbedder(
            pretrained_model_name="alabnii/jmedroberta-base-sentencepiece",
            max_length=128,
            trainable=False
        )
        assert frozen_embedder.is_trainable() == False
        assert frozen_embedder.embedding_layer.weight.requires_grad == False
    
    def test_save_load(self, embedder, tmp_path, test_texts):
        """Test saving and loading functionality."""
        save_dir = tmp_path / "test_embedder"
        
        # Change trainability to test preservation
        embedder.set_trainable(False)
        
        # Process text with original embedder
        orig_tokens, orig_embeddings = embedder.process(test_texts[0])
        
        # Save embedder
        embedder.save_pretrained(str(save_dir))
        
        # Check that files were created
        assert os.path.exists(save_dir / "text_embedder_metadata.yaml")
        assert os.path.exists(save_dir / "embedding_weights.pt")
        
        # Load embedder
        loaded = TextEmbedder.from_pretrained(str(save_dir))
        
        # Check that trainability was preserved
        assert loaded.is_trainable() == False
        assert loaded.embedding_layer.weight.requires_grad == False
        
        # Process same text with loaded embedder
        loaded_tokens, loaded_embeddings = loaded.process(test_texts[0])
        
        # Check that outputs match
        assert torch.allclose(orig_embeddings, loaded_embeddings)
        assert torch.all(orig_tokens["input_ids"] == loaded_tokens["input_ids"])
    
    def test_get_phi_token_ids(self, embedder):
        """Test that PHI token IDs can be retrieved."""
        phi_token_ids = embedder.get_phi_token_ids()
        
        # Check that we have PHI tokens
        assert len(phi_token_ids) > 0
        
        # Check format of tokens
        for token, token_id in phi_token_ids.items():
            assert token.startswith('<')
            assert token.endswith('>')
            assert isinstance(token_id, int)
            
        # Verify specific expected tokens
        expected_categories = {"ADDRESS", "EMAIL", "FACILITY", "ID", "NAME", "PHONE", "POSTAL"}
        found_categories = {token.strip('<>') for token in phi_token_ids.keys()}
        assert expected_categories.issubset(found_categories), \
            f"Expected to find at least these categories: {expected_categories}, but found: {found_categories}"
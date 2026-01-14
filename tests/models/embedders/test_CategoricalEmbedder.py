# tests/models/embedders/test_CategoricalEmbedder.py

import os
import yaml
import torch
import pytest
import tempfile
import shutil
from pathlib import Path

from src.models.embedders.CategoricalEmbedder import CategoricalEmbedder


class TestCategoricalEmbedder:
    """Test suite for CategoricalEmbedder class."""
    
    @pytest.fixture
    def test_vocab(self):
        """Create a test vocabulary for embedder tests."""
        vocab = {
            "token_to_idx": {
                "<PAD>": 0,
                "<UNK>": 1,
                "000000300004": 2,
                "000000300005": 3,
                "000000400005": 4,
                "000000400005=UNK": 5,
                "000000400005=1": 6,
                "000000400005=2": 7,
                "000000400005=3": 8,
                "000000400005=4": 9
            },
            "idx_to_token": {
                "0": "<PAD>",
                "1": "<UNK>",
                "2": "000000300004",
                "3": "000000300005",
                "4": "000000400005",
                "5": "000000400005=UNK",
                "6": "000000400005=1",
                "7": "000000400005=2",
                "8": "000000400005=3",
                "9": "000000400005=4"
            },
            "vocab_size": 10,
            "metadata": {
                "n_tests": 3,
                "n_categorical_tests": 1,
                "n_categorical_values": 4
            }
        }
        
        # Create a temporary directory and save the vocabulary
        temp_dir = tempfile.mkdtemp()
        vocab_path = os.path.join(temp_dir, "test_vocab.yaml")
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            yaml.dump(vocab, f, default_flow_style=False, allow_unicode=True)
            
        yield vocab_path
        
        # Clean up
        shutil.rmtree(temp_dir)
        

    @pytest.fixture
    def test_config(self, test_vocab):
        """Create a test configuration for embedder tests."""
        config = {
            "vocab_path": test_vocab,
            "embedding_dim": 16,  # Small dim for testing
            "trainable": True,
            "padding_idx": 0,
            "use_xavier_init": True,
            "output_dir": None,
            "device": "cpu"
        }
        
        # Create a temporary directory and save the config
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config["output_dir"] = os.path.join(temp_dir, "embedder_output")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        yield config_path
        
        # Clean up
        shutil.rmtree(temp_dir)


    def test_initialization(self, test_vocab):
        """Test basic initialization of the embedder."""
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        assert embedder.vocab_size == 10
        assert embedder.embedding_dim == 16
        assert embedder.embeddings.weight.size() == (10, 16)
        assert embedder.embeddings.weight.requires_grad == True
        
        # Test non-trainable
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16,
            trainable=False
        )
        assert embedder.embeddings.weight.requires_grad == False


    def test_from_config(self, test_config):
        """Test initialization from config file."""
        embedder = CategoricalEmbedder.from_config(test_config)
        
        assert embedder.vocab_size == 10
        assert embedder.embedding_dim == 16
        assert embedder.trainable == True


    def test_map_single_sequence(self, test_vocab):
        """Test mapping a single sequence of tokens."""
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        # Test known tokens
        tokens = ["000000300004", "000000400005=1", "000000300005"]
        token_ids = embedder.map(tokens)
        
        assert token_ids.size() == (3,)
        assert token_ids.tolist() == [2, 6, 3]
        
        # Test unknown tokens
        tokens = ["000000300004", "unknown_token", "000000300005"]
        token_ids = embedder.map(tokens)
        
        assert token_ids.tolist() == [2, 1, 3]  # 1 is <UNK>


    def test_map_batch_sequences(self, test_vocab):
        """Test mapping a batch of token sequences."""
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        # Test batch with different lengths
        batch_tokens = [
            ["000000300004", "000000400005=1"],
            ["000000300004", "000000400005=2", "000000300005"]
        ]
        
        token_ids = embedder.map(batch_tokens)
        
        assert token_ids.size() == (2, 3)  # Padded to longest sequence
        assert token_ids[0].tolist() == [2, 6, 0]  # 0 is padding
        assert token_ids[1].tolist() == [2, 7, 3]


    def test_embedding(self, test_vocab):
        """Test embedding token IDs."""
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        # Embed a single token ID
        token_ids = torch.tensor([2], dtype=torch.long)
        embeddings = embedder.embed(token_ids)
        
        assert embeddings.size() == (1, 16)
        
        # Embed a sequence
        token_ids = torch.tensor([2, 6, 3], dtype=torch.long)
        embeddings = embedder.embed(token_ids)
        
        assert embeddings.size() == (3, 16)
        
        # Embed a batch
        token_ids = torch.tensor([[2, 6, 0], [2, 7, 3]], dtype=torch.long)
        embeddings = embedder.embed(token_ids)
        
        assert embeddings.size() == (2, 3, 16)


    def test_forward_pass(self, test_vocab):
        """Test the forward pass (map + embed)."""
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        # Single sequence
        tokens = ["000000300004", "000000400005=1", "000000300005"]
        embeddings = embedder(tokens)
        
        assert embeddings.size() == (3, 16)
        
        # Batch of sequences
        batch_tokens = [
            ["000000300004", "000000400005=1"],
            ["000000300004", "000000400005=2", "000000300005"]
        ]
        
        embeddings = embedder(batch_tokens)
        
        assert embeddings.size() == (2, 3, 16)


    def test_decode(self, test_vocab):
        """Test decoding token IDs back to tokens."""
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        # Decode a single sequence
        token_ids = torch.tensor([2, 6, 3], dtype=torch.long)
        tokens = embedder.decode(token_ids)
        
        assert tokens == ["000000300004", "000000400005=1", "000000300005"]
        
        # Decode a batch with padding
        token_ids = torch.tensor([[2, 6, 0], [2, 7, 3]], dtype=torch.long)
        tokens = embedder.decode(token_ids)
        
        assert tokens == [
            ["000000300004", "000000400005=1"],
            ["000000300004", "000000400005=2", "000000300005"]
        ]


    def test_save_load(self, test_vocab):
        """Test saving and loading the embedder."""
        # Create an embedder and modify its weights to check persistence
        embedder = CategoricalEmbedder(
            vocab_path=test_vocab,
            embedding_dim=16
        )
        
        # Set a specific weight pattern to test persistence
        with torch.no_grad():
            embedder.embeddings.weight[1, 0] = 0.42
        
        # Save to temporary directory
        temp_dir = tempfile.mkdtemp()
        saved_dir = embedder.save_pretrained(temp_dir)
        
        # Check saved files
        assert os.path.exists(os.path.join(saved_dir, "embedding_weights.pt"))
        assert os.path.exists(os.path.join(saved_dir, "config.yaml"))
        
        # Load back
        loaded_embedder = CategoricalEmbedder.from_pretrained(saved_dir)
        
        # Check if weights were preserved
        assert loaded_embedder.embeddings.weight[1, 0].item() == pytest.approx(0.42)
        assert loaded_embedder.embedding_dim == 16
        assert loaded_embedder.vocab_size == 10
        
        # Clean up
        shutil.rmtree(temp_dir)
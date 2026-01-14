# tests/models/embedders/v1/test_TextEmbedder_phi_loader.py
"""
Focused tests for the PHI token loader with v1 YAML schema.

Design goals:
- Avoid heavy model/tokenizer loading by bypassing __init__.
- Validate schema detection and extracted categories for v1 file.
- Sanity-check legacy file support using the same private loader.
"""

import os
import pytest

from src.models.embedders.TextEmbedder import TextEmbedder


@pytest.fixture(scope="module")
def text_embedder_shim() -> TextEmbedder:
    """Create a TextEmbedder instance without running __init__.

    This allows calling the private loader directly without downloading
    any models or tokenizers. This is appropriate for unit-testing the
    YAML parsing logic in isolation.
    """
    return object.__new__(TextEmbedder)


def _categories_from_tokens(tokens):
    return {t.strip('<>') for t in tokens}


def test_load_phi_tokens_from_v1_yaml(text_embedder_shim: TextEmbedder):
    """Loader should detect v1 schema and extract expected categories."""
    v1_path = "config/cleaning/v1/deidentification/phi_patterns.yaml"
    assert os.path.exists(v1_path), f"Missing v1 patterns file: {v1_path}"

    tokens = text_embedder_shim._load_phi_tokens(v1_path)
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)

    found = _categories_from_tokens(tokens)
    expected = {"ADDRESS", "EMAIL", "FACILITY", "ID", "NAME", "PHONE", "POSTAL"}

    # Validate that all expected categories are present
    assert expected.issubset(found), f"Expected at least {expected}, found {found}"

    # Ensure meta keys are not treated as categories
    assert "META" not in found
    assert "PATTERNS" not in found


def test_load_phi_tokens_from_legacy_yaml(text_embedder_shim: TextEmbedder):
    """Loader should still support the legacy top-level categories schema."""
    legacy_path = "config/cleaning/phi_patterns.yaml"
    assert os.path.exists(legacy_path), f"Missing legacy patterns file: {legacy_path}"

    tokens = text_embedder_shim._load_phi_tokens(legacy_path)
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)

    found = _categories_from_tokens(tokens)
    expected = {"ADDRESS", "EMAIL", "FACILITY", "ID", "NAME", "PHONE", "POSTAL"}
    assert expected.issubset(found)


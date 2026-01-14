# scripts/embedders/cache_default_models.py
"""
Cache default pretrained models for faster TextEmbedder initialization.
Downloads and caches the default model to avoid repeated HuggingFace downloads.

Usage:
    python scripts/embedders/cache_default_models.py
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Calculate project root (script is in scripts/embedders/, so go up two levels)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Change working directory to project root to ensure relative paths work
os.chdir(project_root)

# Verify we're in the correct directory by checking for expected files
if not (project_root / "src").exists() or not (project_root / "config").exists():
    raise RuntimeError(f"Not in project root directory. Current: {project_root}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model to cache
DEFAULT_MODEL = "alabnii/jmedroberta-base-sentencepiece"


def get_cache_path(model_name: str) -> Path:
    """
    Get cache directory path for a given model.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        Path to cache directory
    """
    # Convert model name to safe directory name
    safe_name = model_name.replace("/", "-")
    return project_root / "cache" / "embedders" / "text" / safe_name


def cache_exists(model_name: str) -> bool:
    """
    Check if cache exists for a given model.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        True if all required cache files exist
    """
    cache_dir = get_cache_path(model_name)
    
    required_files = [
        cache_dir / "tokenizer" / "tokenizer.json",
        cache_dir / "config.json", 
        cache_dir / "embedding_weights.pt"
    ]
    
    return all(f.exists() for f in required_files)


def cache_model(model_name: str) -> None:
    """
    Download and cache a pretrained model.
    
    Args:
        model_name: Name of the pretrained model to cache
    """
    logger.info(f"Caching model: {model_name}")
    cache_dir = get_cache_path(model_name)
    
    # Create cache directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = cache_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # 1. Download and save tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Tokenizer saved to: {tokenizer_dir}")
        
        # 2. Download and save config
        logger.info("Downloading model config...")
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(cache_dir)
        logger.info(f"Config saved to: {cache_dir / 'config.json'}")
        
        # 3. Download model and extract embedding weights
        logger.info("Downloading full model (this may take a while)...")
        temp_model = AutoModel.from_pretrained(model_name)
        embedding_weights = temp_model.get_input_embeddings().weight.data.clone()
        
        # Save embedding weights
        embedding_path = cache_dir / "embedding_weights.pt"
        torch.save(embedding_weights, embedding_path)
        logger.info(f"Embedding weights saved to: {embedding_path}")
        logger.info(f"Embedding shape: {embedding_weights.shape}")
        
        # Clean up temporary model
        del temp_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log success
        total_time = time.time() - start_time
        logger.info(f"✓ Model {model_name} cached successfully in {total_time:.2f}s")
        logger.info(f"Cache location: {cache_dir}")
        
    except Exception as e:
        logger.error(f"Failed to cache model {model_name}: {e}")
        # Clean up partial cache on failure
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        raise


def main():
    """Main caching function."""
    logger.info("="*60)
    logger.info("DEFAULT MODEL CACHING SCRIPT")
    logger.info("="*60)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Model to cache: {DEFAULT_MODEL}")
    
    # Check if already cached
    if cache_exists(DEFAULT_MODEL):
        logger.info(f"✓ Model {DEFAULT_MODEL} is already cached")
        cache_dir = get_cache_path(DEFAULT_MODEL)
        logger.info(f"Cache location: {cache_dir}")
        
        # Show cache contents
        logger.info("\nCache contents:")
        for item in sorted(cache_dir.rglob("*")):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info(f"  {item.relative_to(cache_dir)}: {size_mb:.1f}MB")
        
        return
    
    # Cache the model
    logger.info(f"Model {DEFAULT_MODEL} not cached. Starting download...")
    
    try:
        cache_model(DEFAULT_MODEL)
        
        logger.info("\n" + "="*60)
        logger.info("CACHING COMPLETE")
        logger.info("="*60)
        logger.info("✓ Default model cached successfully")
        logger.info("✓ Future TextEmbedder initializations will be much faster")
        
    except KeyboardInterrupt:
        logger.info("\nCaching interrupted by user")
    except Exception as e:
        logger.error(f"Caching failed: {e}")
        raise


if __name__ == "__main__":
    main()
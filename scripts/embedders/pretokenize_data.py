# scripts/embedders/pretokenize_data.py
"""
Pre-tokenize result text data for accelerated training.

This script tokenizes result text data offline and caches the token IDs 
and attention masks to eliminate tokenization overhead during training.
"""

import os
import argparse
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import yaml
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to Python path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import TextEmbedder from project
from src.models.embedders.TextEmbedder import TextEmbedder


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"pretokenize_data_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set up new handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Set format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file for verification."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_output_filename(
    pretrained_model_name: str,
    max_length: int,
    truncation: bool,
    add_phi_tokens: bool
) -> str:
    """Generate descriptive filename for cached tokenized data."""
    # Simplify model name for filename
    model_short = pretrained_model_name.split('/')[-1].replace('-', '')
    trunc_suffix = "_trunc" if truncation else "_notrunc"
    phi_suffix = "_phi" if add_phi_tokens else "_nophi"
    return f"result_{model_short}_tok{max_length}{trunc_suffix}{phi_suffix}.parquet"


def create_metadata(
    pretrained_model_name: str,
    max_length: int,
    truncation: bool,
    add_phi_tokens: bool,
    phi_patterns_path: str,
    device: str,
    source_path: Path,
    output_path: Path,
    text_embedder: TextEmbedder
) -> Dict[str, Any]:
    """Create metadata dictionary for caching record."""
    return {
        'creation_timestamp': datetime.now().isoformat(),
        'tokenizer': {
            'pretrained_model_name': pretrained_model_name,
            'max_length': max_length,
            'padding': False,  # Individual processing, no padding
            'truncation': truncation,
            'add_phi_tokens': add_phi_tokens,
            'phi_patterns_path': phi_patterns_path,
            'device': device,
            'vocab_size': text_embedder.get_vocab_size(),
            'special_tokens': text_embedder.added_special_tokens
        },
        'source_data': {
            'file_path': str(source_path),
            'file_hash': compute_file_hash(source_path),
            'num_samples': None  # Will be filled after reading
        },
        'cached_data': {
            'file_path': str(output_path),
            'file_hash': None,  # Will be filled after writing
            'num_samples': None  # Will be filled after processing
        }
    }


def tokenize_result_data(
    source_path: Path,
    text_embedder: TextEmbedder
) -> pa.Table:
    """
    Tokenize result text data individually for each sample.
    
    Returns PyArrow table with preserved key columns and tokenized data.
    """
    logging.info(f"Loading source data from {source_path}")
    # Read only necessary columns to reduce memory usage and I/O
    source_table = pq.read_table(
        source_path,
        columns=["AnonymousID", "McExamDt", "ResultText"]
    )
    num_samples = len(source_table)
    logging.info(f"Processing {num_samples} samples")
    
    # Extract data for processing
    anonymous_ids = source_table['AnonymousID'].to_pylist()
    exam_dates = source_table['McExamDt'].to_pylist()
    result_texts = source_table['ResultText'].to_pylist()
    
    # Process each sample individually
    tokenized_data = {
        'AnonymousID': [],
        'McExamDt': [],
        'input_ids': [],
        'attention_mask': []
    }
    
    for i, (anon_id, exam_date, result_text) in enumerate(
        zip(anonymous_ids, exam_dates, result_texts)
    ):
        if i % 10000 == 0:
            logging.info(f"Processed {i}/{num_samples} samples ({i/num_samples*100:.1f}%)")
        
        # Handle potential null/empty text
        if not result_text or result_text is None:
            result_text = ""  # Use empty string for null values
        
        # Tokenize individual sample
        tokens = text_embedder.tokenize(result_text)
        
        # Move tensors to CPU before converting to lists (handles GPU tensors)
        input_ids = tokens['input_ids'][0].cpu().tolist()
        attention_mask = tokens['attention_mask'][0].cpu().tolist()
        
        # Ensure native Python int types for PyArrow compatibility
        input_ids = [int(x) for x in input_ids]
        attention_mask = [int(x) for x in attention_mask]
        
        # Store results preserving original key column values
        tokenized_data['AnonymousID'].append(anon_id)
        tokenized_data['McExamDt'].append(exam_date)
        tokenized_data['input_ids'].append(input_ids)
        tokenized_data['attention_mask'].append(attention_mask)
    
    logging.info(f"Tokenization complete for {num_samples} samples (100.0%)")
    
    # Create PyArrow table with preserved column types
    schema = pa.schema([
        ('AnonymousID', source_table.schema.field('AnonymousID').type),
        ('McExamDt', source_table.schema.field('McExamDt').type),
        ('input_ids', pa.list_(pa.int32())),
        ('attention_mask', pa.list_(pa.int8()))
    ])
    
    return pa.table(tokenized_data, schema=schema)


def main():
    """Main tokenization pipeline."""
    parser = argparse.ArgumentParser(description="Pre-tokenize result text data")
    
    # Core tokenizer parameters
    parser.add_argument(
        '--pretrained_model_name',
        default='alabnii/jmedroberta-base-sentencepiece',
        help='Pretrained model name for tokenizer'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length for tokenization'
    )
    parser.add_argument(
        '--truncation',
        action='store_true',
        default=True,
        help='Enable truncation for long sequences (default: True)'
    )
    parser.add_argument(
        '--no_truncation', 
        dest='truncation',
        action='store_false',
        help='Disable truncation for long sequences'
    )
    parser.add_argument(
        '--add_phi_tokens',
        action='store_true',
        default=True,
        help='Add PHI replacement tokens to vocabulary (default: True)'
    )
    parser.add_argument(
        '--no_add_phi_tokens',
        dest='add_phi_tokens',
        action='store_false', 
        help='Do not add PHI replacement tokens to vocabulary'
    )
    parser.add_argument(
        '--phi_patterns_path',
        default='config/cleaning/phi_patterns.yaml',
        help='Path to PHI patterns configuration'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device for tokenizer (cpu or cuda:x)'
    )
    
    # Data processing parameters
    parser.add_argument(
        '--source_path',
        default='data/processed/result.parquet',
        help='Path to source result data'
    )
    parser.add_argument(
        '--output_dir',
        default='cache/pretokenized/',
        help='Output directory for cached data'
    )
    parser.add_argument(
        '--force_regenerate',
        action='store_true',
        help='Force regeneration even if cache exists'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    source_path = Path(args.source_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_output_dir = Path('outputs/cache')
    setup_logging(log_output_dir)
    
    # Generate output filename
    output_filename = generate_output_filename(
        args.pretrained_model_name,
        args.max_length,
        args.truncation,
        args.add_phi_tokens
    )
    output_path = output_dir / output_filename
    metadata_path = output_dir / f"{Path(output_filename).stem}_metadata.yaml"
    
    # Check if cache exists
    if output_path.exists() and not args.force_regenerate:
        logging.info(f"Cache already exists at {output_path}")
        logging.info("Use --force_regenerate to recreate")
        return
    
    logging.info("Starting tokenization pipeline")
    logging.info(f"Model: {args.pretrained_model_name}")
    logging.info(f"Max length: {args.max_length}")
    logging.info(f"Truncation: {args.truncation}")
    logging.info(f"Add PHI tokens: {args.add_phi_tokens}")
    logging.info(f"Device: {args.device}")
    
    # Initialize TextEmbedder
    logging.info("Initializing TextEmbedder")
    text_embedder = TextEmbedder(
        pretrained_model_name=args.pretrained_model_name,
        max_length=args.max_length,
        padding=False,  # No padding for individual sample processing
        truncation=args.truncation,
        add_phi_tokens=args.add_phi_tokens,
        phi_patterns_path=args.phi_patterns_path,
        device=args.device
    )
    
    # Create metadata
    metadata = create_metadata(
        args.pretrained_model_name,
        args.max_length,
        args.truncation,
        args.add_phi_tokens,
        args.phi_patterns_path,
        args.device,
        source_path,
        output_path,
        text_embedder
    )
    
    # Tokenize data
    tokenized_table = tokenize_result_data(source_path, text_embedder)
    
    # Update metadata with processing results
    metadata['source_data']['num_samples'] = len(tokenized_table)
    metadata['cached_data']['num_samples'] = len(tokenized_table)
    
    # Save tokenized data
    logging.info(f"Saving tokenized data to {output_path}")
    pq.write_table(tokenized_table, output_path)
    
    # Compute output file hash and finalize metadata
    metadata['cached_data']['file_hash'] = compute_file_hash(output_path)
    
    # Save metadata
    logging.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logging.info("Tokenization pipeline completed successfully")
    logging.info(f"Cached data: {output_path}")
    logging.info(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
# scripts/data_preparation/deidentify_data.py
"""
De-identify healthcare data files according to privacy requirements.
Applies ID rehashing, date jittering, and PHI scrubbing as appropriate for each data type.
"""
import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the phi_scrubber module functions
from src.data.cleaning.phi_scrubber import (
    PHIScrubber,
    process_mcinfo_data,
    process_interview_data,
    process_result_data
)

def setup_logging(log_dir="outputs/audit/phi_removal_logs"):
    """
    Configure logging to file and console.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"deidentification_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("deidentify")

def process_mcinfo_file(input_path, output_path, mapping, chunk_size=None):
    """
    Process the mcinfo file (test-per-row) with ID mapping and date shifting only.
    
    Args:
        input_path: Path to input mcinfo file
        output_path: Path to output deidentified file
        mapping: DataFrame with ID mapping
        chunk_size: Size of chunks for processing (None to process all at once)
        
    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger("deidentify")
    logger.info(f"Processing mcinfo data: {input_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process in full or in chunks depending on size
    if chunk_size is None:
        try:
            # Try to load the entire file
            logger.info("Attempting to load entire mcinfo file")
            df = pd.read_parquet(input_path)
            logger.info(f"Loaded mcinfo data: {len(df):,} rows")
            
            # Process data (ID mapping and date shifting only)
            result, stats = process_mcinfo_data(df, mapping)
            logger.info(f"Processed mcinfo data: {len(result):,} rows kept, {stats['removed_rows']:,} rows removed")
            
            # Save result
            result.to_parquet(output_path, index=False)
            logger.info(f"Saved deidentified mcinfo data to {output_path}")
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to process entire file at once: {str(e)}")
            logger.info("Falling back to chunk processing")
            chunk_size = 1000000  # Default chunk size
    
    # Process in chunks
    logger.info(f"Processing mcinfo data in chunks of {chunk_size:,} rows")
    
    # Try to get total number of rows for progress bar
    try:
        total_rows = pd.read_parquet(input_path, columns=[]).shape[0]
        logger.info(f"Total rows in mcinfo file: {total_rows:,}")
    except Exception as e:
        total_rows = None
        logger.warning(f"Could not determine total row count: {str(e)}")
    
    # Initialize statistics
    stats = {
        "original_rows": 0,
        "kept_rows": 0, 
        "removed_rows": 0,
        "chunks_processed": 0
    }
    
    # Process in chunks
    temp_dir = f"{os.path.dirname(output_path)}/temp_mcinfo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create progress bar
    pbar = tqdm(total=total_rows, desc="Processing mcinfo data")
    
    # Read and process in chunks
    for chunk_idx, chunk in enumerate(pd.read_parquet(input_path, chunksize=chunk_size)):
        # Process chunk
        chunk_result, chunk_stats = process_mcinfo_data(chunk, mapping)
        
        # Save to temp file
        temp_file = f"{temp_dir}/chunk_{chunk_idx:04d}.parquet"
        chunk_result.to_parquet(temp_file, index=False)
        
        # Update statistics
        stats["original_rows"] += chunk_stats["original_rows"]
        stats["kept_rows"] += chunk_stats["kept_rows"]
        stats["removed_rows"] += chunk_stats["removed_rows"]
        stats["chunks_processed"] += 1
        
        # Update progress bar
        pbar.update(len(chunk))
        
        # Log progress
        if chunk_idx % 10 == 0 or chunk_idx < 5:
            logger.info(f"Processed chunk {chunk_idx + 1}, "
                        f"kept {chunk_stats['kept_rows']:,}/{chunk_stats['original_rows']:,} rows")
    
    pbar.close()
    
    # Combine chunks
    logger.info("Combining processed chunks")
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # List all chunk files
    chunk_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                          if f.startswith("chunk_") and f.endswith(".parquet")])
    
    if not chunk_files:
        logger.error("No processed chunks found!")
        raise ValueError("No processed chunks found")
    
    # Combine chunks
    try:
        # Use PyArrow for efficient concatenation
        tables = [pq.read_table(f) for f in chunk_files]
        combined = pa.concat_tables(tables)
        pq.write_table(combined, output_path)
        logger.info(f"Successfully combined {len(chunk_files)} chunks into {output_path}")
    except Exception as e:
        logger.error(f"Failed to combine chunks: {str(e)}")
        raise
    
    # Clean up temp directory
    import shutil
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to remove temp directory: {str(e)}")
    
    logger.info(f"Mcinfo data processing complete: "
               f"{stats['kept_rows']:,}/{stats['original_rows']:,} rows kept")
    
    return stats

def process_interview_file(input_path, output_path, mapping, phi_scrubber):
    """
    Process the interview file with ID mapping, date shifting, and text PHI scrubbing.
    
    Args:
        input_path: Path to input interview file
        output_path: Path to output deidentified file
        mapping: DataFrame with ID mapping
        phi_scrubber: PHIScrubber instance
        
    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger("deidentify")
    logger.info(f"Processing interview data: {input_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load interview data
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded interview data: {len(df):,} rows")
        
        # Log data type information to help with debugging
        if 'Interview' in df.columns:
            # Sample the first non-null interview
            for idx in range(min(100, len(df))):
                try:
                    interview_data = df.iloc[idx]['Interview']
                    if interview_data is not None:
                        logger.info(f"Interview column data type: {type(interview_data).__name__}")
                        
                        # Additional info for NumPy arrays
                        if isinstance(interview_data, np.ndarray):
                            if hasattr(interview_data, 'dtype') and interview_data.dtype.names is not None:
                                logger.info(f"NumPy structured array with fields: {interview_data.dtype.names}")
                            else:
                                logger.info(f"NumPy array shape: {interview_data.shape}")
                                
                            # Check first element if available
                            if len(interview_data) > 0:
                                first_item = interview_data[0]
                                logger.info(f"First interview item type: {type(first_item).__name__}")
                        break
                except:
                    continue
                    
        # Process data (ID mapping, date shifting, and PHI scrubbing)
        result, stats = process_interview_data(df, mapping, phi_scrubber)
        logger.info(f"Processed interview data: {len(result):,} rows kept, {stats['removed_rows']:,} rows removed")
        
        # Verify structure is preserved
        if set(df.columns) != set(result.columns):
            logger.warning("Column structure changed during processing!")
            logger.warning(f"Original columns: {set(df.columns)}")
            logger.warning(f"Result columns: {set(result.columns)}")
        
        # Save result
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved deidentified interview data to {output_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to process interview data: {str(e)}")
        raise

def process_result_file(input_path, output_path, mapping, phi_scrubber):
    """
    Process the result file with ID mapping, date shifting, and text PHI scrubbing.
    
    Args:
        input_path: Path to input result file
        output_path: Path to output deidentified file
        mapping: DataFrame with ID mapping
        phi_scrubber: PHIScrubber instance
        
    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger("deidentify")
    logger.info(f"Processing result data: {input_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load result data
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded result data: {len(df):,} rows")
        
        # Process data (ID mapping, date shifting, and PHI scrubbing)
        result, stats = process_result_data(df, mapping, phi_scrubber)
        logger.info(f"Processed result data: {len(result):,} rows kept, {stats['removed_rows']:,} rows removed")
        
        # Verify structure is preserved
        if set(df.columns) != set(result.columns):
            logger.warning("Column structure changed during processing!")
            logger.warning(f"Original columns: {set(df.columns)}")
            logger.warning(f"Result columns: {set(result.columns)}")
        
        # Save result
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved deidentified result data to {output_path}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to process result data: {str(e)}")
        raise

def save_audit_data(stats, config, log_dir="outputs/audit/phi_removal_logs"):
    """
    Save detailed audit data to JSON file.
    
    Args:
        stats: Dictionary with processing statistics
        config: Dictionary with configuration settings
        log_dir: Directory for audit files
    """
    logger = logging.getLogger("deidentify")
    
    # Helper function to make objects JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    # Create comprehensive audit data
    interview_phi = stats.get("interview", {}).get("phi_scrubbing", {}).get("total_replacements", 0)
    result_phi = stats.get("result", {}).get("phi_scrubbing", {}).get("total_replacements", 0)
    
    audit_data = {
        "timestamp": datetime.now().isoformat(),
        "config": make_serializable(config),
        "stats": make_serializable(stats),
        "summary": {
            "total_original_rows": int(sum(s.get("original_rows", 0) for s in stats.values())),
            "total_kept_rows": int(sum(s.get("kept_rows", 0) for s in stats.values())),
            "total_removed_rows": int(sum(s.get("removed_rows", 0) for s in stats.values())),
            "phi_replacements": int(interview_phi + result_phi)
        }
    }
    
    # Save to JSON file with error handling
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_file = os.path.join(log_dir, f"deidentification_audit_{timestamp}.json")
    
    try:
        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, default=str)
        logger.info(f"Saved audit data to {audit_file}")
    except Exception as e:
        logger.error(f"Failed to save audit data: {str(e)}")
        # Fallback method - try saving with simpler format
        try:
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump({"error": "Failed to serialize full data", 
                          "summary": audit_data["summary"]}, f, indent=2)
            logger.warning(f"Saved simplified audit data to {audit_file}")
        except Exception as e2:
            logger.error(f"Failed to save even simplified audit data: {str(e2)}")
    
    return audit_file

def main():
    """Main entry point for de-identification process."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="De-identify healthcare data files")
    parser.add_argument("--config", default="config/cleaning/deidentification.yaml",
                      help="Path to configuration file")
    parser.add_argument("--mapping", default="data/private_backup/id_mapping.parquet",
                      help="Path to ID mapping file")
    parser.add_argument("--patterns", default="config/cleaning/phi_patterns.yaml",
                      help="Path to PHI patterns file")
    parser.add_argument("--chunk-size", type=int, default=None,
                      help="Chunk size for large file processing (default: process whole file)")
    parser.add_argument("--output-dir", default="data/deidentified",
                      help="Directory for deidentified output files")
    parser.add_argument("--log-dir", default="outputs/audit/phi_removal_logs",
                      help="Directory for log files")
    parser.add_argument("--skip-mcinfo", action="store_true",
                      help="Skip processing mcinfo data")
    parser.add_argument("--skip-interview", action="store_true",
                      help="Skip processing interview data")
    parser.add_argument("--skip-result", action="store_true",
                      help="Skip processing result data")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting de-identification process")
    
    # Load configuration if it exists
    config = {
        "input": {
            "mcinfo": "data/raw/df_mcinfo_cleaned_final.parquet",
            "interview": "data/normalized/interview_per_exam_normalized.parquet",
            "result": "data/normalized/result_per_exam_normalized.parquet"
        },
        "output": {
            "mcinfo": os.path.join(args.output_dir, "df_mcinfo_deidentified.parquet"),
            "interview": os.path.join(args.output_dir, "interview_per_exam_deidentified.parquet"),
            "result": os.path.join(args.output_dir, "result_per_exam_deidentified.parquet")
        },
        "processing": {
            "chunk_size": args.chunk_size
        }
    }
    
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                # Update config with loaded values
                if loaded_config:
                    if 'input_files' in loaded_config:
                        config["input"].update(loaded_config["input_files"])
                    if 'output_files' in loaded_config:
                        config["output"].update(loaded_config["output_files"])
                    if 'processing' in loaded_config:
                        config["processing"].update(loaded_config["processing"])
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {str(e)}")
            logger.info("Using default configuration")
    else:
        logger.warning(f"Configuration file not found: {args.config}")
        logger.info("Using default configuration")
    
    # Load ID mapping
    try:
        logger.info(f"Loading ID mapping from {args.mapping}")
        mapping = pd.read_parquet(args.mapping)
        logger.info(f"Loaded mapping for {len(mapping):,} unique IDs")
        
        # Count IDs with keep_record=False
        filtered_ids = len(mapping[~mapping['keep_record']])
        logger.info(f"Mapping contains {filtered_ids:,} IDs with keep_record=False")
    except Exception as e:
        logger.error(f"Failed to load ID mapping: {str(e)}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file and collect statistics
    stats = {}
    
    # 1. Process mcinfo file (ID mapping and date shifting only)
    if not args.skip_mcinfo and os.path.exists(config["input"]["mcinfo"]):
        try:
            stats["mcinfo"] = process_mcinfo_file(
                config["input"]["mcinfo"],
                config["output"]["mcinfo"],
                mapping,
                chunk_size=config["processing"]["chunk_size"]
            )
        except Exception as e:
            logger.error(f"Failed to process mcinfo file: {str(e)}")
            stats["mcinfo"] = {"error": str(e)}
    else:
        if args.skip_mcinfo:
            logger.info("Skipping mcinfo processing as requested")
        else:
            logger.warning(f"Mcinfo file not found: {config['input']['mcinfo']}")
        stats["mcinfo"] = {"error": "Skipped or file not found"}

    # Initialize PHI scrubber for interview data
    try:
        logger.info(f"Initializing PHI scrubber for interview data with patterns from {args.patterns}")
        phi_scrubber_interview = PHIScrubber(args.patterns)
        pattern_count = sum(len(patterns) for cat, patterns in phi_scrubber_interview.compiled_patterns.items())
        logger.info(f"Initialized PHI scrubber with {len(phi_scrubber_interview.compiled_patterns)} categories, {pattern_count} patterns")
    except Exception as e:
        logger.error(f"Failed to initialize PHI scrubber: {str(e)}")
        return 1        
    
    # 2. Process interview file (ID mapping, date shifting, and PHI scrubbing)
    if not args.skip_interview and os.path.exists(config["input"]["interview"]):
        try:
            stats["interview"] = process_interview_file(
                config["input"]["interview"],
                config["output"]["interview"],
                mapping,
                phi_scrubber_interview
            )
        except Exception as e:
            logger.error(f"Failed to process interview file: {str(e)}")
            stats["interview"] = {"error": str(e)}
    else:
        if args.skip_interview:
            logger.info("Skipping interview processing as requested")
        else:
            logger.warning(f"Interview file not found: {config['input']['interview']}")
        stats["interview"] = {"error": "Skipped or file not found"}

    # Initialize PHI scrubber for result data
    try:
        logger.info(f"Initializing PHI scrubber for result data with patterns from {args.patterns}")
        phi_scrubber_result = PHIScrubber(args.patterns)
        pattern_count = sum(len(patterns) for cat, patterns in phi_scrubber_result.compiled_patterns.items())
        logger.info(f"Initialized PHI scrubber with {len(phi_scrubber_result.compiled_patterns)} categories, {pattern_count} patterns")
    except Exception as e:
        logger.error(f"Failed to initialize PHI scrubber: {str(e)}")
        return 1  
    
    # 3. Process result file (ID mapping, date shifting, and PHI scrubbing)
    if not args.skip_result and os.path.exists(config["input"]["result"]):
        try:
            stats["result"] = process_result_file(
                config["input"]["result"],
                config["output"]["result"],
                mapping,
                phi_scrubber_result
            )
        except Exception as e:
            logger.error(f"Failed to process result file: {str(e)}")
            stats["result"] = {"error": str(e)}
    else:
        if args.skip_result:
            logger.info("Skipping result processing as requested")
        else:
            logger.warning(f"Result file not found: {config['input']['result']}")
        stats["result"] = {"error": "Skipped or file not found"}
    
    # Save audit data
    audit_file = save_audit_data(stats, {
        "input_files": config["input"],
        "output_files": config["output"],
        "mapping_file": args.mapping,
        "patterns_file": args.patterns,
        "chunk_size": config["processing"]["chunk_size"]
    }, args.log_dir)
    
    # Print summary
    logger.info("\nDe-identification process complete")
    
    print("\n" + "="*60)
    print("DE-IDENTIFICATION PROCESS SUMMARY")
    print("="*60)
    
    if "mcinfo" in stats and "original_rows" in stats["mcinfo"]:
        print(f"Mcinfo data: {stats['mcinfo']['kept_rows']:,}/{stats['mcinfo']['original_rows']:,} rows kept "
              f"({stats['mcinfo']['kept_rows']/stats['mcinfo']['original_rows']*100:.1f}%)")
    
    if "interview" in stats and "original_rows" in stats["interview"]:
        print(f"Interview data: {stats['interview']['kept_rows']:,}/{stats['interview']['original_rows']:,} rows kept "
              f"({stats['interview']['kept_rows']/stats['interview']['original_rows']*100:.1f}%)")
        
        # Show PHI scrubbing stats
        if "phi_scrubbing" in stats["interview"]:
            phi_stats = stats["interview"]["phi_scrubbing"]
            print(f"  PHI replacements: {phi_stats['total_replacements']:,} in {phi_stats['processed_fields']:,} fields")
            for category, count in phi_stats["replacements_by_category"].items():
                if count > 0:
                    print(f"    {category}: {count:,}")
    
    if "result" in stats and "original_rows" in stats["result"]:
        print(f"Result data: {stats['result']['kept_rows']:,}/{stats['result']['original_rows']:,} rows kept "
              f"({stats['result']['kept_rows']/stats['result']['original_rows']*100:.1f}%)")
        
        # Show PHI scrubbing stats
        if "phi_scrubbing" in stats["result"]:
            phi_stats = stats["result"]["phi_scrubbing"]
            print(f"  PHI replacements: {phi_stats['total_replacements']:,} in {phi_stats['processed_fields']:,} fields")
            for category, count in phi_stats["replacements_by_category"].items():
                if count > 0:
                    print(f"    {category}: {count:,}")
    
    print("\nOutput files:")
    for file_type, path in config["output"].items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {file_type}: {path} ({size_mb:.1f} MB)")
    
    print(f"\nDetailed audit data saved to: {audit_file}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
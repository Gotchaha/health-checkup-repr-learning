# scripts/data_preparation/normalize_text.py

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add the project root to the Python path if needed
project_root = Path(__file__).parents[2]  # Go up two levels from scripts/data_preparation
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.cleaning.text_normalizer import process_interview_file, process_result_file

def setup_logging(log_dir):
    """Set up logging to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"text_normalization_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Normalize text in interview and result files')
    
    parser.add_argument('--interview-input', 
                        type=str, 
                        default='data/raw/interview_per_exam.parquet',
                        help='Path to the input interview Parquet file')
    
    parser.add_argument('--interview-output', 
                        type=str, 
                        default='data/normalized/interview_per_exam_normalized.parquet',
                        help='Path to save the normalized interview Parquet file')
    
    parser.add_argument('--result-input', 
                        type=str, 
                        default='data/raw/result_per_exam.parquet',
                        help='Path to the input result Parquet file')
    
    parser.add_argument('--result-output', 
                        type=str, 
                        default='data/normalized/result_per_exam_normalized.parquet',
                        help='Path to save the normalized result Parquet file')
    
    parser.add_argument('--log-dir', 
                        type=str, 
                        default='outputs/audit/normalizer_logs',
                        help='Directory to save log files')
    
    parser.add_argument('--process-interview', 
                        action='store_true',
                        help='Process the interview file')
    
    parser.add_argument('--process-result', 
                        action='store_true',
                        help='Process the result file')
    
    args = parser.parse_args()
    
    # If neither flag is specified, process both files
    if not args.process_interview and not args.process_result:
        args.process_interview = True
        args.process_result = True
    
    return args

def ensure_directory(file_path):
    """Ensure the directory for the file exists."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to run the text normalization process."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting text normalization process")
    
    # Create output directories if they don't exist
    if args.process_interview:
        ensure_directory(args.interview_output)
    if args.process_result:
        ensure_directory(args.result_output)
    
    # Process files
    start_time = time.time()
    
    if args.process_interview:
        logger.info(f"Processing interview file: {args.interview_input}")
        try:
            interview_stats = process_interview_file(args.interview_input, args.interview_output)
            logger.info(f"Interview file processed successfully: {interview_stats}")
        except Exception as e:
            logger.error(f"Error processing interview file: {e}", exc_info=True)
    
    if args.process_result:
        logger.info(f"Processing result file: {args.result_input}")
        try:
            result_stats = process_result_file(args.result_input, args.result_output)
            logger.info(f"Result file processed successfully: {result_stats}")
        except Exception as e:
            logger.error(f"Error processing result file: {e}", exc_info=True)
    
    # Log summary
    elapsed_time = time.time() - start_time
    logger.info(f"Text normalization completed in {elapsed_time:.2f} seconds")
    
    # Return stats if available
    stats = {}
    if args.process_interview and 'interview_stats' in locals():
        stats['interview'] = interview_stats
    if args.process_result and 'result_stats' in locals():
        stats['result'] = result_stats
    
    return stats

if __name__ == "__main__":
    main()
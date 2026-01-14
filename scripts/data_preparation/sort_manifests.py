# scripts/data_preparation/sort_manifests.py

import os
import time
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path


def compute_file_hash(file_path):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sort_manifest(input_path, output_path):
    """Sort manifest by person_id and ExamDate."""
    # Read the manifest
    table = pq.read_table(input_path)
    
    # Convert to pandas for easier sorting
    df = table.to_pandas()
    
    # Sort by person_id first, then by ExamDate
    df_sorted = df.sort_values(by=['person_id', 'ExamDate'])
    
    # Convert back to Arrow table, preserving schema
    table_sorted = pa.Table.from_pandas(df_sorted, schema=table.schema)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write sorted table
    pq.write_table(table_sorted, output_path)
    
    # Return file statistics
    return {
        'input_path': input_path,
        'input_rows': table.num_rows,
        'input_hash': compute_file_hash(input_path),
        'output_path': output_path,
        'output_rows': table_sorted.num_rows,
        'output_hash': compute_file_hash(output_path)
    }


def main():
    """Sort all manifest files and log the operation."""
    # Prepare paths
    splits = ['train_ssl', 'val_ssl', 'test_future']
    input_dir = 'data/splits/core'
    output_dir = 'data/splits/core/sorted'
    log_dir = 'outputs/audit/sort_manifests_logs'
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"{log_dir}/sort_log_{timestamp}.txt"
    
    # Process each manifest
    results = []
    for split in splits:
        input_path = f"{input_dir}/{split}.parquet"
        output_path = f"{output_dir}/{split}.parquet"
        
        print(f"Sorting {split} manifest...")
        start_time = time.time()
        result = sort_manifest(input_path, output_path)
        elapsed_time = time.time() - start_time
        
        # Add timing information
        result['processing_time'] = elapsed_time
        results.append(result)
        
        print(f"  Completed in {elapsed_time:.2f}s")
        print(f"  Input rows: {result['input_rows']}, Output rows: {result['output_rows']}")
        
    # Log results
    with open(log_path, 'w') as log_file:
        log_file.write(f"Manifest Sorting Operation Log - {timestamp}\n")
        log_file.write("=" * 80 + "\n\n")
        
        for result in results:
            log_file.write(f"Split: {Path(result['input_path']).stem}\n")
            log_file.write(f"  Input path: {result['input_path']}\n")
            log_file.write(f"  Input rows: {result['input_rows']}\n")
            log_file.write(f"  Input hash: {result['input_hash']}\n")
            log_file.write(f"  Output path: {result['output_path']}\n")
            log_file.write(f"  Output rows: {result['output_rows']}\n")
            log_file.write(f"  Output hash: {result['output_hash']}\n")
            log_file.write(f"  Processing time: {result['processing_time']:.2f}s\n")
            log_file.write("\n")
            
        log_file.write("\nVerification:\n")
        for result in results:
            equal_rows = result['input_rows'] == result['output_rows']
            different_hash = result['input_hash'] != result['output_hash']
            log_file.write(f"  {Path(result['input_path']).stem}: Row count preserved: {equal_rows}, Content changed (expected): {different_hash}\n")
            
    print(f"\nSorting complete! Log saved to {log_path}")


if __name__ == "__main__":
    main()
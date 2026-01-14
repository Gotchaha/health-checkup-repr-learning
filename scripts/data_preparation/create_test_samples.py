# scripts/data_preparation/create_test_samples.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def create_test_samples(interview_path, result_path, sample_size=10):
    """Create small sample files for testing.
    
    Args:
        interview_path: Path to the full interview Parquet file
        result_path: Path to the full result Parquet file
        sample_size: Number of rows to sample
    """
    # Create test data directory if it doesn't exist
    Path("tests/data/samples").mkdir(parents=True, exist_ok=True)
    
    # Create interview sample
    interview_table = pq.read_table(interview_path)
    interview_df = interview_table.to_pandas()
    interview_sample = interview_df.head(sample_size)
    
    sample_table = pa.Table.from_pandas(interview_sample)
    pq.write_table(sample_table, "tests/data/samples/sample_interview.parquet")
    
    # Create result sample
    result_table = pq.read_table(result_path)
    result_df = result_table.to_pandas()
    result_sample = result_df.head(sample_size)
    
    sample_table = pa.Table.from_pandas(result_sample)
    pq.write_table(sample_table, "tests/data/samples/sample_result.parquet")
    
    print(f"Created test samples with {sample_size} rows each")

if __name__ == "__main__":
    create_test_samples(
        "data/raw/interview_per_exam.parquet",
        "data/raw/result_per_exam.parquet"
    )
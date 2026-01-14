# scripts/data_preparation/create_deid_test_sample.py
"""
Generate test data samples for de-identification unit tests.
Creates small samples from real data files while preserving their structure.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Constants
SAMPLES_DIR = Path("tests/data/samples")
SAMPLE_MCINFO = SAMPLES_DIR / "deid_sample_mcinfo.parquet"
SAMPLE_INTERVIEW = SAMPLES_DIR / "deid_sample_interview.parquet"
SAMPLE_RESULT = SAMPLES_DIR / "deid_sample_result.parquet"

def sample_mcinfo_data(input_path, output_path, sample_size=1000, seed=42):
    """
    Create a sample from the mcinfo (test-per-row) data.
    
    Args:
        input_path: Path to mcinfo data file
        output_path: Path to save the sample
        sample_size: Number of rows to sample
        seed: Random seed for reproducibility
    """
    print(f"Creating sample from mcinfo data: {input_path}")
    
    try:
        # First get all unique IDs
        import pyarrow.parquet as pq
        
        # Read just the ID column for efficiency
        table = pq.read_table(input_path, columns=["AnonymousID"])
        id_col = table.to_pandas()
        unique_ids = id_col["AnonymousID"].unique()
        
        # Select a smaller subset of IDs
        np.random.seed(seed)
        sampled_ids = np.random.choice(unique_ids, size=min(30, len(unique_ids)), replace=False)
        print(f"Selected {len(sampled_ids)} unique person IDs for sampling")
        
        # Use PyArrow filters to read only the rows for these IDs
        filters = [('AnonymousID', 'in', sampled_ids.tolist())]
        table = pq.read_table(input_path, filters=filters)
        sample_df = table.to_pandas()
        
        # Limit to sample_size if needed
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(sample_size, random_state=seed)
        
        print(f"Created mcinfo sample with {len(sample_df)} rows from {len(sampled_ids)} persons")
        
        # Save sample
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        sample_df.to_parquet(output_path, index=False)
        print(f"Saved mcinfo sample to {output_path}")
        
        # Print sample stats
        num_text_fields = sum(1 for col in sample_df.columns if col.endswith('Text') or col == 'Comment')
        print(f"Sample stats: {len(sample_df)} rows, {sample_df['AnonymousID'].nunique()} unique IDs, {num_text_fields} text columns")
        
        return sample_df
        
    except Exception as e:
        print(f"Error creating mcinfo sample: {str(e)}")
        raise

def sample_interview_data(input_path, output_path, sample_size=50, seed=42):
    """
    Create a sample from the interview data.
    
    Args:
        input_path: Path to interview data file
        output_path: Path to save the sample
        sample_size: Number of rows to sample
        seed: Random seed for reproducibility
    """
    print(f"Creating sample from interview data: {input_path}")
    
    try:
        # Load interview data
        df = pd.read_parquet(input_path)
        print(f"Loaded interview data: {len(df)} rows")
        
        # Sample rows
        np.random.seed(seed)
        if len(df) <= sample_size:
            sample_df = df.copy()
        else:
            sample_df = df.sample(sample_size, random_state=seed)
        
        print(f"Created interview sample with {len(sample_df)} rows")
        
        # Save sample
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        sample_df.to_parquet(output_path, index=False)
        print(f"Saved interview sample to {output_path}")
        
        # Print sample stats
        print(f"Sample stats: {len(sample_df)} rows, {sample_df['AnonymousID'].nunique()} unique IDs")
        
        # Check if nested Interview field exists
        if 'Interview' in sample_df.columns:
            # Count average number of Q&A pairs
            qa_counts = sample_df['Interview'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            print(f"Average Q&A pairs per interview: {qa_counts.mean():.1f}")
        
        return sample_df
        
    except Exception as e:
        print(f"Error creating interview sample: {str(e)}")
        raise

def sample_result_data(input_path, output_path, sample_size=50, seed=42):
    """
    Create a sample from the result data.
    
    Args:
        input_path: Path to result data file
        output_path: Path to save the sample
        sample_size: Number of rows to sample
        seed: Random seed for reproducibility
    """
    print(f"Creating sample from result data: {input_path}")
    
    try:
        # Load result data
        df = pd.read_parquet(input_path)
        print(f"Loaded result data: {len(df)} rows")
        
        # Sample rows
        np.random.seed(seed)
        if len(df) <= sample_size:
            sample_df = df.copy()
        else:
            sample_df = df.sample(sample_size, random_state=seed)
        
        print(f"Created result sample with {len(sample_df)} rows")
        
        # Save sample
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        sample_df.to_parquet(output_path, index=False)
        print(f"Saved result sample to {output_path}")
        
        # Print sample stats
        text_cols = [col for col in sample_df.columns if col in ['GradeDescription', 'ResultText']]
        print(f"Sample stats: {len(sample_df)} rows, {sample_df['AnonymousID'].nunique()} unique IDs")
        print(f"Text columns: {', '.join(text_cols)}")
        
        return sample_df
        
    except Exception as e:
        print(f"Error creating result sample: {str(e)}")
        raise

def main():
    """Main function to generate all sample data files."""
    print("Generating test data samples for de-identification testing...")
    
    # Create directories
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    # Input file paths
    mcinfo_path = "data/raw/df_mcinfo_cleaned_final.parquet"
    interview_path = "data/normalized/interview_per_exam_normalized.parquet"
    result_path = "data/normalized/result_per_exam_normalized.parquet"
    
    # Create samples
    try:
        mcinfo_sample = sample_mcinfo_data(mcinfo_path, SAMPLE_MCINFO, sample_size=1000)
    except Exception as e:
        print(f"Failed to generate mcinfo sample: {str(e)}")
    
    try:
        interview_sample = sample_interview_data(interview_path, SAMPLE_INTERVIEW, sample_size=50)
    except Exception as e:
        print(f"Failed to generate interview sample: {str(e)}")
    
    try:
        result_sample = sample_result_data(result_path, SAMPLE_RESULT, sample_size=50)
    except Exception as e:
        print(f"Failed to generate result sample: {str(e)}")
    
    print("\nSample data generation complete. Files created:")
    if os.path.exists(SAMPLE_MCINFO):
        print(f"- {SAMPLE_MCINFO}")
    if os.path.exists(SAMPLE_INTERVIEW):
        print(f"- {SAMPLE_INTERVIEW}")
    if os.path.exists(SAMPLE_RESULT):
        print(f"- {SAMPLE_RESULT}")

if __name__ == "__main__":
    main()
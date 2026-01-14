# scripts/parquet_inspector.py
"""
Parquet File Inspector

A utility script to explore and visualize the contents of Parquet files
in the research project. Handles nested structures and provides basic stats.
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pathlib

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# Now you can construct paths relative to project root
def resolve_path(path_str):
    path = pathlib.Path(path_str)
    if not path.is_absolute():
        # Check if path exists relative to current directory
        if not path.exists():
            # Try relative to project root
            project_relative_path = PROJECT_ROOT / path
            if project_relative_path.exists():
                return project_relative_path
    return path


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get basic file information"""
    stats = {}
    stats["file_path"] = os.path.abspath(file_path)
    stats["file_size"] = f"{os.path.getsize(file_path) / (1024 * 1024):.2f} MB"
    stats["modified_time"] = pd.Timestamp(os.path.getmtime(file_path), unit='s')
    return stats


def get_schema_info(file_path: str) -> pa.Schema:
    """Extract schema information from the parquet file"""
    return pq.read_schema(file_path)


def safe_isna(x):
    """Safely check if a value is NA, handling numpy arrays"""
    if isinstance(x, np.ndarray):
        return False  # Arrays themselves are not NA
    try:
        return pd.isna(x)
    except (TypeError, ValueError):
        return False


def format_nested_value(value: Any, max_items: int = 3, max_depth: int = 2, 
                        current_depth: int = 0) -> str:
    """Format nested values for display with truncation for deep nesting"""
    if current_depth >= max_depth:
        if isinstance(value, list):
            return f"[... {len(value)} items]"
        elif isinstance(value, dict):
            return f"{{... {len(value)} keys}}"
        elif isinstance(value, np.ndarray):
            return f"ndarray({value.shape}, {value.dtype})"
        else:
            return str(value)
    
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        elif len(value) > max_items:
            formatted = [format_nested_value(v, max_items, max_depth, current_depth + 1) 
                         for v in value[:max_items]]
            return f"[{', '.join(formatted)}, ... ({len(value) - max_items} more)]"
        else:
            formatted = [format_nested_value(v, max_items, max_depth, current_depth + 1) 
                         for v in value]
            return f"[{', '.join(formatted)}]"
    
    elif isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        elif len(value) > max_items:
            keys = list(value.keys())[:max_items]
            formatted = [f"{k}: {format_nested_value(value[k], max_items, max_depth, current_depth + 1)}" 
                         for k in keys]
            return f"{{{', '.join(formatted)}, ... ({len(value) - max_items} more keys)}}"
        else:
            formatted = [f"{k}: {format_nested_value(v, max_items, max_depth, current_depth + 1)}" 
                         for k, v in value.items()]
            return f"{{{', '.join(formatted)}}}"
    
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return f"ndarray(empty, {value.dtype})"
        elif value.size > max_items:
            return f"ndarray({value.shape}, {value.dtype})"
        # Handle structured arrays (like those with named fields)
        elif value.dtype.names:
            # For structured arrays, show field names and first item
            if value.size == 1:
                return f"ndarray(fields={value.dtype.names})"
            else:
                return f"ndarray({value.shape}, fields={value.dtype.names})"
        else:
            # For simple arrays, show a few values
            items = value.flat[:max_items]
            formatted = ", ".join(str(i) for i in items)
            if value.size > max_items:
                formatted += f", ... ({value.size - max_items} more)"
            return f"ndarray([{formatted}], {value.dtype})"
    
    else:
        if isinstance(value, str) and len(value) > 50:
            return f"{value[:47]}..."
        return str(value)


def format_df_for_display(df: pd.DataFrame, max_rows: int = 5) -> pd.DataFrame:
    """Format a dataframe for display, handling nested structures"""
    # Create a copy to avoid modifying the original
    display_df = df.head(max_rows).copy()
    
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            # Check if column contains lists or dicts
            sample = None
            if not display_df[col].isna().all():
                for idx, val in display_df[col].dropna().items():
                    sample = val
                    break
                    
            if isinstance(sample, (list, dict, np.ndarray)):
                # Use a safe way to format each cell, handling numpy arrays properly
                formatted_col = []
                for idx, val in display_df[col].items():
                    if safe_isna(val):
                        formatted_col.append(val)
                    else:
                        formatted_col.append(format_nested_value(val))
                display_df[col] = formatted_col
            elif isinstance(sample, str) and len(sample) > 50:
                formatted_col = []
                for idx, val in display_df[col].items():
                    if safe_isna(val):
                        formatted_col.append(val)
                    elif isinstance(val, str) and len(val) > 50:
                        formatted_col.append(f"{val[:47]}...")
                    else:
                        formatted_col.append(val)
                display_df[col] = formatted_col
    
    return display_df


def print_table(data, headers=None):
    """Print data in a tabular format"""
    if not data:
        return
    
    if headers:
        # Print header row
        header_str = " | ".join(str(h) for h in headers)
        print(header_str)
        print("-" * len(header_str))
    
    # Print data rows
    for row in data:
        if isinstance(row, dict) and headers:
            print(" | ".join(str(row.get(h, "")) for h in headers))
        else:
            print(" | ".join(str(cell) for cell in row))


def get_column_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate statistics for each column in the dataframe"""
    stats = []
    for col in df.columns:
        col_type = df[col].dtype
        n_nulls = df[col].isna().sum()
        null_pct = f"{100 * n_nulls / len(df):.1f}%"
        
        # Check if this is a nested column and what kind of nested structure
        is_nested = False
        has_numpy_array = False
        nested_item_type = None
        
        # Get a non-null sample if available
        sample = None
        if not df[col].isna().all():
            for idx, val in df[col].dropna().items():
                sample = val
                break
                
        if isinstance(sample, (list, dict)):
            is_nested = True
            if sample and isinstance(sample, list):
                if sample and isinstance(sample[0], dict):
                    nested_item_type = "dict"
                else:
                    nested_item_type = type(sample[0]).__name__ if sample else None
        elif isinstance(sample, np.ndarray):
            is_nested = True
            has_numpy_array = True
            nested_item_type = f"numpy.ndarray[{sample.dtype}]"
            
        col_stat = {
            "Column": col,
            "Type": str(col_type),
            "Nulls": f"{null_pct} ({n_nulls})",
            "Nested": "Yes" if is_nested else "No"
        }
        
        # Add statistics specific to numpy arrays
        if has_numpy_array:
            arrays = df[col].dropna()
            # Safe function to get array lengths
            def get_array_len(arr):
                return len(arr) if isinstance(arr, np.ndarray) else 0
            
            if len(arrays) > 0:
                # Calculate array length statistics
                array_lengths = arrays.apply(get_array_len)
                col_stat["Array Items"] = f"Min: {array_lengths.min()}, Max: {array_lengths.max()}, Avg: {array_lengths.mean():.1f}"
                col_stat["Array Type"] = nested_item_type
                
                # Add example of array structure if possible
                if sample is not None and len(sample) > 0:
                    # For numpy arrays containing dictionaries (like Interview data)
                    if sample.dtype.names:  # Structured array
                        col_stat["Structure"] = f"Structured array with fields: {sample.dtype.names}"
                    else:
                        col_stat["Structure"] = f"Array shape: {sample.shape}, dtype: {sample.dtype}"
        
        # Add more stats for non-nested columns or handle other types of nesting
        elif not is_nested:
            if pd.api.types.is_numeric_dtype(col_type):
                col_stat["Min"] = df[col].min()
                col_stat["Max"] = df[col].max()
                if not pd.api.types.is_integer_dtype(col_type):
                    col_stat["Mean"] = f"{df[col].mean():.2f}"
            
            # Add stats for categorical/string columns
            if (pd.api.types.is_object_dtype(col_type) or 
                pd.api.types.is_string_dtype(col_type) or
                isinstance(col_type, pd.api.types.CategoricalDtype)):
                # Only calculate nunique for hashable types
                try:
                    n_unique = df[col].nunique()
                    col_stat["Unique"] = f"{n_unique} ({100 * n_unique / len(df):.1f}%)"
                    
                    if n_unique < 10:  # Show value counts for low-cardinality columns
                        top_values = df[col].value_counts().head(3)
                        values_str = ", ".join(f"{v}: {c}" for v, c in top_values.items())
                        col_stat["Top Values"] = values_str
                except (TypeError, ValueError):
                    # More comprehensive error handling
                    col_stat["Unique"] = "Cannot compute (unhashable or complex type)"
        # Handle other nested types (lists, dicts)
        elif is_nested and not has_numpy_array:
            if nested_item_type:
                col_stat["Nested Type"] = nested_item_type
            
            # Analyze list lengths for list columns
            if isinstance(sample, list):
                list_lengths = df[col].dropna().apply(len)
                if not list_lengths.empty:
                    col_stat["List Items"] = f"Min: {list_lengths.min()}, Max: {list_lengths.max()}, Avg: {list_lengths.mean():.1f}"
        
        stats.append(col_stat)
    
    return stats


def inspect_parquet_file(file_path: str, max_rows: int = 5) -> None:
    """Main function to inspect a parquet file and print a summary"""
    print("\n" + "="*80)
    print(f"PARQUET FILE INSPECTION: {os.path.basename(file_path)}")
    print("="*80)
    
    # Get file info
    file_info = get_file_info(file_path)
    print("\nFILE INFORMATION:")
    print(f"Path: {file_info['file_path']}")
    print(f"Size: {file_info['file_size']}")
    print(f"Last Modified: {file_info['modified_time']}")
    
    # Get schema information
    schema = get_schema_info(file_path)
    print("\nSCHEMA:")
    print(schema)
    
    # Read the data
    try:
        # First try reading with pyarrow
        table = pq.read_table(file_path)
        df = table.to_pandas()
    except Exception as e:
        print(f"Error reading with PyArrow: {e}")
        print("Falling back to pandas...")
        try:
            df = pd.read_parquet(file_path)
        except Exception as e2:
            print(f"Error reading with pandas: {e2}")
            print("Unable to read the file. Exiting.")
            return
    
    # Print basic data info
    print("\nDATA SUMMARY:")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    
    # Print column statistics
    print("\nCOLUMN STATISTICS:")
    col_stats = get_column_stats(df)
    headers = ["Column", "Type", "Nulls", "Nested"]
    print_table([{k: v for k, v in stat.items() if k in headers} for stat in col_stats], headers)
    
    # Additional stats for numeric and categorical columns
    print("\nADDITIONAL COLUMN DETAILS:")
    for stat in col_stats:
        print(f"\n{stat['Column']}:")
        
        # Print details based on column type
        if "Min" in stat:
            print(f"  Range: {stat['Min']} to {stat['Max']}")
            if "Mean" in stat:
                print(f"  Mean: {stat['Mean']}")
        if "Unique" in stat:
            print(f"  Unique Values: {stat['Unique']}")
            if "Top Values" in stat:
                print(f"  Top Values: {stat['Top Values']}")
        if "Array Items" in stat:
            print(f"  {stat['Array Items']}")
            if "Array Type" in stat:
                print(f"  Type: {stat['Array Type']}")
            if "Structure" in stat:
                print(f"  {stat['Structure']}")
        if "List Items" in stat:
            print(f"  {stat['List Items']}")
            if "Nested Type" in stat:
                print(f"  Contains: {stat['Nested Type']}")
    
    # Format and display sample rows
    try:
        display_df = format_df_for_display(df, max_rows=max_rows)
        print(f"\nSAMPLE ROWS (showing {min(max_rows, len(df))} of {len(df):,} rows):")
        print(display_df.to_string())
    except Exception as e:
        print(f"\nError formatting data for display: {e}")
        print("Skipping sample row display.")
    
    # Special handling for known nested structures
    if 'tests' in df.columns:
        # This is likely the exam-level data
        tests_count = df['tests'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        print("\nTESTS PER EXAM STATISTICS:")
        print(f"  Min: {tests_count.min()}")
        print(f"  Max: {tests_count.max()}")
        print(f"  Mean: {tests_count.mean():.2f}")
        print(f"  Median: {tests_count.median()}")
        
        # Show sample test structure if available
        if not df['tests'].isna().all():
            sample_row = None
            for _, row in df.iterrows():
                if isinstance(row['tests'], list) and len(row['tests']) > 0:
                    sample_row = row
                    break
                    
            if sample_row is not None:
                print("\nSAMPLE TEST STRUCTURE:")
                print(f"  {sample_row['tests'][0]}")
    
    if 'Interview' in df.columns:
        # This is likely the interview data
        def safe_len(x):
            if isinstance(x, (list, np.ndarray)):
                return len(x)
            return 0
            
        interview_count = df['Interview'].apply(safe_len)
        print("\nQUESTIONS PER INTERVIEW STATISTICS:")
        print(f"  Min: {interview_count.min()}")
        print(f"  Max: {interview_count.max()}")
        print(f"  Mean: {interview_count.mean():.2f}")
        print(f"  Median: {interview_count.median()}")
        
        # Sample question/answer if available
        sample_row = None
        for _, row in df.iterrows():
            if isinstance(row['Interview'], (list, np.ndarray)) and safe_len(row['Interview']) > 0:
                sample_row = row
                break
                
        if sample_row is not None:
            print("\nSAMPLE QUESTION/ANSWER STRUCTURE:")
            try:
                sample_qa = sample_row['Interview'][0]
                
                if isinstance(sample_row['Interview'], np.ndarray) and hasattr(sample_row['Interview'].dtype, 'names') and sample_row['Interview'].dtype.names:
                    # Structured array
                    fields = sample_row['Interview'].dtype.names
                    print(f"  Fields: {fields}")
                    for field in fields:
                        print(f"  {field}: {sample_qa[field]}")
                elif isinstance(sample_qa, dict):
                    # Dictionary structure
                    print(f"  Question: {sample_qa.get('question', 'N/A')}")
                    print(f"  Answer: {sample_qa.get('answer', 'N/A')}")
                else:
                    # Fall back to printing the raw structure
                    print(f"  Structure: {type(sample_qa).__name__}")
                    print(f"  Content: {str(sample_qa)[:100]}...")
            except Exception as e:
                print(f"  Error displaying sample: {e}")
    
    print("\n" + "="*80)


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Inspect Parquet files for the health checkup research project.")
    parser.add_argument("file_path", help="Path to the Parquet file to inspect")
    parser.add_argument("-r", "--rows", type=int, default=5, 
                      help="Number of sample rows to display (default: 5)")
    
    args = parser.parse_args()
    
    file_path = resolve_path(args.file_path)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    
    inspect_parquet_file(file_path, max_rows=args.rows)


if __name__ == "__main__":
    main()
# scripts/embedders/gen_cat_vocab.py

"""
Generate categorical vocabulary YAML for the CategoricalEmbedder.

This script processes the test code master file and creates a vocabulary mapping
test codes and their categorical values to unique indices.
"""

import argparse
import csv
import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate categorical vocabulary YAML for embedder")
    parser.add_argument(
        "--master-file",
        type=str,
        default="data/external/test_code_master/Fmaster.csv",
        help="Path to the test code master file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="config/embedders/cat_vocab.yaml",
        help="Output path for the vocabulary YAML file"
    )
    return parser.parse_args()


def load_master_file(file_path: str) -> List[Dict[str, str]]:
    """Load the master file containing test codes and their encodings."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        test_codes = []
        for row in reader:
            test_codes.append({
                "name": row[0],
                "code": row[1],
                "type": row[2],
                "encoding": row[3]
            })
    return test_codes


def is_categorical_test(encoding: str) -> bool:
    """Check if a test is categorical based on whether it has encoding values."""
    return bool(encoding.strip())


def create_vocabulary(test_codes: List[Dict[str, str]]) -> Dict:
    """
    Create a vocabulary mapping test codes and categorical values to indices.
    
    Returns:
        Dictionary containing token_to_idx, idx_to_token, and metadata
    """
    token_to_idx = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}  # Global PAD, UNK and MASK tokens at index 0, 1, 2
    idx_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<MASK>"}
    
    n_tests = 0
    n_categorical_tests = 0
    n_categorical_values = 0
    
    current_idx = 3
    
    for test in test_codes:
        code = test["code"]
        encoding = test["encoding"]
        
        # Add the test code itself
        token_to_idx[code] = current_idx
        idx_to_token[current_idx] = code
        current_idx += 1
        n_tests += 1
        
        # If this is a categorical test, add entries for each possible value
        if is_categorical_test(encoding):
            try:
                values = json.loads(encoding)
                n_categorical_tests += 1
                
                # Add special UNK token for this categorical test
                missing_token = f"{code}=UNK"
                token_to_idx[missing_token] = current_idx
                idx_to_token[current_idx] = missing_token
                current_idx += 1
                
                # Add entries for each categorical value (using the keys)
                for value_key in values:
                    token = f"{code}={value_key}"
                    token_to_idx[token] = current_idx
                    idx_to_token[current_idx] = token
                    current_idx += 1
                    n_categorical_values += 1
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not parse encoding for {code}: {encoding}")
    
    vocabulary = {
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "vocab_size": len(token_to_idx),
        "metadata": {
            "n_tests": n_tests,
            "n_categorical_tests": n_categorical_tests,
            "n_categorical_values": n_categorical_values
        }
    }
    
    return vocabulary


def represent_str(dumper, data):
    """Force PyYAML to quote all string values."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

    
def save_vocabulary(vocabulary: Dict, output_file: str) -> None:
    """Save the vocabulary mappings to a YAML file with consistent quoting."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Register string representer to force quoting of all string values
    yaml.add_representer(str, represent_str)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(vocabulary, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Vocabulary saved to {output_file}")
    print(f"Total vocabulary size: {vocabulary['vocab_size']}")
    print(f"Number of tests: {vocabulary['metadata']['n_tests']}")
    print(f"Number of categorical tests: {vocabulary['metadata']['n_categorical_tests']}")
    print(f"Number of categorical values: {vocabulary['metadata']['n_categorical_values']}")


def main():
    """Main execution function."""
    args = parse_args()
    
    master_file = Path(args.master_file).resolve()
    output_file = Path(args.output_file).resolve()
    
    print(f"Loading master file from: {master_file}")
    test_codes = load_master_file(master_file)
    print(f"Loaded {len(test_codes)} test codes")
    
    vocabulary = create_vocabulary(test_codes)
    
    save_vocabulary(vocabulary, output_file)


if __name__ == "__main__":
    main()
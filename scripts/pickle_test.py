# scripts/pickle_test.py
"""
Simple test script to verify HealthExamDataset can be pickled for multiprocessing.
"""

import sys
import os
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

from src.models.dataset import HealthExamDataset


def main():
    print("Testing HealthExamDataset pickling...")
    
    # Create dataset with default parameters
    dataset = HealthExamDataset('train_ssl')
    
    # Test if the dataset can be pickled after your fix
    pickled = pickle.dumps(dataset)
    unpickled = pickle.loads(pickled)
    print("Dataset pickling successful!")


if __name__ == "__main__":
    main()
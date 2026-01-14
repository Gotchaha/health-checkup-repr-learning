# tests/models/test_sampler.py

import sys
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set
from src.models.sampler import PersonBatchSampler, InfinitePersonBatchSampler
from src.models.dataset import HealthExamDataset
from src.models.collate_fn import collate_exams
from src.models.embedders.CategoricalEmbedder import CategoricalEmbedder
from src.models.embedders.TextEmbedder import TextEmbedder


def test_basic_functionality():
    """Test basic sampler initialization and iteration."""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)
    
    # Use train split manifest (sorted)
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    
    if not Path(manifest_path).exists():
        print(f"FAIL: Manifest file not found: {manifest_path}")
        print("Please ensure the sorted manifest files exist.")
        return False
    
    # Test initialization
    print("Testing sampler initialization...")
    batch_size = 8
    sampler = PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    print(f"PASS: Sampler initialized successfully")
    print(f"   Total persons: {len(sampler.persons)}")
    print(f"   Total exams: {sampler.total_exams}")
    print(f"   Expected batches: {len(sampler)}")
    
    # Test iteration
    print("\nTesting batch generation...")
    batch_count = 0
    total_indices = set()
    batch_sizes = []
    
    for batch_indices in sampler:
        batch_count += 1
        batch_size_actual = len(batch_indices)
        batch_sizes.append(batch_size_actual)
        
        # Track all indices to check for duplicates
        for idx in batch_indices:
            if idx in total_indices:
                print(f"FAIL: Duplicate index found: {idx}")
                return False
            total_indices.add(idx)
        
        # Print first few batches for inspection
        if batch_count <= 3:
            print(f"   Batch {batch_count}: {batch_size_actual} exams, indices sample: {batch_indices[:3]}...")
    
    # Verify batch statistics
    print(f"\nBatch generation complete:")
    print(f"   Generated batches: {batch_count}")
    print(f"   Expected batches: {len(sampler)}")
    print(f"   Total indices collected: {len(total_indices)}")
    print(f"   Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, target={batch_size}")
    
    success = (batch_count == len(sampler) and 
               len(total_indices) == sampler.total_exams)
    
    if success:
        print("PASS: Basic functionality test PASSED")
    else:
        print("FAIL: Basic functionality test FAILED")
    
    return success


def test_person_grouping():
    """Verify person-level exam grouping and chronological order."""
    print("\n" + "=" * 60)
    print("TEST 2: Person-Level Grouping")
    print("=" * 60)
    
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    batch_size = 12
    
    sampler = PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for deterministic testing
        drop_last=False
    )
    
    # Load manifest for verification
    df = pd.read_parquet(manifest_path).reset_index(drop=True)
    
    print("Testing person grouping and chronological order...")
    
    person_violations = 0
    chronological_violations = 0
    batches_analyzed = 0
    
    for batch_indices in sampler:
        batches_analyzed += 1
        
        # Get exam data for this batch
        batch_data = df.iloc[batch_indices]
        
        # Check within-person chronological order
        for person_id in batch_data['person_id'].unique():
            person_exams = batch_data[batch_data['person_id'] == person_id].sort_values('ExamDate')
            person_indices = person_exams.index.tolist()
            
            # Check if indices are in ascending order (chronological)
            if person_indices != sorted(person_indices):
                chronological_violations += 1
                if chronological_violations <= 3:  # Print first few violations
                    print(f"   WARNING: Chronological violation for person {person_id} in batch {batches_analyzed}")
        
        # Analyze person distribution in batch
        person_counts = batch_data['person_id'].value_counts()
        if batches_analyzed <= 3:
            print(f"   Batch {batches_analyzed}: {len(person_counts)} persons, "
                  f"max exams per person: {person_counts.max()}")
        
        # Stop after analyzing several batches for efficiency
        if batches_analyzed >= 10:
            break
    
    print(f"\nPerson grouping analysis complete:")
    print(f"   Batches analyzed: {batches_analyzed}")
    print(f"   Chronological violations: {chronological_violations}")
    
    # Test person splitting behavior
    print("\nTesting person splitting across batches...")
    person_batch_map = defaultdict(list)
    
    batch_num = 0
    for batch_indices in sampler:
        batch_num += 1
        batch_data = df.iloc[batch_indices]
        
        for person_id in batch_data['person_id'].unique():
            person_batch_map[person_id].append(batch_num)
        
        if batch_num >= 20:  # Sample first 20 batches
            break
    
    # Count persons appearing in multiple batches
    split_persons = sum(1 for batches in person_batch_map.values() if len(batches) > 1)
    total_persons = len(person_batch_map)
    
    print(f"   Persons in sample: {total_persons}")
    print(f"   Persons split across batches: {split_persons} ({split_persons/total_persons*100:.1f}%)")
    
    success = chronological_violations == 0
    
    if success:
        print("PASS: Person grouping test PASSED")
    else:
        print("FAIL: Person grouping test FAILED")
    
    return success


def test_shuffle_behavior():
    """Test shuffle vs no-shuffle behavior."""
    print("\n" + "=" * 60)
    print("TEST 3: Shuffle Behavior")
    print("=" * 60)
    
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    batch_size = 10
    
    print("Testing shuffle behavior...")
    
    # Test without shuffle (should be deterministic)
    sampler_no_shuffle = PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    # Get first 5 batches from two iterations without shuffle
    batches_1 = []
    batches_2 = []
    
    for i, batch in enumerate(sampler_no_shuffle):
        if i >= 5:
            break
        batches_1.append(batch)
    
    for i, batch in enumerate(sampler_no_shuffle):
        if i >= 5:
            break
        batches_2.append(batch)
    
    # Check deterministic behavior
    deterministic = all(b1 == b2 for b1, b2 in zip(batches_1, batches_2))
    print(f"   No-shuffle deterministic: {'YES' if deterministic else 'NO'}")
    
    # Test with shuffle (should be different)
    sampler_shuffle = PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Get first 5 batches from two iterations with shuffle
    batches_shuffle_1 = []
    batches_shuffle_2 = []
    
    for i, batch in enumerate(sampler_shuffle):
        if i >= 5:
            break
        batches_shuffle_1.append(batch)
    
    for i, batch in enumerate(sampler_shuffle):
        if i >= 5:
            break
        batches_shuffle_2.append(batch)
    
    # Check if shuffle produces different orders
    different = any(b1 != b2 for b1, b2 in zip(batches_shuffle_1, batches_shuffle_2))
    print(f"   Shuffle produces different orders: {'YES' if different else 'NO'}")
    
    success = deterministic and different
    
    if success:
        print("PASS: Shuffle behavior test PASSED")
    else:
        print("FAIL: Shuffle behavior test FAILED")
    
    return success


def test_batch_composition():
    """Analyze actual batch composition in detail."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Composition Analysis")
    print("=" * 60)
    
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    batch_size = 16
    
    sampler = PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Load manifest for analysis
    df = pd.read_parquet(manifest_path).reset_index(drop=True)
    
    print("Analyzing batch composition...")
    
    batch_sizes = []
    persons_per_batch = []
    max_exams_per_person_batch = []
    
    batches_to_analyze = 50  # Analyze first 50 batches
    
    for batch_num, batch_indices in enumerate(sampler):
        if batch_num >= batches_to_analyze:
            break
        
        batch_data = df.iloc[batch_indices]
        
        batch_size_actual = len(batch_indices)
        person_counts = batch_data['person_id'].value_counts()
        num_persons = len(person_counts)
        max_exams_per_person = person_counts.max()
        
        batch_sizes.append(batch_size_actual)
        persons_per_batch.append(num_persons)
        max_exams_per_person_batch.append(max_exams_per_person)
        
        # Print details for first few batches
        if batch_num < 5:
            print(f"   Batch {batch_num + 1}: {batch_size_actual} exams, "
                  f"{num_persons} persons, max {max_exams_per_person} exams/person")
    
    print(f"\nBatch composition statistics (n={len(batch_sizes)}):")
    print(f"   Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "
          f"mean={sum(batch_sizes)/len(batch_sizes):.1f}")
    print(f"   Persons per batch: min={min(persons_per_batch)}, max={max(persons_per_batch)}, "
          f"mean={sum(persons_per_batch)/len(persons_per_batch):.1f}")
    print(f"   Max exams per person in batch: min={min(max_exams_per_person_batch)}, "
          f"max={max(max_exams_per_person_batch)}")
    
    # Test drop_last behavior
    print("\nTesting drop_last behavior...")
    
    sampler_drop = PersonBatchSampler(manifest_path, batch_size, shuffle=False, drop_last=True)
    sampler_no_drop = PersonBatchSampler(manifest_path, batch_size, shuffle=False, drop_last=False)
    
    batches_drop = list(sampler_drop)
    batches_no_drop = list(sampler_no_drop)
    
    print(f"   drop_last=True: {len(batches_drop)} batches")
    print(f"   drop_last=False: {len(batches_no_drop)} batches")
    print(f"   Last batch size (no drop): {len(batches_no_drop[-1]) if batches_no_drop else 0}")
    
    # Verify all batches have correct size when drop_last=True
    all_correct_size = all(len(batch) == batch_size for batch in batches_drop)
    print(f"   All batches correct size (drop=True): {'YES' if all_correct_size else 'NO'}")
    
    success = all_correct_size and len(batches_no_drop) >= len(batches_drop)
    
    if success:
        print("PASS: Batch composition test PASSED")
    else:
        print("FAIL: Batch composition test FAILED")
    
    return success


def test_integration_with_dataset():
    """Test integration with HealthExamDataset and collate_exams."""
    print("\n" + "=" * 60)
    print("TEST 5: Dataset Integration")
    print("=" * 60)
    
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    batch_size = 4  # Small batch size for testing
    
    print("Testing integration with HealthExamDataset...")
    
    # Create sampler
    sampler = PersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    # Create dataset
    try:
        dataset = HealthExamDataset(
            split_name="train_ssl",
            use_result=True,
            use_interview=False  # Skip interview for faster testing
        )
        print("PASS: Dataset created successfully")
    except Exception as e:
        print(f"FAIL: Dataset creation failed: {e}")
        return False
    
    # Test that sampler indices work with dataset
    print("Testing sampler indices with dataset...")
    
    try:
        batch_indices = next(iter(sampler))
        batch_data = []
        
        for idx in batch_indices:
            sample = dataset[idx]
            batch_data.append(sample)
        
        print(f"PASS: Successfully loaded {len(batch_data)} samples from dataset")
        
        # Check sample structure
        sample = batch_data[0]
        expected_keys = ['exam_id', 'person_id', 'ExamDate', 'tests', 'birth_year', 'gender']
        missing_keys = [key for key in expected_keys if key not in sample]
        
        if missing_keys:
            print(f"FAIL: Missing keys in sample: {missing_keys}")
            return False
        
        print("PASS: Sample structure validated")
        
    except Exception as e:
        print(f"FAIL: Dataset integration failed: {e}")
        return False
    
    # Test with collate_exams (simplified)
    print("Testing with collate_exams...")
    
    try:
        # Create minimal embedders for testing
        code_embedder = CategoricalEmbedder.from_config("config/embedders/categorical_embedder.yaml")
        text_embedder = TextEmbedder.from_config("config/embedders/text_embedder.yaml")
        
        print("PASS: Embedders loaded successfully")
        
        # Test collate function
        # NEW interface (config-based)
        # Need to create a minimal config for testing
        test_config = {
            'training': {
                'p_mlm': 0.15,
                'p_mcm': 0.15,
                'p_cvr': 0.20,
                'p_mcc': 0.20,
                'mcc': {
                    'enabled': True,
                    'K': 5,
                    'noise': {
                        'gaussian_scales': [0.05, 0.20],
                        'mix_probs': [0.5, 0.35, 0.15],
                        'large_min_norm_dist': 0.3
                    }
                }
            }
        }
        
        batch_output = collate_exams(
            batch_data,
            code_embedder,
            text_embedder,
            config=test_config,
            device="cpu"
        )
        
        print("PASS: collate_exams executed successfully")
        print(f"   Batch metadata: B={batch_output['B']}, T_max={batch_output['T_max']}")
        print(f"   Segment lengths: {batch_output['segment_lengths']}")
        
        # Verify segment_lengths makes sense
        total_batch_size = sum(batch_output['segment_lengths'])
        if total_batch_size != len(batch_data):
            print(f"FAIL: Segment lengths sum ({total_batch_size}) != batch size ({len(batch_data)})")
            return False
        
        print("PASS: Segment lengths validated")
        
    except Exception as e:
        print(f"FAIL: collate_exams integration failed: {e}")
        return False
    
    print("PASS: Dataset integration test PASSED")
    return True


def test_statistics():
    """Verify sampler statistics and __len__ calculation."""
    print("\n" + "=" * 60)
    print("TEST 6: Statistics Verification")
    print("=" * 60)
    
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    
    # Test different configurations
    configs = [
        {"batch_size": 8, "drop_last": True},
        {"batch_size": 8, "drop_last": False},
        {"batch_size": 16, "drop_last": True},
        {"batch_size": 32, "drop_last": False}
    ]
    
    print("Testing statistics for different configurations...")
    
    all_passed = True
    
    for config in configs:
        batch_size = config["batch_size"]
        drop_last = config["drop_last"]
        
        sampler = PersonBatchSampler(
            manifest_path=manifest_path,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last
        )
        
        # Get statistics
        stats = sampler.get_stats()
        predicted_len = len(sampler)
        
        # Count actual batches
        actual_batches = sum(1 for _ in sampler)
        
        # Calculate expected length
        if drop_last:
            expected_len = stats['total_exams'] // batch_size
        else:
            expected_len = (stats['total_exams'] + batch_size - 1) // batch_size
        
        # Verify calculations
        len_correct = (predicted_len == actual_batches == expected_len)
        
        print(f"   Config: batch_size={batch_size}, drop_last={drop_last}")
        print(f"      Total exams: {stats['total_exams']}")
        print(f"      Predicted batches: {predicted_len}")
        print(f"      Actual batches: {actual_batches}")
        print(f"      Expected batches: {expected_len}")
        print(f"      Length calculation: {'CORRECT' if len_correct else 'INCORRECT'}")
        
        if not len_correct:
            all_passed = False
    
    # Test get_stats() content
    sampler = PersonBatchSampler(manifest_path, 16, shuffle=True, drop_last=False)
    stats = sampler.get_stats()
    
    required_stats = ['total_persons', 'total_exams', 'avg_exams_per_person', 
                     'batch_size', 'batches_per_epoch', 'shuffle', 'drop_last']
    
    missing_stats = [stat for stat in required_stats if stat not in stats]
    
    print(f"\nStatistics content verification:")
    print(f"   Required stats present: {'YES' if not missing_stats else 'NO'}")
    if missing_stats:
        print(f"   Missing stats: {missing_stats}")
        all_passed = False
    
    print(f"   Sample stats: {dict(list(stats.items())[:3])}")
    
    if all_passed:
        print("PASS: Statistics verification test PASSED")
    else:
        print("FAIL: Statistics verification test FAILED")
    
    return all_passed


def test_infinite_sampler():
    """Test InfinitePersonBatchSampler infinite iteration and statistics."""
    print("\n" + "=" * 60)
    print("TEST 7: Infinite Sampler")
    print("=" * 60)
    
    manifest_path = "data/splits/core/sorted/train_ssl.parquet"
    batch_size = 8
    
    print("Testing InfinitePersonBatchSampler...")
    
    # Test initialization
    infinite_sampler = InfinitePersonBatchSampler(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    print(f"PASS: InfinitePersonBatchSampler initialized successfully")
    print(f"   Total persons: {len(infinite_sampler.persons)}")
    print(f"   Total exams: {infinite_sampler.total_exams}")
    print(f"   Infinite attribute: {getattr(infinite_sampler, 'infinite', False)}")
    
    # Test inheritance
    is_subclass = isinstance(infinite_sampler, PersonBatchSampler)
    print(f"   Inherits from PersonBatchSampler: {'YES' if is_subclass else 'NO'}")
    
    # Test __len__ method (should return very large number)
    sampler_len = len(infinite_sampler)
    very_large = sampler_len > 1e18  # Should be 2**63 - 1
    print(f"   Length is very large: {'YES' if very_large else 'NO'} (len={sampler_len})")
    
    # Test get_stats with infinity
    stats = infinite_sampler.get_stats()
    has_infinite_batches = stats.get('batches_per_epoch') == '∞'
    has_infinite_flag = stats.get('infinite') == True
    print(f"   Stats show infinite batches: {'YES' if has_infinite_batches else 'NO'}")
    print(f"   Stats have infinite flag: {'YES' if has_infinite_flag else 'NO'}")
    
    # Test infinite iteration (with safety limit using islice)
    print("   Testing infinite iteration with safety limit...")
    from itertools import islice
    
    batch_count = 0
    unique_batches = set()
    
    # Take first 100 batches to test infinite behavior
    for batch_indices in islice(infinite_sampler, 100):
        batch_count += 1
        
        # Convert to tuple so it's hashable for set
        batch_tuple = tuple(batch_indices)
        unique_batches.add(batch_tuple)
        
        # Check batch size
        if len(batch_indices) != batch_size:
            print(f"FAIL: Incorrect batch size: {len(batch_indices)} != {batch_size}")
            return False
    
    print(f"   Generated {batch_count} batches safely")
    print(f"   Unique batch patterns: {len(unique_batches)}")
    
    # Test reshuffling behavior (if shuffle=True, should get different patterns)
    if infinite_sampler.shuffle:
        # Take another 10 batches and see if we get different patterns
        additional_batches = set()
        for batch_indices in islice(infinite_sampler, 10):
            additional_batches.add(tuple(batch_indices))
        
        # Should have some overlap but not identical sets (due to reshuffling)
        total_patterns = len(unique_batches.union(additional_batches))
        print(f"   Total unique patterns after more iteration: {total_patterns}")
    
    # Test __repr__ method
    repr_str = repr(infinite_sampler)
    has_infinite_in_repr = "InfinitePersonBatchSampler" in repr_str
    print(f"   Repr contains correct class name: {'YES' if has_infinite_in_repr else 'NO'}")
    
    success = (is_subclass and very_large and has_infinite_batches and 
              has_infinite_flag and batch_count == 100 and has_infinite_in_repr)
    
    if success:
        print("PASS: Infinite sampler test PASSED")
    else:
        print("FAIL: Infinite sampler test FAILED")
    
    return success

def run_all_tests():
    """Run all tests and provide summary."""
    print("PersonBatchSampler Test Suite")
    print("Testing with real medical examination data")
    print("=" * 60)
    
    tests = [
        # ("Basic Functionality", test_basic_functionality),
        # ("Person-Level Grouping", test_person_grouping),
        # ("Shuffle Behavior", test_shuffle_behavior),
        # ("Batch Composition", test_batch_composition),
        # ("Dataset Integration", test_integration_with_dataset),
        # ("Statistics Verification", test_statistics),
        ("Infinite Sampler", test_infinite_sampler)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\nFAIL: {test_name} CRASHED: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:25} {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests PASSED! PersonBatchSampler is ready for training.")
    else:
        print("Some tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
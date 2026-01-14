# tests/models/embedders/test_TabEmbedder.py

import time
import torch
from src.models.dataset import HealthExamDataset
from src.models.collate_fn import collate_exams
from src.models.embedders.TextEmbedder import TextEmbedder
from src.models.embedders.CategoricalEmbedder import CategoricalEmbedder
from src.models.embedders.NumericalEmbedder import NumericalEmbedder
from src.models.embedders.TabEmbedder import TabEmbedder


def initialize_embedders(device="cpu"):
    """Initialize all required embedders."""
    print(f"Initializing embedders on {device}...")
    
    start_time = time.time()
    text_embedder = TextEmbedder(
        pretrained_model_name="alabnii/jmedroberta-base-sentencepiece",
        max_length=256,  # Smaller for testing
        padding="longest",
        truncation=True,
        add_phi_tokens=True,
        device=device
    )
    text_time = time.time() - start_time
    print(f"  TextEmbedder initialized in {text_time:.2f}s")
    
    start_time = time.time()
    code_embedder = CategoricalEmbedder(
        vocab_path="config/embedders/cat_vocab.yaml",
        embedding_dim=768,
        device=device
    )
    cat_time = time.time() - start_time
    print(f"  CategoricalEmbedder initialized in {cat_time:.2f}s")
    
    start_time = time.time()
    num_embedder = NumericalEmbedder(
        d_embedding=768,
        n_bands=16,
        sigma=1.0,
        device=device
    )
    num_time = time.time() - start_time
    print(f"  NumericalEmbedder initialized in {num_time:.2f}s")
    
    print("All embedders initialized!")
    return text_embedder, code_embedder, num_embedder


def initialize_tab_embedder(device="cpu"):
    """Initialize TabEmbedder."""
    print(f"Initializing TabEmbedder on {device}...")
    
    start_time = time.time()
    tab_embedder = TabEmbedder(
        D=768,
        tiny_text_config={
            'd_model': 768,
            'nhead': 4,
            'd_ff': 1536,
            'n_layers': 2,
            'D_out': 768,
            'dropout': 0.1
        },
        device=device
    )
    tab_time = time.time() - start_time
    print(f"  TabEmbedder initialized in {tab_time:.2f}s")
    
    return tab_embedder


def test_tab_embedder_basic(text_embedder, code_embedder, num_embedder, tab_embedder, device="cpu"):
    """Test basic functionality of TabEmbedder."""
    print("\nTesting TabEmbedder with a small batch...")
    
    # Initialize dataset
    dataset = HealthExamDataset(split_name="val_ssl", use_result=True)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Create a small batch manually
    batch_size = 3
    batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    
    # Inspect raw samples
    print("\nInspecting raw samples before collation:")
    for i, sample in enumerate(batch):
        print(f"\nSample {i}:")
        print(f"  Person ID: {sample['person_id']}")
        print(f"  Exam Date: {sample['ExamDate']}")
        print(f"  Number of tests: {len(sample['tests'])}")
        print(f"  Birth Year: {sample.get('birth_year', 'N/A')}")
        print(f"  Gender: {sample.get('gender', 'N/A')}")
        
        # Check result text
        result_text = sample.get('result_text')
        if result_text:
            print(f"  Result text length: {len(result_text)} chars")
            print(f"  Result text preview: {result_text[:50]}...")
        else:
            print("  Result text: None or empty")
    
    # Create batch data using collate_exams
    print("\nCreating batch data using collate_exams...")
    start_time = time.time()
    config = {
        'training': {
            'p_cvr': 0.1,
            'p_mcm': 0.1,
            'p_mlm': 0.1,
            'p_mcc': 0.1,  # Add MCC for testing
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
    
    batch_data = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=config,
        device=device
    )
    collate_time = time.time() - start_time
    print(f"Collate function executed in {collate_time:.4f}s")
    
    # Print batch data summary
    B = batch_data["B"]
    T_max = batch_data["T_max"]
    new_seq_len = 4 + 2 * T_max
    
    print(f"\nBatch data summary:")
    print(f"  B (batch size): {B}")
    print(f"  T_max (max tests): {T_max}")
    print(f"  Expected new sequence length: {new_seq_len}")
    
    # Key tensor shapes from collate
    print(f"\nKey tensor shapes from collate_exams:")
    for key in ["code_ids", "num_values", "text_token_ids", "result_input_ids"]:
        if key in batch_data and isinstance(batch_data[key], torch.Tensor):
            print(f"  {key}: {tuple(batch_data[key].shape)}")
    
    # Test TabEmbedder forward pass
    print("\nRunning TabEmbedder forward pass...")
    start_time = time.time()
    
    final_emb, final_mask, expanded_labels, result_emb = tab_embedder(
        code_embedder=code_embedder,
        num_embedder=num_embedder,
        text_embedder=text_embedder,
        batch_data=batch_data
    )
    
    forward_time = time.time() - start_time
    print(f"TabEmbedder forward pass executed in {forward_time:.4f}s")
    
    # Print output shapes
    print(f"\nTabEmbedder output shapes:")
    print(f"  final_emb: {tuple(final_emb.shape)}")
    print(f"  final_mask: {tuple(final_mask.shape)}")
    print(f"  result_emb: {tuple(result_emb.shape)}")
    
    # Check expanded labels
    print(f"\nExpanded labels shapes:")
    for key, tensor in expanded_labels.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tuple(tensor.shape)}")
    
    # Shape validation
    print(f"\nShape validation:")
    expected_final_shape = (B, new_seq_len, 768)
    expected_mask_shape = (B, new_seq_len)
    
    assert final_emb.shape == expected_final_shape, f"final_emb shape mismatch: got {final_emb.shape}, expected {expected_final_shape}"
    assert final_mask.shape == expected_mask_shape, f"final_mask shape mismatch: got {final_mask.shape}, expected {expected_mask_shape}"
    print(f"  ✓ final_emb shape: {final_emb.shape}")
    print(f"  ✓ final_mask shape: {final_mask.shape}")
    
    # Device validation
    assert final_emb.device.type == device.split(':')[0], f"final_emb on wrong device: {final_emb.device}"
    assert final_mask.device.type == device.split(':')[0], f"final_mask on wrong device: {final_mask.device}"
    print(f"  ✓ Device placement: {final_emb.device}")
    
    # Content validation
    print(f"\nContent validation:")
    
    # Check embeddings are not all zeros
    assert not torch.allclose(final_emb, torch.zeros_like(final_emb)), "final_emb should not be all zeros"
    print(f"  ✓ final_emb is not all zeros")
    
    # Check mask has True values
    assert final_mask.any(), "final_mask should have some True values"
    print(f"  ✓ final_mask has {final_mask.sum().item()} True values out of {final_mask.numel()}")
    
    # Check prefix is always valid (first 4 positions: [CLS] gender age [SEP])
    assert final_mask[:, :4].all(), "Prefix positions should always be valid"
    print(f"  ✓ Prefix mask pattern: {final_mask[0, :4].tolist()}")
    
    return final_emb, final_mask, expanded_labels, result_emb, batch_data


def test_tab_embedder_sequence_structure(final_emb, final_mask, batch_data):
    """Test the sequence structure created by TabEmbedder."""
    print("\nTesting TabEmbedder sequence structure...")
    
    B = batch_data["B"]
    T_max = batch_data["T_max"]
    
    # Check sequence pattern for first sample
    sample_mask = final_mask[0].tolist()
    print(f"First sample mask pattern (first 20 positions): {sample_mask[:20]}")
    
    # Expected pattern: [CLS] gender age [SEP] test1 [SEP] test2 [SEP] ...
    # Positions: 0=CLS, 1=gender, 2=age, 3=SEP, 4=test1, 5=sep1, 6=test2, 7=sep2, ...
    
    print(f"Sequence structure analysis:")
    print(f"  Position 0 ([CLS]): {sample_mask[0]} (should be True)")
    print(f"  Position 1 (gender): {sample_mask[1]} (should be True)")  
    print(f"  Position 2 (age): {sample_mask[2]} (should be True)")
    print(f"  Position 3 ([SEP]): {sample_mask[3]} (should be True)")
    
    # Check test/separator pattern
    for i in range(min(5, T_max)):
        test_pos = 4 + i * 2
        sep_pos = test_pos + 1
        if test_pos < len(sample_mask) and sep_pos < len(sample_mask):
            print(f"  Position {test_pos} (test{i}): {sample_mask[test_pos]}")
            print(f"  Position {sep_pos} (sep{i}): {sample_mask[sep_pos]}")
    
    # Check that separators are valid where tests are valid
    for i in range(T_max):
        test_pos = 4 + i * 2
        sep_pos = test_pos + 1
        if test_pos < final_mask.shape[1] and sep_pos < final_mask.shape[1]:
            test_valid = final_mask[:, test_pos]
            sep_valid = final_mask[:, sep_pos]
            # Separator should be valid where test is valid
            assert torch.equal(test_valid, sep_valid), f"Separator {i} validity mismatch with test {i}"
    
    print(f"  ✓ Test/separator pattern is consistent")


def test_tab_embedder_training_labels(expanded_labels, batch_data):
    """Test expanded training labels from TabEmbedder."""
    print("\nTesting expanded training labels...")
    
    B = batch_data["B"]
    T_max = batch_data["T_max"]
    new_seq_len = 4 + 2 * T_max
    
    # Check expanded labels shapes
    for key in ["mcm_labels", "cvr_mask"]:
        tensor = expanded_labels[key]
        expected_shape = (B, new_seq_len)
        assert tensor.shape == expected_shape, f"{key} shape mismatch: got {tensor.shape}, expected {expected_shape}"
        print(f"  ✓ {key} shape: {tensor.shape}")
    
    # Check that labels are properly positioned
    # Test positions are at 4, 6, 8, 10, ... (4 + i*2)
    test_positions = torch.arange(T_max) * 2 + 4
    
    # Check that non-test positions have -100 (ignored) labels
    print(f"Label positioning analysis:")
    
    # For MCM labels
    mcm_labels = expanded_labels["mcm_labels"]
    non_test_positions = torch.ones(new_seq_len, dtype=torch.bool)
    non_test_positions[test_positions] = False
    
    # All non-test positions should be -100
    non_test_values = mcm_labels[:, non_test_positions]
    assert (non_test_values == -100).all(), "Non-test positions should have -100 labels"
    print(f"  ✓ MCM labels: non-test positions properly ignored")
    
    # Count actual labels vs ignored
    actual_mcm_labels = (mcm_labels != -100).sum().item()
    total_mcm_positions = mcm_labels.numel()
    print(f"  MCM labels: {actual_mcm_labels}/{total_mcm_positions} positions have actual labels")
    
    # Same for CVR mask
    cvr_mask = expanded_labels["cvr_mask"]
    non_test_cvr = cvr_mask[:, non_test_positions]
    assert (non_test_cvr == -100).all(), "Non-test positions should have -100 CVR mask values"
    print(f"  ✓ CVR mask: non-test positions properly ignored")
    actual_cvr_labels = (cvr_mask != -100).sum().item()
    total_cvr_positions = cvr_mask.numel()
    print(f"  CVR mask: {actual_cvr_labels}/{total_cvr_positions} positions have actual labels")

    # Test CVR-specific outputs
    print(f"\nCVR-specific output testing:")
    if "cvr_candidates" in expanded_labels and "cvr_labels" in expanded_labels:
        cvr_candidates = expanded_labels["cvr_candidates"]
        cvr_labels = expanded_labels["cvr_labels"]
        
        print(f"  cvr_candidates shape: {cvr_candidates.shape}")
        print(f"  cvr_labels shape: {cvr_labels.shape}")
        
        # Check that candidates and labels have consistent dimensions
        n_cvr = cvr_candidates.shape[0]
        assert cvr_labels.shape[0] == n_cvr, f"CVR candidates/labels shape mismatch: {n_cvr} vs {cvr_labels.shape[0]}"
        print(f"  ✓ CVR candidates and labels have consistent dimensions: {n_cvr}")
        
        # Check that candidates are non-zero (should contain meaningful embeddings)
        if n_cvr > 0:
            assert not torch.allclose(cvr_candidates, torch.zeros_like(cvr_candidates)), "CVR candidates should not be all zeros"
            print(f"  ✓ CVR candidates contain non-zero embeddings")
        else:
            print(f"  No CVR candidates in this batch (normal with low p_cvr)")
    else:
        print(f"  No CVR candidates found in expanded_labels (normal with low p_cvr)")

    # Test MCC-specific outputs
    print(f"\nMCC-specific output testing:")
    if "mcc_candidates" in expanded_labels and "mcc_labels" in expanded_labels and "mcc_mask" in expanded_labels:
        mcc_candidates = expanded_labels["mcc_candidates"]
        mcc_labels = expanded_labels["mcc_labels"]
        mcc_mask = expanded_labels["mcc_mask"]
        
        print(f"  mcc_candidates shape: {mcc_candidates.shape}")
        print(f"  mcc_labels shape: {mcc_labels.shape}")
        print(f"  mcc_mask shape: {mcc_mask.shape}")
        
        # Check that candidates and labels have consistent dimensions
        n_mcc = mcc_candidates.shape[0] if mcc_candidates.ndim == 3 else 0
        assert mcc_labels.shape[0] == n_mcc, f"MCC candidates/labels shape mismatch: {n_mcc} vs {mcc_labels.shape[0]}"
        print(f"  ✓ MCC candidates and labels have consistent dimensions: {n_mcc}")
        
        # Check K dimension
        if n_mcc > 0:
            K = mcc_candidates.shape[1]
            print(f"  MCC K (candidates per masked cell): {K}")
            assert K >= 2, f"MCC should have at least 2 candidates, got {K}"
            
            # Check that candidates are non-zero (should contain meaningful embeddings)
            assert not torch.allclose(mcc_candidates, torch.zeros_like(mcc_candidates)), "MCC candidates should not be all zeros"
            print(f"  ✓ MCC candidates contain non-zero embeddings")
            
            # Check label range [0, K-1]
            assert mcc_labels.min() >= 0 and mcc_labels.max() < K, f"MCC labels should be in range [0, {K-1}], got [{mcc_labels.min()}, {mcc_labels.max()}]"
            print(f"  ✓ MCC labels in valid range [0, {K-1}]")
            
            # Check MCC mask expansion
            mcc_mask_expanded = expanded_labels["mcc_mask"]
            non_test_mcc = mcc_mask_expanded[:, non_test_positions]
            assert (non_test_mcc == False).all(), "Non-test positions should have False MCC mask values"
            print(f"  ✓ MCC mask: non-test positions properly set to False")
            actual_mcc_positions = mcc_mask_expanded.sum().item()
            print(f"  MCC mask: {actual_mcc_positions} positions marked for MCC")
        else:
            print(f"  No MCC candidates in this batch (normal with low p_mcc)")
    else:
        print(f"  No MCC data found in expanded_labels (normal with low p_mcc)")


def test_tab_embedder_demographics(final_emb, batch_data):
    """Test demographic embeddings in TabEmbedder."""
    print("\nTesting demographic embeddings...")
    
    # Extract demographic information from batch_data
    exam_ages = batch_data["exam_ages"]
    exam_genders = batch_data["exam_genders"]
    
    print(f"Demographic information:")
    print(f"  exam_ages: {exam_ages.tolist()}")
    print(f"  exam_genders: {exam_genders.tolist()}")
    
    # Check that demographics are embedded at positions 1 (gender) and 2 (age)
    # We can't directly verify the embedding values, but we can check they're not zero
    gender_emb = final_emb[:, 1, :]  # [B, D]
    age_emb = final_emb[:, 2, :]     # [B, D]
    
    assert not torch.allclose(gender_emb, torch.zeros_like(gender_emb)), "Gender embeddings should not be all zeros"
    assert not torch.allclose(age_emb, torch.zeros_like(age_emb)), "Age embeddings should not be all zeros"
    
    print(f"  ✓ Gender embeddings non-zero: {gender_emb.abs().mean().item():.4f} avg magnitude")
    print(f"  ✓ Age embeddings non-zero: {age_emb.abs().mean().item():.4f} avg magnitude")
    
    # Check that different age/gender values produce different embeddings
    if len(set(exam_ages.tolist())) > 1:
        # Check that different ages produce different embeddings
        age0_emb = age_emb[exam_ages == exam_ages[0]]
        other_ages = exam_ages != exam_ages[0]
        if other_ages.any():
            other_age_emb = age_emb[other_ages][:1]  # Take first different age
            assert not torch.allclose(age0_emb[0], other_age_emb[0]), "Different ages should produce different embeddings"
            print(f"  ✓ Different ages produce different embeddings")
    
    if len(set(exam_genders.tolist())) > 1:
        # Check that different genders produce different embeddings
        gender0_emb = gender_emb[exam_genders == exam_genders[0]]
        other_genders = exam_genders != exam_genders[0]
        if other_genders.any():
            other_gender_emb = gender_emb[other_genders][:1]
            assert not torch.allclose(gender0_emb[0], other_gender_emb[0]), "Different genders should produce different embeddings"
            print(f"  ✓ Different genders produce different embeddings")


def test_tab_embedder_with_text_values(text_embedder, code_embedder, num_embedder, tab_embedder, device="cpu"):
    """Test TabEmbedder with text test values."""
    print("\nTesting TabEmbedder with text test values...")
    
    # Try to find examples with text test values
    dataset = HealthExamDataset(split_name="train_ssl")
    
    # Search for examples with text test values
    found_batch = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        tests = sample["tests"]
        
        # Check if any test has text values
        has_text = any(test.get("type") == "ST" and test.get("value_text") for test in tests)
        if has_text:
            found_batch.append(sample)
            if len(found_batch) >= 2:
                break
    
    if not found_batch:
        print("  No examples with text test values found, using regular examples")
        found_batch = [dataset[i] for i in range(2)]
    else:
        print(f"  Found {len(found_batch)} examples with text test values")
    
    # Create batch data
    config = {
        'training': {
            'p_cvr': 0.0,
            'p_mcm': 0.0,
            'p_mlm': 0.0,
            'p_mcc': 0.0,  # Disable training tasks for this test
            'mcc': {
                'enabled': False,
                'K': 5
            }
        }
    }
    
    batch_data = collate_exams(
        batch=found_batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=config,
        device=device
    )
    
    # Check text optimization results
    if batch_data["text_locations"].numel() > 0:
        print(f"  Text locations shape: {batch_data['text_locations'].shape}")
        print(f"  Number of text sequences: {len(batch_data['text_locations'])}")
        print(f"  Text token IDs shape: {batch_data['text_token_ids'].shape}")
    else:
        print("  No text test values in this batch")
    
    # Run TabEmbedder
    final_emb, final_mask, expanded_labels, result_emb = tab_embedder(
        code_embedder=code_embedder,
        num_embedder=num_embedder,
        text_embedder=text_embedder,
        batch_data=batch_data
    )
    
    print(f"  ✓ TabEmbedder processed batch with text values successfully")
    print(f"  final_emb shape: {final_emb.shape}")


def test_tab_embedder_config():
    """Test TabEmbedder configuration."""
    print("\nTesting TabEmbedder configuration...")
    
    # Create TabEmbedder with custom config
    custom_config = {
        'D': 512,
        'tiny_text_config': {
            'd_model': 768,
            'nhead': 8,
            'd_ff': 2048,
            'n_layers': 3,
            'D_out': 512,
            'dropout': 0.2
        },
        'device': 'cpu'
    }
    
    tab_embedder = TabEmbedder(**custom_config)
    
    # Get config back
    saved_config = tab_embedder.get_config()
    
    print(f"  Original config D: {custom_config['D']}")
    print(f"  Saved config D: {saved_config['D']}")
    assert saved_config['D'] == custom_config['D']
    
    print(f"  ✓ Configuration save/load works correctly")


def test_tab_embedder_mcc_high_probability(text_embedder, code_embedder, num_embedder, tab_embedder, device="cpu"):
    """Test TabEmbedder MCC task with high probability."""
    print("\nTesting TabEmbedder MCC task with high probability...")
    
    # Initialize dataset
    dataset = HealthExamDataset(split_name="val_ssl", use_result=True)
    
    # Create batch with high MCC probability to ensure MCC candidates are generated
    batch = [dataset[i] for i in range(min(3, len(dataset)))]
    
    config = {
        'training': {
            'p_cvr': 0.0,  # Disable other tasks
            'p_mcm': 0.0,
            'p_mlm': 0.0,
            'p_mcc': 0.7,  # High probability for MCC
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
    
    # Create batch data
    batch_data = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=config,
        device=device
    )
    
    print(f"Batch info: B={batch_data['B']}, T_max={batch_data['T_max']}")
    
    # Check raw MCC data from collate_exams
    if "opts_raw" in batch_data and "mcc_labels" in batch_data:
        opts_raw = batch_data["opts_raw"]
        mcc_labels = batch_data["mcc_labels"]
        mcc_mask = batch_data["mcc_mask"]
        
        print(f"Raw MCC data from collate_exams:")
        print(f"  opts_raw shape: {opts_raw.shape}")
        print(f"  mcc_labels shape: {mcc_labels.shape}")
        print(f"  mcc_mask shape: {mcc_mask.shape}")
        print(f"  Number of masked cells: {mcc_mask.sum().item()}")
        
        if opts_raw.numel() > 0:
            n_mcc, K = opts_raw.shape
            print(f"  Generated {n_mcc} MCC candidate sets with K={K}")
            print(f"  Sample candidates: {opts_raw[0].tolist()}")
            print(f"  Sample label: {mcc_labels[0].item()}")
    
    # Run TabEmbedder
    final_emb, final_mask, expanded_labels, result_emb = tab_embedder(
        code_embedder=code_embedder,
        num_embedder=num_embedder,
        text_embedder=text_embedder,
        batch_data=batch_data
    )
    
    # Test MCC candidate processing
    if "mcc_candidates" in expanded_labels:
        mcc_candidates = expanded_labels["mcc_candidates"]
        print(f"MCC candidates after TabEmbedder processing:")
        print(f"  mcc_candidates shape: {mcc_candidates.shape}")
        
        if mcc_candidates.numel() > 0:
            # Check that candidates have been processed through embeddings
            assert not torch.allclose(mcc_candidates, torch.zeros_like(mcc_candidates)), "MCC candidates should not be all zeros"
            print(f"  ✓ MCC candidates processed through embeddings")
            
            # Check embedding dimension
            assert mcc_candidates.shape[-1] == 768, f"MCC candidates should have embedding dim 768, got {mcc_candidates.shape[-1]}"
            print(f"  ✓ MCC candidates have correct embedding dimension")
            
            # Check that numerical masking was applied (-999.0 replacement)
            num_values = batch_data["num_values"]
            masked_positions = (num_values == -999.0)
            print(f"  Number of positions with -999.0 masking: {masked_positions.sum().item()}")
            
            if masked_positions.any():
                print(f"  ✓ Numerical masking (-999.0) applied successfully")
    
    print(f"  ✓ MCC task testing completed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TABEMBEDDER")
    print("=" * 60)
    
    # Use CPU by default for consistent testing
    device = "cpu"
    
    # Initialize all components
    text_embedder, code_embedder, num_embedder = initialize_embedders(device)
    tab_embedder = initialize_tab_embedder(device)
    
    # Run tests
    final_emb, final_mask, expanded_labels, result_emb, batch_data = test_tab_embedder_basic(
        text_embedder, code_embedder, num_embedder, tab_embedder, device
    )
    
    test_tab_embedder_sequence_structure(final_emb, final_mask, batch_data)
    test_tab_embedder_training_labels(expanded_labels, batch_data)
    test_tab_embedder_demographics(final_emb, batch_data)
    test_tab_embedder_with_text_values(text_embedder, code_embedder, num_embedder, tab_embedder, device)
    test_tab_embedder_config()
    test_tab_embedder_mcc_high_probability(text_embedder, code_embedder, num_embedder, tab_embedder, device)
    
    print("\n All TabEmbedder tests completed successfully!")
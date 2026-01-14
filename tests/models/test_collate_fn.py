# tests/models/test_collate_fn.py
import datetime as dt
import time
import random
import torch
from src.models.dataset import HealthExamDataset
from src.models.collate_fn import collate_exams
from src.models.embedders.TextEmbedder import TextEmbedder
from src.models.embedders.CategoricalEmbedder import CategoricalEmbedder
from torch.utils.data import DataLoader, Subset


def initialize_embedders(device="cpu"):
    """Initialize all required embedders."""
    print(f"Initializing embedders on {device}...")
    
    start_time = time.time()
    # Use max_length and other parameters during initialization
    text_embedder = TextEmbedder(
        pretrained_model_name="alabnii/jmedroberta-base-sentencepiece",
        max_length=512,
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
    
    print("All embedders initialized!")
    return text_embedder, code_embedder


def manual_collate_fn_basic(text_embedder, code_embedder, device="cpu"):
    """Test basic functionality of the collate_fn."""
    print("\nTesting collate_fn with a small batch...")
    
    # Initialize dataset
    dataset = HealthExamDataset(split_name="val_ssl")
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Create a small batch manually
    batch_size = 3
    batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    
    # ADDED: Inspect the samples directly
    print("\nInspecting raw samples before collation:")
    for i, sample in enumerate(batch):
        print(f"\nSample {i}:")
        print(f"  Person ID: {sample['person_id']}")
        print(f"  Exam Date: {sample['ExamDate']}")
        print(f"  Birth Year: {sample.get('birth_year', 'N/A')}")
        print(f"  Gender: {sample.get('gender', 'N/A')}")
        print(f"  Number of tests: {len(sample['tests'])}")
        
        # Check result text
        if 'result_text' in sample:
            result_text = sample['result_text']
            print(f"  Result text exists: {bool(result_text)}")
            # Only try to get length if result_text is not None
            if result_text:
                print(f"  Result text length: {len(result_text)} chars")
                print(f"  Result text preview: {result_text[:50]}...")
            else:
                print("  Result text is None (empty)")
        else:
            print("  Result text: [NOT PRESENT]")

    # Test collate function with default config
    default_config = {
        'training': {
            'p_cvr': 0.20,
            'p_mcm': 0.15,
            'p_mlm': 0.15,
            'p_mcc': 0.20,  # Add default MCC probability
            'use_held_out_codes': True,
            'held_out_codes_path': "config/splitting/held_out_codes.yaml",
            'mcc': {
                'enabled': True,  # Now enabled by default
                'K': 5,
                'noise': {
                    'gaussian_scales': [0.05, 0.20],
                    'mix_probs': [0.5, 0.35, 0.15],
                    'large_min_norm_dist': 0.3
                }
            }
        }
    }
    
    start_time = time.time()
    outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=default_config,
        device=device
    )
    collate_time = time.time() - start_time
    
    # Print output information
    print(f"\nCollate function executed in {collate_time:.4f}s")
    print("Output keys:", list(outputs.keys()))
    
    # Print shapes of key tensors
    print("\nTensor shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")
        elif isinstance(value, list):
            print(f"  {key}: List with {len(value)} items")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
    
    # NEW: Check batch dimensions metadata
    print(f"\nBatch dimensions metadata:")
    print(f"  B (batch size): {outputs.get('B', 'NOT FOUND')}")
    print(f"  T_max (max tests): {outputs.get('T_max', 'NOT FOUND')}")
    print(f"  Expected batch size: {len(batch)}")
    print(f"  Actual max tests in batch: {max(len(sample['tests']) for sample in batch)}")
    
    # Verify B and T_max are correct
    assert outputs['B'] == len(batch), f"B mismatch: got {outputs['B']}, expected {len(batch)}"
    expected_t_max = max(len(sample['tests']) for sample in batch)
    assert outputs['T_max'] == expected_t_max, f"T_max mismatch: got {outputs['T_max']}, expected {expected_t_max}"
    print("B and T_max metadata verified!")
    
    # NEW: Check exam-level demographics
    print(f"\nExam-level demographics:")
    print(f"  exam_ages: {outputs['exam_ages'].tolist()}")
    print(f"  exam_genders: {outputs['exam_genders'].tolist()}")
    print(f"  Expected shapes: [{len(batch)}] for both")
    
    # Verify demographic tensor shapes
    assert outputs['exam_ages'].shape == (len(batch),), f"exam_ages shape mismatch"
    assert outputs['exam_genders'].shape == (len(batch),), f"exam_genders shape mismatch"
    print("Exam-level demographics verified!")
    
    # NEW: Check text optimization
    if "text_locations" in outputs:
        print(f"\nText optimization results:")
        print(f"  text_locations shape: {outputs['text_locations'].shape}")
        print(f"  Number of actual text sequences: {len(outputs['text_locations'])}")
        if len(outputs['text_locations']) > 0:
            print(f"  Sample text locations: {outputs['text_locations'][:3].tolist()}")
    else:
        print(f"\nNo text_locations found (no text test values in batch)")
    
    # NEW: Verify text optimization
    total_test_positions = outputs['mask_text'].sum().item()
    actual_text_sequences = len(outputs.get('text_locations', []))
    print(f"Text optimization: {actual_text_sequences} sequences vs {total_test_positions} text positions")

    # Check CVR-specific outputs
    if "cvr_mask" in outputs:
        print(f"\nCVR outputs:")
        print(f"  cvr_mask: {outputs['cvr_mask'].shape}")
        print(f"  cvr_labels: {outputs['cvr_labels'].shape if 'cvr_labels' in outputs else 'Not found'}")
        print(f"  cvr_true_ids: {len(outputs.get('cvr_true_ids', []))} candidates")
        print(f"  cvr_true_attention_masks: {len(outputs.get('cvr_true_attention_masks', []))} masks")

    # Check MCC-specific outputs (enabled with default probability)
    if "mcc_mask" in outputs and "opts_raw" in outputs and "mcc_labels" in outputs:
        print(f"\nMCC outputs (enabled, default probability):")
        print(f"  mcc_mask: {outputs['mcc_mask'].shape}")
        print(f"  opts_raw: {outputs['opts_raw'].shape}")
        print(f"  mcc_labels: {outputs['mcc_labels'].shape}")
        
        # Count MCC selections
        mcc_selected = outputs['mcc_mask'].sum().item()
        mcc_total = outputs['mask_num'].sum().item()
        n_mcc_candidates = outputs['opts_raw'].shape[0]
        
        if mcc_total > 0:
            print(f"  MCC selection: {mcc_selected}/{mcc_total} numerical positions masked ({mcc_selected/mcc_total:.1%})")
            print(f"  MCC candidates generated: {n_mcc_candidates}")
            if n_mcc_candidates > 0:
                K = outputs['opts_raw'].shape[1]
                print(f"  MCC candidate matrix: [{n_mcc_candidates}, {K}] (n_mcc × K)")
                print(f"  Sample candidates: {outputs['opts_raw'][0].tolist() if n_mcc_candidates > 0 else 'None'}")
        else:
            print("  MCC: No numerical test values found in this batch")
    else:
        print("\nMCC: MCC outputs not found in batch")
    
    # ADDED: Inspect tokenized result text
    print("\nInspecting tokenized result text:")
    print(f"  result_input_ids shape: {outputs['result_input_ids'].shape}")
    print(f"  result_attention_mask shape: {outputs['result_attention_mask'].shape}")
    
    # Try to decode the tokens to see what's there
    for i in range(len(outputs['result_input_ids'])):
        tokens = outputs['result_input_ids'][i].tolist()
        mask = outputs['result_attention_mask'][i].tolist()
        print(f"\n  Sample {i} tokens: {tokens}")
        print(f"  Sample {i} attention mask: {mask}")
        print(f"  Non-padding token count: {sum(mask)}")
        
        # Try to decode the tokens
        try:
            decoded = text_embedder.decode(tokens)
            print(f"  Decoded: '{decoded}'")
        except Exception as e:
            print(f"  Decoding failed: {str(e)}")
    
    # Check segment lengths
    print(f"\nSegment lengths: {outputs['segment_lengths']}")
    
    return outputs


# ---------------------------------------------------------------------------
# Unit tests for missing-result handling (no external data dependency)
# ---------------------------------------------------------------------------


class _UnitDummyCodeEmbedder:
    def __init__(self):
        self._token_to_id = {'<PAD>': 0, '<UNK>': 1}
        self._next_id = 2

    def _get_id(self, token):
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._next_id += 1
        return self._token_to_id[token]

    def map(self, tokens, device):
        ids = [self._get_id(token) for token in tokens]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def get_token_id(self, token):
        return self._get_id(token)


class _UnitDummyTokenizer:
    pad_token_id = 0
    mask_token_id = 103

    def __len__(self):
        return 30522

    def get_special_tokens_mask(self, seq, already_has_special_tokens=True):
        return [1 if token in (101, 102) else 0 for token in seq]


class _UnitDummyTextEmbedder:
    def __init__(self):
        self.tokenizer = _UnitDummyTokenizer()

    def tokenize(self, texts):
        ids = []
        masks = []
        for text in texts:
            seq = [101]
            if text:
                seq.extend([200 + (idx % 5) for idx, _ in enumerate(text.split())])
            seq.append(102)
            ids.append(torch.tensor(seq, dtype=torch.int32))
            masks.append(torch.ones(len(seq), dtype=torch.int32))

        if ids:
            input_ids = torch.stack(ids)
            attention_mask = torch.stack(masks)
        else:
            input_ids = torch.empty(0, 0, dtype=torch.int32)
            attention_mask = torch.empty(0, 0, dtype=torch.int32)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _unit_make_sample(person_id: str, result_key: str, result_value):
    sample = {
        "person_id": person_id,
        "ExamDate": dt.date(2020, 1, 1),
        "birth_year": 1980,
        "gender": "M",
        "tests": [
            {"code": "C1", "type": "PQ", "value_num": 1.0},
        ],
    }
    sample[result_key] = result_value
    return sample


_UNIT_CONFIG = {
    "training": {
        "p_cvr": 0.0,
        "p_mcm": 0.0,
        "p_mlm": 0.0,
        "p_mcc": 0.0,
        "use_held_out_codes": False,
        "mcc": {
            "enabled": False,
            "K": 5,
        },
    }
}


def test_collate_raw_missing_result_masks_out():
    batch = [
        _unit_make_sample("p1", "result_text", "patient reports improvement"),
        _unit_make_sample("p2", "result_text", ""),
    ]

    outputs = collate_exams(
        batch=batch,
        code_embedder=_UnitDummyCodeEmbedder(),
        text_embedder=_UnitDummyTextEmbedder(),
        config=_UNIT_CONFIG,
        device="cpu",
    )

    assert outputs["result_attention_mask"].shape[0] == 2
    assert outputs["result_attention_mask"][0].sum() > 0
    assert outputs["result_attention_mask"][1].sum() == 0
    assert torch.all(outputs["result_input_ids"][1] == 0)


def test_collate_pretokenized_missing_result_masks_out():
    sample_present = _unit_make_sample("p1", "result_input_ids", [7, 8, 9])
    sample_present["result_attention_mask"] = [1, 1, 1]
    sample_missing = _unit_make_sample("p2", "result_input_ids", [])
    sample_missing["result_attention_mask"] = []

    batch = [sample_present, sample_missing]

    outputs = collate_exams(
        batch=batch,
        code_embedder=_UnitDummyCodeEmbedder(),
        text_embedder=_UnitDummyTextEmbedder(),
        config=_UNIT_CONFIG,
        device="cpu",
    )

    assert outputs["result_attention_mask"][0].sum() == 3
    assert outputs["result_attention_mask"][1].sum() == 0
    assert torch.all(outputs["result_input_ids"][1] == 0)


def manual_collate_fn_masking(text_embedder, code_embedder, device="cpu"):
    """Test masking behavior in collate_fn."""
    print("\nTesting masking behavior with high probabilities...")
    
    # Initialize dataset
    dataset = HealthExamDataset(split_name="val_ssl")
    
    # Create a small batch
    batch_size = 3
    batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    
    # Test with high masking probabilities including MCC
    high_masking_config = {
        'training': {
            'p_cvr': 0.8,
            'p_mcm': 0.8,
            'p_mlm': 0.5,
            'p_mcc': 0.7,  # Enable MCC with high probability
            'use_held_out_codes': True,
            'held_out_codes_path': "config/splitting/held_out_codes.yaml",
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
    
    outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=high_masking_config,
        device=device
    )
    
    # Check MCM masking
    num_masked = (outputs["mcm_inputs"] != outputs["cat_value_ids"]).sum().item()
    num_labeled = (outputs["mcm_labels"] != -100).sum().item()
    print(f"MCM masking: {num_masked} tokens masked, {num_labeled} labels generated")
    
    # Check CVR selection
    if "cvr_mask" in outputs and "cvr_labels" in outputs:
        cvr_selected = (outputs["cvr_mask"] == 1).sum().item()
        cvr_total = (outputs["cvr_mask"] != -100).sum().item()
        n_cvr_candidates = len(outputs.get("cvr_true_ids", []))
        if cvr_total > 0:
            print(f"CVR selection: {cvr_selected}/{cvr_total} text test positions selected ({cvr_selected/cvr_total:.1%})")
            print(f"CVR candidates generated: {n_cvr_candidates}")
            if n_cvr_candidates > 0:
                print(f"CVR labels shape: {outputs['cvr_labels'].shape}")
        else:
            print("CVR: No text test values found in this batch")
    else:
        print("CVR: CVR outputs not found in batch")

    # Check MCC masking (enabled with high probability)
    if "mcc_mask" in outputs and "opts_raw" in outputs and "mcc_labels" in outputs:
        print(f"\nMCC outputs (high probability):")
        print(f"  mcc_mask: {outputs['mcc_mask'].shape}")
        print(f"  opts_raw: {outputs['opts_raw'].shape}")
        print(f"  mcc_labels: {outputs['mcc_labels'].shape}")
        
        # Count MCC selections
        mcc_selected = outputs['mcc_mask'].sum().item()
        mcc_total = outputs['mask_num'].sum().item()
        n_mcc_candidates = outputs['opts_raw'].shape[0]
        
        if mcc_total > 0:
            print(f"  MCC selection: {mcc_selected}/{mcc_total} numerical positions masked ({mcc_selected/mcc_total:.1%})")
            print(f"  MCC candidates generated: {n_mcc_candidates}")
            if n_mcc_candidates > 0:
                K = outputs['opts_raw'].shape[1]
                print(f"  MCC candidate matrix: [{n_mcc_candidates}, {K}] (n_mcc × K)")
                print(f"  MCC labels: {outputs['mcc_labels'].tolist()}")
                # Check that all labels are valid (0 <= label < K)
                valid_labels = all(0 <= label < K for label in outputs['mcc_labels'].tolist())
                print(f"  All MCC labels valid: {valid_labels}")
        else:
            print("  MCC: No numerical test values found in this batch")
    else:
        print("\nMCC: MCC outputs not found in batch")
    
    # Check MLM masking
    mlm_masked = (outputs["result_mlm_labels"] != -100).sum().item()
    mlm_total = outputs["result_attention_mask"].sum().item()
    if mlm_total > 0:
        print(f"MLM masking: {mlm_masked}/{mlm_total} result text tokens masked ({mlm_masked/mlm_total:.1%})")
    else:
        print("MLM: No result texts found in this batch")


def manual_dataloader_integration(text_embedder, code_embedder, device="cpu"):
    """Test integration with DataLoader."""
    print("\nTesting integration with DataLoader...")
    
    # Initialize dataset - use val_ssl which should be smaller
    dataset = HealthExamDataset(split_name="val_ssl")
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # For test purposes, take just a small subset
    subset_size = min(20, len(dataset))
    dataset_subset = Subset(dataset, range(subset_size))
    print(f"Using subset of {len(dataset_subset)} examples")
    
    # Create a custom collate function that uses our embedders
    integration_config = {
        'training': {
            'p_cvr': 0.20,
            'p_mcm': 0.15,
            'p_mlm': 0.15,
            'use_held_out_codes': True,
            'held_out_codes_path': "config/splitting/held_out_codes.yaml",
            'mcc': {
                'enabled': False,
                'K': 5
            }
        }
    }
    
    def collate_fn(batch):
        return collate_exams(
            batch=batch,
            code_embedder=code_embedder,
            text_embedder=text_embedder,
            config=integration_config,
            device=device
        )
    
    # Create DataLoader
    batch_size = 4
    loader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for easier debugging
    )
    
    # Time the data loading
    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= 1:  # Only process a couple of batches
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Batch size: {batch.get('B', 'unknown')}")
        print(f"  T_max: {batch.get('T_max', 'unknown')}")
        
        # Print a few key tensor shapes
        print("  Selected tensor shapes:")
        for key in ["code_ids", "num_values", "result_input_ids"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                print(f"    {key}: {tuple(batch[key].shape)}")
        
        # Check segment lengths
        print(f"  Segment lengths: {batch['segment_lengths']}")
        
        # Print device information for a tensor
        if "code_ids" in batch and isinstance(batch["code_ids"], torch.Tensor):
            print(f"  Device: {batch['code_ids'].device}")
    
    total_time = time.time() - start_time
    print(f"\nProcessed in {total_time:.4f}s")


def manual_device_handling(text_embedder, code_embedder):
    """Test device handling (CPU vs CUDA if available)."""
    if not torch.cuda.is_available():
        print("\nSkipping CUDA device test (not available)")
        return
    
    print("\nTesting device handling (CPU vs CUDA)...")
    
    # Initialize dataset
    dataset = HealthExamDataset(split_name="val_ssl")
    batch = [dataset[i] for i in range(3)]
    
    device_test_config = {
        'training': {
            'p_cvr': 0.20,
            'p_mcm': 0.15,
            'p_mlm': 0.15,
            'use_held_out_codes': True,
            'held_out_codes_path': "config/splitting/held_out_codes.yaml",
            'mcc': {
                'enabled': False,
                'K': 5
            }
        }
    }
    
    # Test on CPU
    start_time = time.time()
    cpu_outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=device_test_config,
        device="cpu"
    )
    
    # Test on CUDA
    start_time = time.time()
    cuda_outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=device_test_config,
        device="cuda:0"
    )
    cuda_time = time.time() - start_time
    print(f"CUDA processing time: {cuda_time:.4f}s")
    
    # Check that all tensors are on the expected device
    cpu_device_check = all(
        tensor.device.type == "cpu" 
        for tensor in cpu_outputs.values() 
        if isinstance(tensor, torch.Tensor)
    )
    
    cuda_device_check = all(
        tensor.device.type == "cuda" 
        for tensor in cuda_outputs.values() 
        if isinstance(tensor, torch.Tensor)
    )
    
    print(f"All CPU tensors on CPU: {cpu_device_check}")
    print(f"All CUDA tensors on CUDA: {cuda_device_check}")
    print(f"Speed improvement with CUDA: {cpu_time/cuda_time:.2f}x")


def manual_result_text_handling(text_embedder, code_embedder, device="cpu"):
    """Test handling of result text in collate_fn."""
    print("\nTesting result text handling in collate_fn...")
    
    # Initialize dataset with result_text
    dataset = HealthExamDataset(split_name="train_ssl", use_result=True)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Find examples with non-empty result_text
    found_examples = []
    found_indices = []
    max_tries = 50
    num_needed = 4
    
    print(f"Searching for {num_needed} examples with non-empty result_text...")
    for i in range(max_tries):
        idx = random.randint(0, len(dataset) - 1)
        example = dataset[idx]
        
        if 'result_text' in example and example['result_text'] is not None and example['result_text']:
            found_examples.append(example)
            found_indices.append(idx)
            print(f"  Found example at index {idx} with text length: {len(example['result_text'])}")
            
            if len(found_examples) >= num_needed:
                break
    
    if not found_examples:
        print("No examples with result_text found, falling back to regular examples")
        found_examples = [dataset[i] for i in range(num_needed)]
    
    # Create batch with result text
    batch = found_examples
    print(f"Created batch with {len(batch)} examples")
    
    # Print summary of result texts
    print("\nResult text summary:")
    for i, example in enumerate(batch):
        text = example.get('result_text', '')
        if text:
            print(f"  Example {i}: {len(text)} chars: {text[:50]}...")
        else:
            print(f"  Example {i}: Empty or None")
    
    # Process batch with high MLM probability
    mlm_config = {
        'training': {
            'p_cvr': 0.20,
            'p_mcm': 0.15,
            'p_mlm': 0.5,  # High MLM probability for testing
            'use_held_out_codes': True,
            'held_out_codes_path': "config/splitting/held_out_codes.yaml",
            'mcc': {
                'enabled': False,
                'K': 5
            }
        }
    }
    
    outputs = collate_exams(
        batch=batch,
        code_embedder=code_embedder,
        text_embedder=text_embedder,
        config=mlm_config,
        device=device
    )
    
    # Examine result text tokenization
    if "result_input_ids" in outputs and "result_attention_mask" in outputs:
        input_ids = outputs["result_input_ids"]
        attn_mask = outputs["result_attention_mask"]
        mlm_labels = outputs["result_mlm_labels"]
        
        print("\nResult text tokenization:")
        print(f"  Shape: {input_ids.shape}")
        
        # Count non-padding tokens
        token_counts = attn_mask.sum(dim=1).tolist()
        print(f"  Non-padding token counts: {token_counts}")
        
        # Count masked tokens (for MLM)
        masked_counts = (mlm_labels != -100).sum(dim=1).tolist()
        print(f"  MLM masked token counts: {masked_counts}")
        
        # Calculate masking percentage
        for i in range(len(token_counts)):
            if token_counts[i] > 0:
                mask_pct = masked_counts[i] / token_counts[i] * 100
                print(f"  Example {i}: {masked_counts[i]}/{token_counts[i]} tokens masked ({mask_pct:.1f}%)")
        
        # Decode and print
        print("\n  Decoded text samples:")
        for i in range(min(2, len(input_ids))):
            # Get only valid tokens
            valid_tokens = input_ids[i, :token_counts[i]].tolist()
            try:
                decoded = text_embedder.decode(valid_tokens)
                print(f"    Example {i}: '{decoded[:100]}...'")
                
                # Show masked positions
                if masked_counts[i] > 0:
                    mask_positions = (mlm_labels[i] != -100).nonzero(as_tuple=True)[0].tolist()
                    print(f"    Masked at positions: {mask_positions[:10]}{'...' if len(mask_positions) > 10 else ''}")
            except Exception as e:
                print(f"    Error decoding: {str(e)}")
    else:
        print("  result_input_ids not found in outputs")
    
    return outputs


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING COLLATE FUNCTION")
    print("=" * 50)
    
    # Use CPU by default for consistent testing
    device = "cpu"
    
    # Initialize embedders
    text_embedder, code_embedder = initialize_embedders(device)
    
    # Run tests
    manual_collate_fn_basic(text_embedder, code_embedder, device)
    manual_collate_fn_masking(text_embedder, code_embedder, device)
    manual_result_text_handling(text_embedder, code_embedder, device)
    manual_dataloader_integration(text_embedder, code_embedder, device)
    # test_device_handling(text_embedder, code_embedder)
    
    print("\nAll tests completed!")

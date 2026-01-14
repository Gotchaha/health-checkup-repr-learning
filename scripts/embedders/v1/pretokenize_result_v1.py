"""Pre-tokenize result text (v1 presence-aware baseline).

Runs from the project root and writes a v1-style cache with 
presence-aware empty handling (missing/blank texts -> empty sequences).
"""

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.embedders.TextEmbedder import TextEmbedder  # noqa: E402


def setup_logging(log_dir: Path, label: str) -> None:
    """Configure logging for v1 scripts (console + file in outputs/cache/v1)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{label}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    logging.info("Logging initialized -> %s", log_path)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 for provenance."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_output_filename(model_name: str, max_length: int, truncation: bool, add_phi: bool) -> str:
    model_tag = model_name.split('/')[-1].replace('-', '')
    trunc_tag = "_trunc" if truncation else "_notrunc"
    phi_tag = "_phi" if add_phi else "_nophi"
    return f"result_{model_tag}_tok{max_length}{trunc_tag}{phi_tag}.parquet"


def create_metadata(
    tokenizer_args: Dict[str, Any],
    source_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """Build metadata blob describing tokenizer + I/O."""
    return {
        "creation_timestamp": datetime.now().isoformat(),
        "tokenizer": {
            **tokenizer_args,
            "presence_aware_empty": True,
        },
        "source_data": {
            "file_path": str(source_path),
            "file_hash": compute_file_hash(source_path),
            "num_samples": None,
        },
        "cached_data": {
            "file_path": str(output_path),
            "file_hash": None,
            "num_samples": None,
        },
    }


def tokenize_result_data(table_path: Path, text_embedder: TextEmbedder) -> pa.Table:
    """Tokenize result texts row-by-row with presence-aware empty handling."""
    logging.info("Loading source result data from %s", table_path)
    table = pq.read_table(table_path, columns=["AnonymousID", "McExamDt", "ResultText"])
    num_rows = len(table)
    logging.info("Tokenizing %d rows", num_rows)

    anon_ids = table['AnonymousID'].to_pylist()
    exam_dates = table['McExamDt'].to_pylist()
    result_texts = table['ResultText'].to_pylist()

    tokenized = {
        'AnonymousID': [],
        'McExamDt': [],
        'input_ids': [],
        'attention_mask': [],
    }

    for idx, (anon_id, exam_date, text) in enumerate(zip(anon_ids, exam_dates, result_texts)):
        if idx % 10000 == 0:
            pct = (idx / num_rows * 100) if num_rows else 100.0
            logging.info("Processed %d/%d rows (%.1f%%)", idx, num_rows, pct)

        raw_text = text or ""
        if not raw_text.strip():
            ids = []
            mask = []
        else:
            encoded = text_embedder.tokenize(raw_text)
            ids = [int(x) for x in encoded['input_ids'][0].cpu().tolist()]
            mask = [int(x) for x in encoded['attention_mask'][0].cpu().tolist()]

        tokenized['AnonymousID'].append(anon_id)
        tokenized['McExamDt'].append(exam_date)
        tokenized['input_ids'].append(ids)
        tokenized['attention_mask'].append(mask)

    schema = pa.schema([
        ('AnonymousID', table.schema.field('AnonymousID').type),
        ('McExamDt', table.schema.field('McExamDt').type),
        ('input_ids', pa.list_(pa.int32())),
        ('attention_mask', pa.list_(pa.int8())),
    ])

    result = pa.table(tokenized, schema=schema)
    logging.info("Tokenization complete -> %d rows", len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize result text (v1 presence-aware)")
    parser.add_argument('--pretrained_model_name', default='alabnii/jmedroberta-base-sentencepiece')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--truncation', action='store_true', default=True)
    parser.add_argument('--no_truncation', dest='truncation', action='store_false')
    parser.add_argument('--add_phi_tokens', action='store_true', default=True)
    parser.add_argument('--no_add_phi_tokens', dest='add_phi_tokens', action='store_false')
    parser.add_argument('--phi_patterns_path', default='config/cleaning/v1/deidentification/phi_patterns.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--source_path', default='data/processed/v1/result.parquet')
    parser.add_argument('--output_dir', default='cache/pretokenized/v1/')
    parser.add_argument('--log_dir', default='outputs/cache/v1/')
    parser.add_argument('--force_regenerate', action='store_true')
    args = parser.parse_args()

    setup_logging(Path(args.log_dir), 'pretokenize_result_v1')

    source_path = Path(args.source_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = generate_output_filename(
        args.pretrained_model_name,
        args.max_length,
        args.truncation,
        args.add_phi_tokens,
    )
    output_path = output_dir / output_filename
    metadata_path = output_dir / f"{output_filename.rsplit('.', 1)[0]}_metadata.yaml"

    if output_path.exists() and not args.force_regenerate:
        logging.info("Cache already exists at %s (use --force_regenerate to overwrite)", output_path)
        return

    logging.info("Initializing TextEmbedder (%s)", args.pretrained_model_name)
    embedder = TextEmbedder(
        pretrained_model_name=args.pretrained_model_name,
        max_length=args.max_length,
        padding=False,
        truncation=args.truncation,
        add_phi_tokens=args.add_phi_tokens,
        phi_patterns_path=args.phi_patterns_path,
        device=args.device,
    )

    tokenizer_args = {
        'pretrained_model_name': args.pretrained_model_name,
        'max_length': args.max_length,
        'padding': False,
        'truncation': args.truncation,
        'add_phi_tokens': args.add_phi_tokens,
        'phi_patterns_path': args.phi_patterns_path,
        'device': args.device,
        'vocab_size': embedder.get_vocab_size(),
        'special_tokens': embedder.added_special_tokens,
    }

    metadata = create_metadata(tokenizer_args, source_path, output_path)

    tokenized_table = tokenize_result_data(source_path, embedder)
    metadata['source_data']['num_samples'] = len(tokenized_table)
    metadata['cached_data']['num_samples'] = len(tokenized_table)

    logging.info("Writing tokenized table to %s", output_path)
    pq.write_table(tokenized_table, output_path)
    metadata['cached_data']['file_hash'] = compute_file_hash(output_path)

    logging.info("Writing metadata to %s", metadata_path)
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


if __name__ == "__main__":
    main()

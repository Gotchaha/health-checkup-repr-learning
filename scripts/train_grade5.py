"""
Train Grade5 downstream classifier with a frozen SSL backbone.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

if not (project_root / "src").exists() or not (project_root / "config").exists():
    raise RuntimeError(f"Not in project root directory. Current: {project_root}")

from src.models import create_embedders_from_config
from src.downstream.grade5.datamodule import create_grade5_data_loaders
from src.downstream.grade5.trainer import Grade5Trainer
from src.downstream.grade5.repr_datamodule import create_repr_grade5_data_loaders
from src.downstream.grade5.repr_trainer import ReprGrade5Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Grade5 downstream classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dry-run", action="store_true", help="Dry run data loading")
    parser.add_argument("--no-timestamp", action="store_true", help="Do not add timestamp")
    parser.add_argument("--test-only", action="store_true", help="Run test only (requires --resume)")
    parser.add_argument("--use-repr", action="store_true", help="Use exported representations")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args) -> dict:
    if args.device:
        config["device"] = args.device
    if not args.no_timestamp:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config["experiment_name"] = f"{config['experiment_name']}_{ts}"
    if args.debug:
        debug_cfg = config.get("debug", {})
        config["training"]["num_epochs"] = debug_cfg.get("num_epochs", config["training"]["num_epochs"])
    return config


def _load_ssl_checkpoint_config(config: dict) -> dict:
    checkpoint_path = config["ssl_backbone"]["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["config"]["model"]["embedders"]


def dry_run_one_batch(trainer, batch: dict, use_repr: bool) -> None:
    batch = trainer._move_batch_to_device(batch)
    with torch.no_grad():
        if use_repr:
            embeddings = batch["embeddings"]
            attention_mask = batch["attention_mask"]
            logits = trainer.prediction_model(embeddings)
            print(f"embeddings: {embeddings.shape}")
        else:
            post_causal_emb, attention_mask = trainer._forward_backbone(batch)
            logits = trainer.prediction_model(post_causal_emb)
            print(f"post_causal_emb: {post_causal_emb.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"logits: {logits.shape}")
    print(f"grade5_targets: {batch['grade5_targets'].shape}")
    print(f"grade5_label_mask: {batch['grade5_label_mask'].shape}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    config = apply_overrides(config, args)

    use_repr = args.use_repr or bool(config.get("data", {}).get("repr_path"))
    if use_repr:
        train_loader, val_loader, test_loader = create_repr_grade5_data_loaders(
            config=config,
            debug=args.debug,
        )
        trainer = ReprGrade5Trainer(
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            resume_from=args.resume,
        )
    else:
        embedders_config = _load_ssl_checkpoint_config(config)
        embedders = create_embedders_from_config(embedders_config, device="cpu")

        train_loader, val_loader, test_loader = create_grade5_data_loaders(
            config=config,
            embedders=embedders,
            debug=args.debug,
        )

        trainer = Grade5Trainer(
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            resume_from=args.resume,
        )

    if args.dry_run:
        first_batch = next(iter(train_loader))
        dry_run_one_batch(trainer, first_batch, use_repr=use_repr)
        return

    if args.test_only:
        if not args.resume:
            raise ValueError("--test-only requires --resume")
        trainer.test()
        return

    trainer.train()
    if test_loader is not None:
        trainer.test()


if __name__ == "__main__":
    main()

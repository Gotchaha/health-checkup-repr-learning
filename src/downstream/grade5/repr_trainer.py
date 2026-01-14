# src/downstream/grade5/repr_trainer.py

"""
Trainer for Grade5 classification using exported representations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .model import Grade5LinearHead
from .metrics import (
    accuracy_from_confusion,
    macro_f1_from_confusion,
    macro_recall_from_confusion,
    compute_confusion_matrix,
    compute_per_patient_macro_f1,
    compute_per_patient_macro_f1_observed,
    normalize_confusion_matrix_rows,
)
from .trainer import Logger, WandbLogger, create_experiment_dirs, setup_reproducibility


class ReprGrade5Trainer:
    """Trainer for Grade5 linear probing on exported embeddings."""

    def __init__(
        self,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dirs = create_experiment_dirs(config)
        self.logger = Logger(self.dirs["log_dir"], config["experiment_name"])
        self.wandb_logger = WandbLogger(config, enabled=config.get("wandb", {}).get("enabled", False))

        setup_reproducibility(
            seed=config.get("seed", 42),
            deterministic=config.get("deterministic", False),
        )

        self.prediction_model = Grade5LinearHead.from_config(config).to(self.device)
        self.class_weights = self._load_class_weights()
        self.ignore_index = config["datamodule"]["label_processing"].get("ignore_index", -100)
        self.num_classes = config["datamodule"]["label_processing"]["num_classes"]

        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_early_stopping()

        mp_cfg = config.get("training", {}).get("mixed_precision", {})
        self.use_amp = mp_cfg.get("enabled", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.step_log_freq = config.get("training", {}).get("logging", {}).get("step_log_freq")

        self.current_epoch = 0
        self.global_step = 0
        self.best_per_patient_f1 = float("-inf")
        self.best_per_exam_f1 = float("-inf")
        self.early_stop_wait = 0

        if resume_from:
            self.load_checkpoint(resume_from)

        self.logger.log_message("ReprGrade5Trainer initialized")

    def _load_class_weights(self) -> torch.Tensor:
        weight_path = self.config["data"]["class_weight_path"]
        with open(weight_path, "r") as f:
            payload = json.load(f)
        label_order = self.config["datamodule"]["label_processing"]["label_order"]
        if payload.get("label_order") and payload["label_order"] != label_order:
            raise ValueError("class weight label_order does not match config label_order")
        weights = [payload["class_weight"][label] for label in label_order]
        return torch.tensor(weights, dtype=torch.float, device=self.device)

    def _setup_optimizer(self) -> None:
        opt_cfg = self.config["training"]["optimizer"]
        params = list(self.prediction_model.parameters())
        if opt_cfg["type"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg["learning_rate"],
                betas=opt_cfg.get("betas", [0.9, 0.999]),
                weight_decay=opt_cfg.get("weight_decay", 0.01),
                eps=opt_cfg.get("eps", 1e-8),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg['type']}")

    def _setup_scheduler(self) -> None:
        sched_cfg = self.config["training"].get("scheduler", {})
        if sched_cfg.get("type") == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            num_epochs = self.config["training"]["num_epochs"]
            self.warmup_epochs = sched_cfg.get("warmup_epochs", 0) or 0
            cosine_epochs = max(num_epochs - self.warmup_epochs, 1)
            base_lr = self.optimizer.param_groups[0]["lr"]
            min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)
            eta_min = base_lr * min_lr_ratio
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs, eta_min=eta_min)
            self.base_lr = base_lr
        else:
            self.scheduler = None
            self.warmup_epochs = 0
            self.base_lr = self.optimizer.param_groups[0]["lr"]

    def _setup_early_stopping(self) -> None:
        es_cfg = self.config["training"].get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.early_stopping_patience = es_cfg.get("patience", 10)
        self.early_stopping_min_delta = es_cfg.get("min_delta", 0.001)

    def _apply_epoch_scheduler(self, epoch: int) -> None:
        if not self.scheduler:
            return
        if self.warmup_epochs and epoch < self.warmup_epochs:
            scale = float(epoch + 1) / float(max(1, self.warmup_epochs))
            for group in self.optimizer.param_groups:
                group["lr"] = self.base_lr * scale
        else:
            self.scheduler.step()

    def _move_batch_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
        return batch_data

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        valid = label_mask & attention_mask
        valid_steps = int(valid.sum().item())
        per_patient_counts = valid.sum(dim=1)
        valid_patients = int((per_patient_counts > 0).sum().item())

        if valid_steps == 0 or valid_patients == 0:
            return torch.tensor(0.0, device=logits.device), valid_patients, valid_steps

        flat_logits = logits.view(-1, self.num_classes)
        flat_targets = targets.view(-1)
        per_pos_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            weight=self.class_weights,
            reduction="none",
            ignore_index=self.ignore_index,
        ).view_as(targets)

        per_patient_loss = (per_pos_loss * valid).sum(dim=1) / per_patient_counts.clamp(min=1)
        per_patient_loss = per_patient_loss[per_patient_counts > 0]
        loss = per_patient_loss.mean()
        return loss, valid_patients, valid_steps

    def train_epoch(self) -> Dict[str, float]:
        self.prediction_model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(self.train_dataloader):
            batch_data = self._move_batch_to_device(batch_data)
            embeddings = batch_data["embeddings"]
            attention_mask = batch_data["attention_mask"]
            targets = batch_data["grade5_targets"]
            label_mask = batch_data["grade5_label_mask"]

            if self.use_amp:
                with autocast(self.device.type):
                    logits = self.prediction_model(embeddings)
                    loss, _, _ = self._compute_loss(logits, targets, label_mask, attention_mask)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.prediction_model(embeddings)
                loss, _, _ = self._compute_loss(logits, targets, label_mask, attention_mask)
                loss.backward()
                self._clip_gradients()
                self.optimizer.step()

            self.optimizer.zero_grad()
            total_loss += float(loss.item())
            num_batches += 1
            self.global_step += 1

            if self.step_log_freq and (batch_idx % self.step_log_freq == 0):
                self.logger.log_message(
                    f"Epoch {self.current_epoch + 1} Step {batch_idx} | "
                    f"loss={loss.item():.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss, "train_num_batches": num_batches}

    def _clip_gradients(self) -> None:
        clip_cfg = self.config["training"].get("gradient_clipping", {})
        if clip_cfg.get("enabled", True):
            max_norm = clip_cfg.get("max_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.prediction_model.parameters(), max_norm)

    def evaluate(self, dataloader: DataLoader, split: str) -> Tuple[Dict[str, float], torch.Tensor]:
        self.prediction_model.eval()
        total_loss = 0.0
        total_batches = 0
        total_f1_sum = 0.0
        total_obs_f1_sum = 0.0
        total_valid_patients = 0
        total_obs_valid = 0
        total_valid_steps = 0
        total_cm = None

        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = self._move_batch_to_device(batch_data)
                embeddings = batch_data["embeddings"]
                attention_mask = batch_data["attention_mask"]
                targets = batch_data["grade5_targets"]
                label_mask = batch_data["grade5_label_mask"]

                logits = self.prediction_model(embeddings)
                loss, valid_patients, valid_steps = self._compute_loss(
                    logits, targets, label_mask, attention_mask
                )
                total_loss += float(loss.item())
                total_batches += 1
                total_valid_patients += valid_patients
                total_valid_steps += valid_steps

                preds = torch.argmax(logits, dim=-1)
                valid = label_mask & attention_mask
                cm = compute_confusion_matrix(preds, targets, valid, self.num_classes).cpu()
                if total_cm is None:
                    total_cm = cm.clone()
                else:
                    total_cm += cm

                per_patient_f1, n_valid = compute_per_patient_macro_f1(
                    logits=logits.detach(),
                    targets=targets.detach(),
                    label_mask=label_mask.detach(),
                    attention_mask=attention_mask.detach(),
                    num_classes=self.num_classes,
                )
                per_patient_obs_f1, n_valid_obs = compute_per_patient_macro_f1_observed(
                    logits=logits.detach(),
                    targets=targets.detach(),
                    label_mask=label_mask.detach(),
                    attention_mask=attention_mask.detach(),
                    num_classes=self.num_classes,
                )
                total_f1_sum += per_patient_f1 * n_valid
                total_obs_f1_sum += per_patient_obs_f1 * n_valid_obs
                total_obs_valid += n_valid_obs

        if total_cm is None:
            total_cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

        if total_cm.sum().item() == 0:
            per_exam_macro_f1 = 0.0
            per_exam_accuracy = 0.0
            per_exam_macro_recall = 0.0
        else:
            per_exam_macro_f1 = macro_f1_from_confusion(total_cm)
            per_exam_accuracy = accuracy_from_confusion(total_cm)
            per_exam_macro_recall = macro_recall_from_confusion(total_cm)

        per_patient_macro_f1 = (
            0.0 if total_valid_patients == 0 else total_f1_sum / total_valid_patients
        )
        per_patient_macro_f1_observed = (
            0.0 if total_obs_valid == 0 else total_obs_f1_sum / total_obs_valid
        )

        avg_loss = total_loss / max(total_batches, 1)
        metrics = {
            f"{split}_loss": avg_loss,
            f"{split}_per_patient_macro_f1": per_patient_macro_f1,
            f"{split}_per_patient_macro_f1_observed": per_patient_macro_f1_observed,
            f"{split}_per_exam_macro_f1": per_exam_macro_f1,
            f"{split}_per_exam_accuracy": per_exam_accuracy,
            f"{split}_per_exam_macro_recall": per_exam_macro_recall,
            f"{split}_valid_patients": total_valid_patients,
            f"{split}_valid_steps": total_valid_steps,
            f"{split}_num_batches": total_batches,
        }
        return metrics, total_cm

    def train(self) -> None:
        num_epochs = self.config["training"]["num_epochs"]
        eval_cfg = self.config.get("evaluation", {})
        val_check_interval = eval_cfg.get("val_check_interval", 1.0)
        if isinstance(val_check_interval, (int, float)) and val_check_interval <= 0:
            val_check_interval = 1.0
        if isinstance(val_check_interval, float) and val_check_interval < 1.0:
            val_check_interval = 1.0
        val_every = int(val_check_interval)
        checkpoint_cfg = self.config["training"].get("checkpointing", {})
        save_every = checkpoint_cfg.get("save_every_n_epochs", 5)
        primary_metrics = eval_cfg.get("primary_metrics", [])
        primary_metric = primary_metrics[0] if primary_metrics else "per_patient_macro_f1"
        tie_breaker_metric = primary_metrics[1] if len(primary_metrics) > 1 else "per_exam_macro_f1"

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = {}
            val_cm = None
            if self.val_dataloader is not None and ((epoch + 1) % val_every == 0):
                val_metrics, val_cm = self.evaluate(self.val_dataloader, "val")

            epoch_metrics = {**train_metrics, **val_metrics}
            self.logger.log_metrics(epoch_metrics, epoch + 1)
            self.logger.log_message(
                f"Epoch {epoch + 1} | train_loss={train_metrics['train_loss']:.4f} "
                f"val_per_patient_macro_f1={val_metrics.get('val_per_patient_macro_f1', 0.0):.4f} "
                f"val_per_exam_macro_f1={val_metrics.get('val_per_exam_macro_f1', 0.0):.4f}"
            )

            if self.wandb_logger.is_enabled():
                self.wandb_logger.log({f"epoch/{k}": v for k, v in epoch_metrics.items()}, step=self.global_step)

            self._apply_epoch_scheduler(epoch)

            if not val_metrics:
                continue

            primary_key = f"val_{primary_metric}"
            tie_key = f"val_{tie_breaker_metric}"
            primary_value = val_metrics.get(primary_key, 0.0)
            tie_value = val_metrics.get(tie_key, 0.0)

            improved = primary_value > self.best_per_patient_f1 + self.early_stopping_min_delta
            if improved or (
                primary_value == self.best_per_patient_f1 and tie_value > self.best_per_exam_f1
            ):
                self.best_per_patient_f1 = primary_value
                self.best_per_exam_f1 = tie_value
                self.save_checkpoint(is_best=True)
                if val_cm is not None:
                    self._save_confusion_matrices(val_cm, split="val", tag="best")
                self.early_stop_wait = 0
            else:
                self.early_stop_wait += 1

            if self.early_stopping_enabled and self.early_stop_wait >= self.early_stopping_patience:
                last_path = self.save_checkpoint(is_best=False)
                self.logger.log_message(
                    f"Early stopping triggered. Saved last checkpoint: {last_path}"
                )
                break

            if save_every and (epoch + 1) % save_every == 0:
                self.save_checkpoint(is_best=False)

        self.wandb_logger.finish()

    def test(self) -> Dict[str, float]:
        if self.test_dataloader is None:
            self.logger.log_message("No test dataloader provided")
            return {}
        test_metrics, test_cm = self.evaluate(self.test_dataloader, "test")
        self.logger.log_metrics(test_metrics, self.current_epoch + 1)
        self._save_confusion_matrices(test_cm, split="test", tag="final")
        return test_metrics

    def _save_confusion_matrices(self, cm: torch.Tensor, split: str, tag: str) -> None:
        results_dir = self.dirs["results_dir"]
        raw_path = results_dir / f"confusion_matrix_{split}_{tag}.pt"
        norm_path = results_dir / f"confusion_matrix_{split}_{tag}_row_norm.pt"
        torch.save(cm.cpu(), raw_path)
        torch.save(normalize_confusion_matrix_rows(cm).cpu(), norm_path)

    def save_checkpoint(self, is_best: bool = False) -> str:
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        path = os.path.join(self.dirs["checkpoint_dir"], filename)

        checkpoint = {
            "prediction_model_state_dict": self.prediction_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_per_patient_f1": self.best_per_patient_f1,
            "best_per_exam_f1": self.best_per_exam_f1,
            "config": self.config,
        }
        if self.use_amp and self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        self.logger.log_message(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.prediction_model.load_state_dict(checkpoint["prediction_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_per_patient_f1 = checkpoint.get("best_per_patient_f1", float("-inf"))
        self.best_per_exam_f1 = checkpoint.get("best_per_exam_f1", float("-inf"))

import os
import sys
import time
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from models.model import create_model
from models.loss import VigilMultiTaskLoss, HumanLoss, AnomalyLoss
from train.dataset import make_multi_dataset, build_targets
from train.config import get_stage_config, load_train_config


def _collate_fn(batch):
    images = torch.stack([s.image for s in batch])
    targets = [build_targets(s) for s in batch]
    return images, targets


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)


class CosineWarmupScheduler:
    """Linear warmup + cosine decay 学习率调度器。"""
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self.current_step = 0
        self.steps_per_epoch = 1

    def set_steps_per_epoch(self, n):
        self.steps_per_epoch = max(1, n)

    def step(self, epoch, step_in_epoch):
        self.current_step = epoch * self.steps_per_epoch + step_in_epoch
        total_steps = self.total_epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch

        if self.current_step < warmup_steps:
            lr = self.base_lr * (self.current_step / max(1, warmup_steps))
        else:
            progress = (self.current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(self, config_path: str, stage: str, device: str = None):
        self.config = get_stage_config(load_train_config(config_path), stage)
        self.stage = stage
        self.device = device or self.config.get("device", "cpu")
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.patience_counter = 0
        self.checkpoint_dir = "checkpoints"

        # ── 模型 ──
        model_cfg = self.config
        variant = model_cfg.get("variant", "n")
        pretrained = model_cfg.get("pretrained")
        self.model = create_model(variant=variant, pretrained_path=pretrained)
        self.model.to(self.device)
        self.model.train()

        freeze = model_cfg.get("freeze", [])
        for name in freeze:
            if hasattr(self.model, name):
                for p in getattr(self.model, name).parameters():
                    p.requires_grad_(False)
                logger.info(f"  Frozen: {name}")

        self.ema = ModelEMA(self.model, decay=model_cfg.get("ema_decay", 0.9999))

        # ── 混合精度 ──
        use_amp = model_cfg.get("amp", False) and self.device == "cuda"
        self.amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ── 损失 ──
        loss_cfg = model_cfg.get("loss", {})
        human_loss = HumanLoss(
            box_w=loss_cfg.get("human", {}).get("box", 7.5),
            cls_w=loss_cfg.get("human", {}).get("person", 0.5),
            helmet_w=loss_cfg.get("human", {}).get("helmet", 1.0),
            smoking_w=loss_cfg.get("human", {}).get("smoking", 1.0),
            kpt_w=loss_cfg.get("human", {}).get("keypoint", 12.0))
        anomaly_loss = AnomalyLoss(
            box_w=loss_cfg.get("anomaly", {}).get("box", 7.5),
            cls_w=loss_cfg.get("anomaly", {}).get("cls", 1.5),
            mask_w=loss_cfg.get("anomaly", {}).get("mask", 1.0))
        balance = model_cfg.get("balance", "manual")
        self.criterion = VigilMultiTaskLoss(
            human_loss=human_loss, anomaly_loss=anomaly_loss,
            balance=balance,
            h_w=model_cfg.get("human_weight", 1.0),
            a_w=model_cfg.get("anomaly_weight", 0.5))

        # ── 优化器 ──
        opt_cfg = model_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 5e-4))
        self.base_lr = opt_cfg.get("lr", 1e-3)

        # ── LR 调度器 ──
        sched_cfg = model_cfg.get("lr_schedule", {})
        sched_type = sched_cfg.get("type", "cosine")
        total_epochs = self.config.get("epochs", 100)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=opt_cfg.get("warmup_epochs", 3),
            total_epochs=total_epochs,
            base_lr=self.base_lr,
            min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.01))

        # ── 训练数据 ──
        input_size = model_cfg.get("input_size", 640)
        if isinstance(input_size, list):
            input_size = input_size[0]

        self.train_dataset = make_multi_dataset(
            model_cfg.get("datasets", {}), augment=True,
            input_size=input_size, split="train")
        if self.train_dataset is None:
            raise RuntimeError("No training datasets found!")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=model_cfg.get("batch_size", 2),
            shuffle=True, num_workers=model_cfg.get("num_workers", 0),
            collate_fn=_collate_fn, pin_memory=(device == "cuda"))

        # ── 验证数据 ──
        val_cfg = model_cfg.get("validation", {})
        self.val_enabled = val_cfg.get("enabled", True)
        self.val_dataset = None
        self.val_loader = None
        if self.val_enabled:
            try:
                self.val_dataset = make_multi_dataset(
                    model_cfg.get("datasets", {}), augment=False,
                    input_size=input_size, split="val")
                if self.val_dataset is not None and len(self.val_dataset) > 0:
                    val_bs = val_cfg.get("batch_size", model_cfg.get("batch_size", 2))
                    self.val_loader = DataLoader(
                        self.val_dataset, batch_size=val_bs,
                        shuffle=False, num_workers=model_cfg.get("num_workers", 0),
                        collate_fn=_collate_fn, pin_memory=(device == "cuda"))
                    logger.info(f"  Val: {len(self.val_dataset)} samples")
                else:
                    logger.warning("  Val dataset empty, disabling validation")
                    self.val_enabled = False
            except Exception as e:
                logger.warning(f"  Val dataset init failed: {e}, disabling validation")
                self.val_enabled = False

        self.scheduler.set_steps_per_epoch(len(self.train_loader))
        self.steps_per_epoch = len(self.train_loader)

        # ── 配置参数 ──
        self.save_interval = model_cfg.get("save_interval", 5)
        self.val_interval = model_cfg.get("val_interval", 1)
        self.log_interval = model_cfg.get("log_interval", 50)
        self.early_stop_patience = val_cfg.get("early_stop_patience", 0)
        self.topk_checkpoints = val_cfg.get("topk_checkpoints", 3)

    def _compute_loss(self, outputs, targets_batch):
        B = len(targets_batch)
        total = 0.0
        metrics = {}
        for i in range(B):
            single_out = {}
            for k, v in outputs.items():
                if isinstance(v, list):
                    single_out[k] = [t[i:i + 1] for t in v]
                elif v is not None and v.dim() >= 1:
                    single_out[k] = v[i:i + 1]
                else:
                    single_out[k] = v
            loss = self.criterion(single_out, targets_batch[i])
            total += loss["total"]
            for mk in ["human_total", "anomaly_total", "human_box", "human_kpt",
                        "human_helmet", "human_smoking", "anomaly_box", "anomaly_cls"]:
                metrics[mk] = metrics.get(mk, 0.0) + loss.get(mk, 0.0)
        return total / B, {k: v / B for k, v in metrics.items()}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        epoch_metrics = {}

        for step, (images, targets_batch) in enumerate(self.train_loader):
            lr = self.scheduler.step(epoch, step)
            images = images.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.amp):
                outputs = self.model(images)
            # loss 计算需在 FP32 下进行，避免 CIoU 的 d²/c² 溢出
            outputs = {k: [o.float() for o in v] if isinstance(v, list) else v.float()
                       for k, v in outputs.items() if isinstance(v, (torch.Tensor, list))}
            loss, metrics = self._compute_loss(outputs, targets_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  NaN/Inf loss at step {step}, skipping batch")
                continue

            self.optimizer.zero_grad()
            if not loss.requires_grad:
                # 可能全 batch 样本都没有有效 GT, 跳过
                if step == 0:
                    logger.warning("  Loss requires no grad, skipping batch "
                                   "(all samples have empty targets?)")
                continue

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.ema.update(self.model)

            total_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            if step > 0 and step % self.log_interval == 0:
                avg = total_loss / step
                logger.info(
                    f"E{epoch:03d} [{step:04d}/{self.steps_per_epoch}] "
                    f"loss={loss.item():.3f} avg={avg:.3f} "
                    f"H={metrics.get('human_total', 0):.3f} A={metrics.get('anomaly_total', 0):.3f} "
                    f"lr={lr:.6f}")

        n = max(self.steps_per_epoch, 1)
        avg_loss = total_loss / n
        avg_metrics = {k: v / n for k, v in epoch_metrics.items()}
        logger.info(f"Epoch {epoch:03d} done — avg_loss={avg_loss:.4f}  "
                     f"H={avg_metrics.get('human_total', 0):.3f} "
                     f"A={avg_metrics.get('anomaly_total', 0):.3f}")
        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self):
        """在验证集上计算损失。"""
        if not self.val_enabled or self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        n_batches = 0

        for images, targets_batch in self.val_loader:
            images = images.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.amp):
                outputs = self.model(images)
            outputs = {k: [o.float() for o in v] if isinstance(v, list) else v.float()
                       for k, v in outputs.items() if isinstance(v, (torch.Tensor, list))}
            loss, metrics = self._compute_loss(outputs, targets_batch)
            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            n_batches += 1

        n = max(n_batches, 1)
        avg_loss = total_loss / n
        avg_metrics = {k: v / n for k, v in all_metrics.items()}
        logger.info(f"  Val — loss={avg_loss:.4f}  "
                     f"H={avg_metrics.get('human_total', 0):.3f} "
                     f"A={avg_metrics.get('anomaly_total', 0):.3f} "
                     f"box={avg_metrics.get('human_box', 0):.3f} "
                     f"helmet={avg_metrics.get('human_helmet', 0):.3f} "
                     f"kpt={avg_metrics.get('human_kpt', 0):.3f}")
        self.model.train()
        return avg_loss, avg_metrics

    def save_checkpoint(self, path: str, val_loss: float = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "stage": self.stage,
        }
        if val_loss is not None:
            ckpt["val_loss"] = val_loss
        torch.save(ckpt, path)
        logger.info(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ema.ema.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", -1)
        logger.info(f"Resumed from {path} at epoch {ckpt['epoch']}")

    def train(self, total_epochs=None):
        total_epochs = total_epochs or self.config.get("epochs", 100)
        base = self.config.get("output", f"checkpoints/{self.stage}")
        output_last = base if base.endswith(".pt") else base.rstrip("/\\") + "_last.pt"
        output_best = output_last.replace("_last.pt", "_best.pt")
        logger.info(f"[{self.stage}] {total_epochs} epochs, "
                     f"{self.steps_per_epoch} steps/epoch, "
                     f"val={'on' if self.val_enabled else 'off'}")

        for ep in range(self.epoch, total_epochs):
            self.epoch = ep
            t0 = time.time()

            train_loss, train_metrics = self.train_epoch(ep)
            elapsed = time.time() - t0
            logger.info(f"Epoch {ep:03d} time={elapsed:.1f}s lr={self.scheduler.get_lr():.6f}")

            # 验证
            val_loss = None
            if self.val_enabled and self.val_interval > 0 and ep % self.val_interval == 0:
                val_loss, val_metrics = self.validate()
            else:
                val_loss = train_loss  # 无验证集时用训练损失作为代理

            # 定期保存
            if self.save_interval > 0 and ep % self.save_interval == 0:
                self.save_checkpoint(output_last, val_loss)

            # 最佳模型保存 (基于验证损失)
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = ep
                self.patience_counter = 0
                self.save_checkpoint(output_best, val_loss)
                logger.info(f"  → new best at epoch {ep} (val_loss={val_loss:.4f})")
            elif val_loss is not None:
                self.patience_counter += 1

            # 早停
            if self.early_stop_patience > 0 and self.patience_counter >= self.early_stop_patience:
                logger.info(f"Early stopping at epoch {ep} "
                             f"(no improvement for {self.early_stop_patience} epochs)")
                break

        logger.info(f"Training complete. Best: epoch={self.best_epoch} "
                     f"val_loss={self.best_val_loss:.4f}")

"""训练器: 多数据集联合训练, 两阶段."""

import os, time, math
import torch
from pathlib import Path
from collections import defaultdict

from models import CenterAssigner, VigilLoss
from train.dataset import make_dataloaders

STRIDES = [4, 8, 16, 32]


class VigilTrainer:
    def __init__(self, model, device="cpu",
                 lr=1e-3, weight_decay=1e-4, warmup_epochs=1,
                 grad_clip=20.0,
                 save_dir="checkpoints", log_interval=10):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr

        self.assigner = CenterAssigner(STRIDES)
        self.loss_fn = VigilLoss()

        self.current_epoch = 0
        self.best_loss = float("inf")

    # ── LR ──
    def _get_lr(self, epoch, max_epochs):
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / max(1, max_epochs - self.warmup_epochs)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ── 构建 targets ──
    def _build_targets(self, sample):
        gt_boxes, gt_classes = [], []
        n_p = len(sample.person_boxes)
        if n_p > 0:
            gt_boxes.append(sample.person_boxes)
            gt_classes.append(torch.zeros(n_p, dtype=torch.long, device=self.device))
        if sample.detect_boxes.numel() > 0:
            n_d = len(sample.detect_boxes)
            gt_boxes.append(sample.detect_boxes)
            gt_classes.append((sample.detect_classes + 1).to(self.device))

        if not gt_boxes:
            return (torch.empty(0, 4, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device), {})

        all_boxes = torch.cat(gt_boxes, dim=0).to(self.device)
        all_classes = torch.cat(gt_classes, dim=0)
        attrs = {}
        if n_p > 0:
            if sample.person_kpts.numel() > 0:
                attrs["kpts"] = sample.person_kpts.to(self.device)
            if sample.person_helmet.numel() > 0:
                attrs["helmet"] = sample.person_helmet.to(self.device)
            if sample.person_smoke.numel() > 0:
                attrs["smoking"] = sample.person_smoke.to(self.device)
        return all_boxes, all_classes, attrs

    # ── 一个 epoch ──
    def train_epoch(self, loaders, max_epochs):
        self.model.train()
        metrics = defaultdict(float)
        n_batches = sum(len(dl) for dl in loaders.values())

        iters = {name: iter(dl) for name, dl in loaders.items()}
        done = set()
        step = 0
        running_detail = defaultdict(float)  # 累计区间内各损失

        while len(done) < len(loaders):
            for name, dl_iter in list(iters.items()):
                if name in done:
                    continue
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    done.add(name)
                    continue

                lr = self._get_lr(self.current_epoch, max_epochs)
                self._set_lr(lr)

                total_loss = torch.tensor(0.0, device=self.device)
                loss_detail = {}

                for sample in batch:
                    img = sample.image.unsqueeze(0).to(self.device)
                    gt_boxes, gt_classes, attrs = self._build_targets(sample)

                    head_outs = self.model(img)
                    feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]

                    targets = self.assigner(
                        [gt_boxes], [gt_classes],
                        [attrs] if attrs else [{}], feat_sizes)

                    losses = self.loss_fn(head_outs, targets, STRIDES, feat_sizes)
                    loss = (losses["cls"] + losses["bbox"] + losses["obj"] +
                            losses["kpt"] + losses["helmet"] + losses["smoke"])
                    total_loss += loss

                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            loss_detail[k] = loss_detail.get(k, 0) + v.item()

                avg_loss = total_loss / len(batch)
                self.optimizer.zero_grad()
                avg_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                metrics["loss"] += avg_loss.item()
                for k, v in loss_detail.items():
                    running_detail[k] += v / len(batch)
                step += 1

                if step % self.log_interval == 0:
                    pct = step / n_batches * 100
                    parts = [f"loss={avg_loss.item():.4f}"]
                    for k in ["cls", "bbox", "obj", "kpt", "helmet", "smoke"]:
                        if k in running_detail:
                            parts.append(f"{k}={running_detail[k]/self.log_interval:.4f}")
                    parts.append(f"lr={lr:.2e}")
                    print(f"  [{step}/{n_batches} {pct:.0f}%] " + " ".join(parts))
                    running_detail.clear()

        for k in metrics:
            metrics[k] /= step
        return metrics

    # ── 验证 ──
    @torch.no_grad()
    def validate(self, loaders):
        self.model.eval()
        metrics = defaultdict(float)
        n = 0

        for dl in loaders.values():
            for batch in dl:
                for sample in batch:
                    img = sample.image.unsqueeze(0).to(self.device)
                    gt_boxes, gt_classes, attrs = self._build_targets(sample)

                    head_outs = self.model(img)
                    feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]

                    targets = self.assigner(
                        [gt_boxes], [gt_classes],
                        [attrs] if attrs else [{}], feat_sizes)

                    losses = self.loss_fn(head_outs, targets, STRIDES, feat_sizes)
                    total = 0.0
                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            metrics["val_" + k] += v.item()
                            total += v.item()
                    metrics["val_loss"] += total
                    n += 1

        for k in metrics:
            metrics[k] /= max(n, 1)
        return metrics

    # ── Checkpoint ──
    def save(self, path, metrics=None):
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, str(path))
        print(f"  Saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        print(f"  Loaded: {path} (epoch {self.current_epoch})")

    # ── 完整训练 ──
    def fit(self, epochs, train_loaders, val_loaders=None, save_prefix="vigil"):
        print(f"\n{'='*50}")
        print(f"Stage: {save_prefix} | Epochs: {epochs} | Datasets: {list(train_loaders.keys())}")
        print(f"{'='*50}")

        for epoch in range(epochs):
            self.current_epoch = epoch
            t0 = time.time()
            train_m = self.train_epoch(train_loaders, epochs)
            elapsed = time.time() - t0

            log = f"Epoch {epoch+1:3d}/{epochs} | {elapsed:.0f}s | loss={train_m['loss']:.4f}"
            if val_loaders:
                val_m = self.validate(val_loaders)
                parts = [f"val={val_m['val_loss']:.4f}"]
                for k in ["val_cls", "val_bbox", "val_obj", "val_kpt", "val_helmet", "val_smoke"]:
                    if k in val_m:
                        parts.append(f"{k[4:]}={val_m[k]:.4f}")
                log += " | " + " ".join(parts)
                if val_m["val_loss"] < self.best_loss:
                    self.best_loss = val_m["val_loss"]
                    self.save(self.save_dir / f"{save_prefix}_best.pt", val_m)
            else:
                if train_m["loss"] < self.best_loss:
                    self.best_loss = train_m["loss"]
            print(log)

            if (epoch + 1) % 10 == 0:
                self.save(self.save_dir / f"{save_prefix}_epoch{epoch+1}.pt")

        self.save(self.save_dir / f"{save_prefix}_last.pt")
        print(f"Best val_loss: {self.best_loss:.4f}")

"""
Vigil 训练入口 (所有配置在 config/train.yaml)
用法:
    python -m train.train --stage pretrain
    python -m train.train --stage finetune
"""

import argparse, sys, yaml
from pathlib import Path
import torch
from models import VigilModel
from train.trainer import VigilTrainer
from train.dataset import make_dataloaders


def load_config(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--device", default=None)     # 覆盖 yaml
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_cfg = cfg[args.stage]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})

    # 合并 CLI 覆盖
    device = args.device or train_cfg.get("device", "cpu")
    epochs = args.epochs or stage_cfg.get("epochs", 50)
    lr = args.lr or stage_cfg.get("optimizer", {}).get("lr", 1e-3)
    batch = args.batch or train_cfg.get("batch_size", 1)
    warmup = stage_cfg.get("optimizer", {}).get("warmup_epochs", 1)
    wd = stage_cfg.get("optimizer", {}).get("weight_decay", 1e-4)
    grad_clip = train_cfg.get("grad_clip", 20.0)
    save_dir = stage_cfg.get("output", f"checkpoints/{args.stage}")
    freeze = stage_cfg.get("freeze", [])
    pretrained = stage_cfg.get("pretrained")

    print(f"Vigil Training: {args.stage}")
    print(f"  config={args.config} device={device} epochs={epochs} lr={lr}")

    # 模型
    w = {"n": 0.5, "s": 1.0}[model_cfg.get("variant", "n")]
    model = VigilModel(backbone_w=w)
    print(f"  params: {model.num_params/1e6:.2f}M")

    # 预训练权重
    if pretrained and Path(pretrained).exists():
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"  pretrained: {pretrained}")

    # 冻结
    for part in freeze:
        if hasattr(model, part):
            for p in getattr(model, part).parameters():
                p.requires_grad = False
            print(f"  frozen: {part}")

    # 数据集
    datasets = stage_cfg["datasets"]
    aug_cfg = cfg.get("augmentation", {})
    val_ratio = train_cfg.get("val_ratio", 0.2)
    print(f"  datasets: {list(datasets.keys())} | val_ratio={val_ratio}")
    result = make_dataloaders(
        datasets, batch_size=batch, augment=aug_cfg, val_ratio=val_ratio, verbose=False)
    if val_ratio > 0:
        train_loaders, val_loaders = result
    else:
        train_loaders, val_loaders = result, {}
    for name, dl in train_loaders.items():
        n_train = len(dl.dataset)
        n_val = len(val_loaders[name].dataset) if name in val_loaders else 0
        print(f"  [{name}] {n_train} train / {n_val} val samples")

    if not train_loaders:
        print("ERROR: no datasets found")
        sys.exit(1)

    # 训练器
    trainer = VigilTrainer(
        model, device=device,
        lr=lr, weight_decay=wd, warmup_epochs=warmup,
        grad_clip=grad_clip, save_dir=save_dir,
    )

    if args.resume:
        trainer.load(args.resume)

    trainer.fit(epochs, train_loaders, val_loaders, save_prefix=args.stage)


if __name__ == "__main__":
    main()

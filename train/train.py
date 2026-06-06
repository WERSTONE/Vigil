"""
Vigil 训练入口 — 模型无关.
用法:
    python -m train.train --stage pretrain
    python -m train.train --stage finetune
    python -m train.train --stage pretrain --model my_model --config config/train.yaml
"""

import argparse, sys, yaml, os
import torch
from models.registry import create_model
from train.trainer import Trainer
from train.dataset import make_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--model", default=None, help="覆盖 yaml 中的 model.name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained", default=None, help="覆盖 stage 的 pretrained 权重路径")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    stage_cfg = cfg[args.stage]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    amp_cfg = cfg.get("amp", {})
    cudnn_cfg = cfg.get("cudnn", {})
    dl_cfg = cfg.get("dataloader", {})

    # ── cuDNN ──
    if cudnn_cfg.get("benchmark", False) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("  cudnn.benchmark: ON")

    device = args.device or train_cfg.get("device", "cpu")
    epochs = args.epochs or stage_cfg.get("epochs", 50)
    lr = args.lr or stage_cfg.get("optimizer", {}).get("lr", 1e-3)
    batch = args.batch or train_cfg.get("batch_size", 1)
    warmup = stage_cfg.get("optimizer", {}).get("warmup_epochs", 1)
    wd = stage_cfg.get("optimizer", {}).get("weight_decay", 1e-4)
    grad_clip = train_cfg.get("grad_clip", 20.0)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 10)
    val_interval = train_cfg.get("val_interval", 1)
    tb_cfg = cfg.get("tensorboard", {})
    use_tb = tb_cfg.get("enabled", False)
    tb_log_dir = tb_cfg.get("log_dir", "logs/train_logs")
    freeze = stage_cfg.get("freeze", [])

    model_name = args.model or model_cfg.get("name", "vigil_v1")
    save_dir = stage_cfg.get("output", f"checkpoints/{model_name}")
    model_kwargs = model_cfg.get("kwargs", {})

    pretrained = args.pretrained or stage_cfg.get("pretrained")
    # finetune 自动查找 pretrain 最佳权重
    if pretrained is None and args.stage == "finetune":
        default_pt = f"checkpoints/{model_name}/pretrain_best.pt"
        if os.path.exists(default_pt):
            pretrained = default_pt
            print(f"  auto pretrained: {default_pt}")

    print(f"Vigil Training: {args.stage}")
    print(f"  model={model_name} device={device} epochs={epochs} lr={lr}")

    model = create_model(model_name, pretrained=pretrained, **model_kwargs)
    print(f"  params: {model.num_params/1e6:.2f}M")

    for part in freeze:
        if hasattr(model, part):
            for p in getattr(model, part).parameters():
                p.requires_grad = False
            print(f"  frozen: {part}")

    datasets = stage_cfg["datasets"]
    aug_cfg = cfg.get("augmentation", {})
    val_ratio = train_cfg.get("val_ratio", 0.2)
    print(f"  datasets: {list(datasets.keys())} | val_ratio={val_ratio}")

    result = make_dataloaders(
        datasets, batch_size=batch, augment=aug_cfg, val_ratio=val_ratio,
        num_workers=dl_cfg.get("num_workers", 0),
        pin_memory=dl_cfg.get("pin_memory", False),
        persistent_workers=dl_cfg.get("persistent_workers", False),
        verbose=False)
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

    trainer = Trainer(
        model, device=device,
        lr=lr, weight_decay=wd, warmup_epochs=warmup,
        grad_clip=grad_clip, log_interval=log_interval,
        save_interval=save_interval, val_interval=val_interval,
        save_dir=save_dir,
        use_tensorboard=use_tb, tb_log_dir=tb_log_dir,
        use_amp=amp_cfg.get("enabled", False),
        amp_dtype=amp_cfg.get("dtype", "float16"),
        map_enabled=cfg.get("map", {}).get("enabled", False),
        map_samples=cfg.get("map", {}).get("val_samples", 500),
    )

    if args.resume:
        trainer.load(args.resume)

    # 提取数据集采样权重 (解决小样本数据集学习不充分问题)
    dataset_weights = {}
    for ds_name, ds_spec in datasets.items():
        w = ds_spec.get("weight", None)
        if w is not None:
            dataset_weights[ds_name] = float(w)

    trainer.fit(epochs, train_loaders, val_loaders, save_prefix=args.stage,
                dataset_weights=dataset_weights if dataset_weights else None)


if __name__ == "__main__":
    main()

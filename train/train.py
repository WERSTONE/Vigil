"""
Vigil 训练入口
用法:
    python -m train.train --stage pretrain
    python -m train.train --stage finetune --config config/train.yaml
    python -m train.train --stage pretrain --device cuda --resume checkpoints/pretrain_last.pt
"""
import argparse
import sys
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Vigil Training")
    parser.add_argument("--stage", default="pretrain", help="训练阶段 (pretrain / finetune)")
    parser.add_argument("--config", default="config/train.yaml", help="配置文件路径")
    parser.add_argument("--device", default=None, help="设备 (cpu / cuda), 默认使用配置文件")
    parser.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数 (用于冒烟测试)")
    args = parser.parse_args()

    trainer = Trainer(args.config, stage=args.stage, device=args.device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train(total_epochs=args.epochs)


if __name__ == "__main__":
    main()

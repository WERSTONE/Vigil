import os
import yaml
from typing import Dict, List, Any


def load_train_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stage_config(config: dict, stage: str) -> dict:
    """提取指定 stage 的完整配置，合并顶层参数。"""
    if stage not in config.get("stages", {}):
        raise ValueError(f"Stage '{stage}' not found. "
                         f"Available: {list(config.get('stages', {}).keys())}")
    sc = dict(config["stages"][stage])

    # 训练超参数
    sc.setdefault("batch_size", config["training"]["batch_size"])
    sc.setdefault("input_size", config["training"]["input_size"])
    sc.setdefault("num_workers", config["training"]["num_workers"])
    sc.setdefault("ema_decay", config["training"]["ema_decay"])
    sc.setdefault("save_interval", config["training"]["save_interval"])
    sc.setdefault("val_interval", config["training"]["val_interval"])
    sc.setdefault("log_interval", config["training"]["log_interval"])
    sc.setdefault("device", config["training"].get("device", "cpu"))
    sc.setdefault("amp", config["training"].get("amp", False))

    # 损失权重
    sc.setdefault("loss", config.get("loss", {}))

    # 增强配置
    sc.setdefault("augmentation", config.get("augmentation", {}))

    # 验证配置
    sc.setdefault("validation", config.get("validation", {}))

    # LR 调度配置
    sc.setdefault("lr_schedule", config.get("lr_schedule", {}))

    # 模型配置
    sc.setdefault("variant", config.get("model", {}).get("variant", "n"))
    sc.setdefault("pretrained", config.get("model", {}).get("pretrained"))

    return sc

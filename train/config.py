"""训练配置加载."""

import yaml
from typing import Dict


def load_train_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stage_config(config: dict, stage: str) -> dict:
    """提取指定 stage 的完整配置, 合并顶层默认值."""
    if stage not in config.get("stages", {}):
        raise ValueError(f"Stage '{stage}' not found. "
                         f"Available: {list(config.get('stages', {}).keys())}")

    sc = dict(config["stages"][stage])

    # 顶层默认值
    top = config.get("training", {})
    for key in ["batch_size", "input_size", "num_workers", "device",
                "ema_decay", "amp", "save_interval", "val_interval",
                "log_interval", "grad_clip"]:
        sc.setdefault(key, top.get(key))

    sc.setdefault("loss", config.get("loss", {}))
    sc.setdefault("augmentation", config.get("augmentation", {}))
    sc.setdefault("validation", config.get("validation", {}))
    sc.setdefault("lr_schedule", config.get("lr_schedule", {}))

    # 模型
    model_cfg = config.get("model", {})
    sc.setdefault("variant", model_cfg.get("variant", "n"))
    sc.setdefault("pretrained", model_cfg.get("pretrained"))

    return sc

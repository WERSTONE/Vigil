"""统一数据集: 读取 58 字段 person 行 + 5 字段检测行."""

import os, json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VigilSample:
    image: torch.Tensor              # [3, 640, 640]
    person_boxes: torch.Tensor       # [N, 4] xyxy
    person_kpts: torch.Tensor        # [N, 17, 3]
    person_helmet: torch.Tensor      # [N] 0=on, 1=off
    person_smoke: torch.Tensor       # [N] 0=no, 1=yes
    detect_boxes: torch.Tensor       # [M, 4] fire/water boxes
    detect_classes: torch.Tensor     # [M] 0=fire, 1=water
    dataset_name: str


# ── 数据集元信息 ──
DATASET_META = {
    "person":     {"has_fire": False, "has_water": False},
    "helmet":     {"has_fire": False, "has_water": False},
    "fire_smoke": {"has_fire": True,  "has_water": False},
    "smoking":    {"has_fire": False, "has_water": False},
    "water_leak": {"has_fire": False, "has_water": True},
}

TRAIN_SIZE = 640


def _parse_label(lbl_path, img_w, img_h, meta):
    """解析标签文件.

    Returns:
        person_boxes: [[x1,y1,x2,y2], ...] 像素坐标
        person_kpts:  [[17,3], ...]
        person_helmet: [0/1, ...]
        person_smoke: [0/1, ...]
        detect_boxes: [[x1,y1,x2,y2], ...]
        detect_classes: [0/1, ...] 0=fire, 1=water
    """
    p_boxes, p_kpts, p_helm, p_smoke = [], [], [], []
    d_boxes, d_cls = [], []

    if not lbl_path.exists():
        return (torch.empty(0,4), torch.empty(0,17,3), torch.empty(0),
                torch.empty(0), torch.empty(0,4), torch.empty(0, dtype=torch.long))

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:]))
            cx, cy, w_n, h_n = vals[0], vals[1], vals[2], vals[3]

            # 归一化 → 像素 xyxy
            x1 = (cx - w_n / 2) * img_w
            y1 = (cy - h_n / 2) * img_h
            x2 = (cx + w_n / 2) * img_w
            y2 = (cy + h_n / 2) * img_h
            box = [x1, y1, x2, y2]

            if cls_id == 0 and len(vals) >= 57:
                # person: 58 字段 (bbox + 51 kpt + helmet + smoke)
                p_boxes.append(box)
                # 关键点
                kpt = np.array(vals[4:55], dtype=np.float32).reshape(17, 3)
                kpt[:, 0] *= img_w
                kpt[:, 1] *= img_h
                p_kpts.append(kpt)
                p_helm.append(int(vals[55]))     # 0=on, 1=off
                p_smoke.append(int(vals[56]))    # 0=no, 1=yes

            elif cls_id == 1 and meta.get("has_fire"):
                # fire box
                d_boxes.append(box)
                d_cls.append(0)  # fire=0

            elif cls_id == 1 and meta.get("has_water"):
                # water box
                d_boxes.append(box)
                d_cls.append(1)  # water=1

    return (
        torch.tensor(p_boxes, dtype=torch.float32) if p_boxes else torch.empty(0, 4),
        torch.tensor(np.stack(p_kpts), dtype=torch.float32) if p_kpts else torch.empty(0, 17, 3),
        torch.tensor(p_helm, dtype=torch.float32) if p_helm else torch.empty(0),
        torch.tensor(p_smoke, dtype=torch.float32) if p_smoke else torch.empty(0),
        torch.tensor(d_boxes, dtype=torch.float32) if d_boxes else torch.empty(0, 4),
        torch.tensor(d_cls, dtype=torch.long) if d_cls else torch.empty(0, dtype=torch.long),
    )


class UnifiedDataset(Dataset):
    """统一数据集.

    读取 data/processed/{name}/ 下的 images/ + labels/,
    返回 VigilSample.
    """

    def __init__(self, root, dataset_name, augment=True):
        self.root = root
        self.name = dataset_name
        self.meta = DATASET_META.get(dataset_name, {})
        self.augment = augment

        # 收集 image-label 对
        img_dir = Path(root) / "images"
        lbl_dir = Path(root) / "labels"
        self.samples = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            self.samples.append((str(img_path), lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        h, w = img.shape[:2]

        # 解析标签
        p_boxes, p_kpts, p_helm, p_smoke, d_boxes, d_cls = _parse_label(
            lbl_path, w, h, self.meta)

        # Resize + pad → 640x640
        scale = TRAIN_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        pad_h = TRAIN_SIZE - new_h
        pad_w = TRAIN_SIZE - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_h - pad_top,
            pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img_t = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0

        # 变换坐标
        def transform_boxes(boxes):
            if boxes.numel() == 0:
                return boxes
            b = boxes.clone()
            b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_left
            b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_top
            return b

        def transform_kpts(kpts):
            if kpts.numel() == 0:
                return kpts
            k = kpts.clone()
            k[:, :, 0] = k[:, :, 0] * scale + pad_left
            k[:, :, 1] = k[:, :, 1] * scale + pad_top
            return k

        p_boxes = transform_boxes(p_boxes)
        p_kpts = transform_kpts(p_kpts)
        d_boxes = transform_boxes(d_boxes)

        return VigilSample(
            image=img_t,
            person_boxes=p_boxes,
            person_kpts=p_kpts,
            person_helmet=p_helm,
            person_smoke=p_smoke,
            detect_boxes=d_boxes,
            detect_classes=d_cls,
            dataset_name=self.name,
        )


def collate_fn(batch):
    """自定义 collate: 返回 list[VigilSample] (不做 batch 堆叠)."""
    return batch


def make_dataloaders(dataset_specs, batch_size=1, augment=True):
    """创建 DataLoader 列表 (每数据集一个).

    Args:
        dataset_specs: {name: {"path": str}, ...}
        batch_size: int
        augment: bool
    """
    from torch.utils.data import DataLoader

    loaders = {}
    for name, spec in dataset_specs.items():
        path = spec["path"]
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        ds = UnifiedDataset(path, name, augment=augment)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0, drop_last=True)
        print(f"  [{name}] {len(ds)} samples")
    return loaders

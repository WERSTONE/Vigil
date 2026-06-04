"""统一数据集: 所有数据集标签格式一致，从 data.yaml 自动读取类映射."""

import os, json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass

TRAIN_SIZE = 640

# data.yaml 中非 person 类名 → 模型检测类别 ID
DETECTION_MAP = {"fire": 0, "water": 1}


@dataclass
class VigilSample:
    image: torch.Tensor              # [3, 640, 640]
    person_boxes: torch.Tensor       # [N, 4] xyxy
    person_kpts: torch.Tensor        # [N, 17, 3]
    person_helmet: torch.Tensor      # [N] 0=on, 1=off
    person_smoke: torch.Tensor       # [N] 0=no, 1=yes
    detect_boxes: torch.Tensor       # [M, 4]
    detect_classes: torch.Tensor     # [M] 0=fire, 1=water
    dataset_name: str


def _load_names(root):
    """从 data.yaml 读取 names 映射."""
    yaml_path = Path(root) / "data.yaml"
    if not yaml_path.exists():
        return {}
    names = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("names:"):
                # 解析 JSON 格式: names: {"0": "person", ...}
                json_str = line.split(":", 1)[1].strip()
                raw = json.loads(json_str)
                names = {int(k): v for k, v in raw.items()}
                break
    return names


def _parse_label(lbl_path, img_w, img_h, names):
    """统一解析标签: class 0=person(58字段), 其他 class=检测框(5字段)."""
    p_boxes, p_kpts, p_helm, p_smoke = [], [], [], []
    d_boxes, d_cls = [], []

    if not lbl_path.exists():
        return (
            torch.empty(0, 4), torch.empty(0, 17, 3), torch.empty(0),
            torch.empty(0), torch.empty(0, 4), torch.empty(0, dtype=torch.long),
        )

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:]))
            cx, cy, w_n, h_n = vals[0], vals[1], vals[2], vals[3]

            x1 = (cx - w_n / 2) * img_w
            y1 = (cy - h_n / 2) * img_h
            x2 = (cx + w_n / 2) * img_w
            y2 = (cy + h_n / 2) * img_h
            box = [x1, y1, x2, y2]

            if cls_id == 0 and len(vals) >= 57:
                # person: bbox + 51 kpt + helmet + smoke
                p_boxes.append(box)
                kpt = np.array(vals[4:55], dtype=np.float32).reshape(17, 3)
                kpt[:, 0] *= img_w
                kpt[:, 1] *= img_h
                p_kpts.append(kpt)
                p_helm.append(int(vals[55]))
                p_smoke.append(int(vals[56]))
            else:
                # 检测框: 从 names 查类别
                cls_name = names.get(cls_id, "")
                detect_id = DETECTION_MAP.get(cls_name)
                if detect_id is not None:
                    d_boxes.append(box)
                    d_cls.append(detect_id)

    return (
        torch.tensor(p_boxes, dtype=torch.float32) if p_boxes else torch.empty(0, 4),
        torch.tensor(np.stack(p_kpts), dtype=torch.float32) if p_kpts else torch.empty(0, 17, 3),
        torch.tensor(p_helm, dtype=torch.float32) if p_helm else torch.empty(0),
        torch.tensor(p_smoke, dtype=torch.float32) if p_smoke else torch.empty(0),
        torch.tensor(d_boxes, dtype=torch.float32) if d_boxes else torch.empty(0, 4),
        torch.tensor(d_cls, dtype=torch.long) if d_cls else torch.empty(0, dtype=torch.long),
    )


class UnifiedDataset(Dataset):
    """统一数据集: 所有数据集共用同一解析逻辑."""

    def __init__(self, root, dataset_name, augment=True):
        self.root = root
        self.name = dataset_name
        self.augment = augment
        self.names = _load_names(root)

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

        p_boxes, p_kpts, p_helm, p_smoke, d_boxes, d_cls = _parse_label(
            lbl_path, w, h, self.names)

        # letterbox → 640x640
        scale = TRAIN_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        pad_h = TRAIN_SIZE - new_h
        pad_w = TRAIN_SIZE - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        img = cv2.copyMakeBorder(
            img, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        def _tx_boxes(b):
            if b.numel() == 0: return b
            b = b.clone()
            b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_left
            b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_top
            return b

        def _tx_kpts(k):
            if k.numel() == 0: return k
            k = k.clone()
            k[:, :, 0] = k[:, :, 0] * scale + pad_left
            k[:, :, 1] = k[:, :, 1] * scale + pad_top
            return k

        return VigilSample(
            image=img_t,
            person_boxes=_tx_boxes(p_boxes),
            person_kpts=_tx_kpts(p_kpts),
            person_helmet=p_helm,
            person_smoke=p_smoke,
            detect_boxes=_tx_boxes(d_boxes),
            detect_classes=d_cls,
            dataset_name=self.name,
        )


def collate_fn(batch):
    return batch


def make_dataloaders(dataset_specs, batch_size=1, augment=True):
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

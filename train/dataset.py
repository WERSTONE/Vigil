import os
import json
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, List, Optional
from dataclasses import dataclass

from train.transforms import TrainTransform, ValTransform


@dataclass
class VigilSample:
    image: torch.Tensor                # [3, 640, 640]
    person_boxes: Optional[torch.Tensor]     # [N, 4] xyxy  人体框
    head_boxes: Optional[torch.Tensor]       # [K, 4] xyxy  头部框
    helmet_labels: Optional[torch.Tensor]    # [K] 0=on, 1=off
    cigarette_boxes: Optional[torch.Tensor]  # [L, 4] xyxy  烟蒂框
    smoking_labels: Optional[torch.Tensor]   # [L] 1=smoking
    keypoints: Optional[torch.Tensor]        # [N, 17, 3]
    anomaly_boxes: Optional[torch.Tensor]    # [M, 4] xyxy
    anomaly_labels: Optional[torch.Tensor]   # [M] class id
    dataset_role: str = ""                   # "person"|"helmet"|"smoking"|"anomaly"


# ── 每个数据集的显式配置 ──

DATASET_CONFIGS = {
    "person": {
        "class_map": {0: "person"},
        "role": "person",
    },
    "helmet": {
        "class_map": {0: "helmet_on", 1: "helmet_off"},
        "role": "helmet",
    },
    "fire_smoke": {
        "class_map": {0: "fire"},
        "role": "anomaly",
        "anomaly_label": 0,  # → SceneAnomalyHead class 0
    },
    "smoking": {
        "class_map": {0: "cigarette"},
        "role": "smoking",
    },
    "water_leak": {
        "class_map": {0: "water"},
        "role": "anomaly",
        "anomaly_label": 1,  # → SceneAnomalyHead class 1
    },
}


# ── YOLO 格式解析 ──

def _parse_yolo_dataset(root: str, dataset_name: str, split: str = "train") -> List[dict]:
    cfg = DATASET_CONFIGS[dataset_name]
    class_map = cfg["class_map"]
    role = cfg["role"]

    img_dir = os.path.join(root, "images", split)
    label_dir = os.path.join(root, "labels", split)
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        return []

    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        label_file = os.path.join(label_dir, base + ".txt")
        if not os.path.exists(label_file):
            continue

        entries = {"person": [], "helmet": [], "cigarette": [], "anomaly": []}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in class_map:
                    continue
                cx, cy, w_norm, h_norm = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                mapped = class_map[cls_id]
                if role == "person":
                    entries["person"].append([cx, cy, w_norm, h_norm])
                elif role == "helmet":
                    helmet_label = 0 if mapped == "helmet_on" else 1
                    entries["helmet"].append(([cx, cy, w_norm, h_norm], helmet_label))
                elif role == "smoking":
                    entries["cigarette"].append([cx, cy, w_norm, h_norm])
                elif role == "anomaly":
                    entries["anomaly"].append(([cx, cy, w_norm, h_norm], cfg["anomaly_label"]))

        if not any(entries.values()):
            continue

        samples.append({
            "path": os.path.join(img_dir, fname),
            "role": role,
            "entries": entries,
        })
    return samples


# ── UnifiedDataset ──

class UnifiedDataset(Dataset):
    def __init__(self, root: str, dataset_name: str,
                 split: str = "train", augment: bool = True, input_size: int = 640):
        self.root = root
        self.split = split
        self.input_size = input_size
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]
        self.role = self.cfg["role"]
        self.transform = TrainTransform(input_size) if augment else ValTransform(input_size)

        self.samples = _parse_yolo_dataset(root, dataset_name, split=split)

        # person 数据集: 加载关键点查找表 (按 COCO image_id 索引)
        self._kpt_lookup = None
        if self.role == "person":
            kpt_file = os.path.join(root, "kpt_annotations.json")
            if os.path.exists(kpt_file):
                with open(kpt_file) as f:
                    kpt_data = json.load(f)
                self._kpt_lookup = {}
                for ann in kpt_data.get("annotations", []):
                    img_id = ann["image_id"]
                    self._kpt_lookup.setdefault(img_id, []).append({
                        "bbox": ann["bbox"],
                        "keypoints": ann.get("keypoints", []),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["path"])
        if img is None:
            import warnings
            warnings.warn(f"Corrupt image skipped: {s['path']}")
            # 随机换一张有效样本，避免训练中断
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
        h, w = img.shape[:2]
        entries = s["entries"]
        role = s["role"]

        person_boxes = []
        head_boxes = []
        helmet_labels_list = []
        cigarette_boxes = []
        smoking_labels_list = []
        anomaly_boxes = []
        anomaly_labels_list = []
        human_kpts = []

        if role == "person":
            for cx, cy_ratio, w_norm, h_norm in entries["person"]:
                bw, bh = w_norm * w, h_norm * h
                x1, y1 = cx * w - bw / 2, cy_ratio * h - bh / 2
                person_boxes.append([x1, y1, x1 + bw, y1 + bh])

            if self._kpt_lookup is not None:
                try:
                    img_id = int(Path(s["path"]).stem)
                except ValueError:
                    img_id = None
                if img_id is not None and img_id in self._kpt_lookup:
                    for kpt_ann in self._kpt_lookup[img_id]:
                        kp = kpt_ann["keypoints"]
                        if len(kp) == 51:
                            human_kpts.append([[kp[i], kp[i + 1], kp[i + 2]] for i in range(0, 51, 3)])

        elif role == "helmet":
            for (cx, cy_ratio, w_norm, h_norm), helm_label in entries["helmet"]:
                bw, bh = w_norm * w, h_norm * h
                x1, y1 = cx * w - bw / 2, cy_ratio * h - bh / 2
                head_boxes.append([x1, y1, x1 + bw, y1 + bh])
                helmet_labels_list.append(helm_label)

        elif role == "smoking":
            for cx, cy_ratio, w_norm, h_norm in entries["cigarette"]:
                bw, bh = w_norm * w, h_norm * h
                x1, y1 = cx * w - bw / 2, cy_ratio * h - bh / 2
                cigarette_boxes.append([x1, y1, x1 + bw, y1 + bh])
                smoking_labels_list.append(1)

        elif role == "anomaly":
            for (cx, cy_ratio, w_norm, h_norm), label in entries["anomaly"]:
                bw, bh = w_norm * w, h_norm * h
                x1, y1 = cx * w - bw / 2, cy_ratio * h - bh / 2
                anomaly_boxes.append([x1, y1, x1 + bw, y1 + bh])
                anomaly_labels_list.append(label)

        # 数据增强
        img_t, person_boxes, human_kpts, anomaly_boxes, head_boxes = self.transform(
            img, person_boxes, human_kpts, anomaly_boxes, head_boxes)

        return VigilSample(
            image=img_t,
            person_boxes=torch.tensor(person_boxes, dtype=torch.float32) if person_boxes else None,
            head_boxes=torch.tensor(head_boxes, dtype=torch.float32) if head_boxes else None,
            helmet_labels=torch.tensor(helmet_labels_list, dtype=torch.long) if helmet_labels_list else None,
            cigarette_boxes=torch.tensor(cigarette_boxes, dtype=torch.float32) if cigarette_boxes else None,
            smoking_labels=torch.tensor(smoking_labels_list, dtype=torch.float32) if smoking_labels_list else None,
            keypoints=torch.tensor(human_kpts, dtype=torch.float32) if human_kpts else None,
            anomaly_boxes=torch.tensor(anomaly_boxes, dtype=torch.float32) if anomaly_boxes else None,
            anomaly_labels=torch.tensor(anomaly_labels_list, dtype=torch.long) if anomaly_labels_list else None,
            dataset_role=role,
        )


# ── build_targets ──

def build_targets(sample: VigilSample) -> dict:
    """将 VigilSample 转为损失函数用的 targets dict。

    每个数据集角色提供什么就传什么，不造假：
    - person:  person_boxes + keypoints + smoking=0(默认)
    - helmet:  head_boxes + helmet_labels  (无 person_boxes, loss 走 head_only)
    - smoking: cigarette_boxes + smoking_labels=1  (无 person_boxes, 当前架构无法训练 smoking)
    - anomaly: anomaly_boxes + anomaly_labels
    """
    role = sample.dataset_role

    human = {}
    if role in ("person", "helmet", "smoking"):
        # assigner 匹配用的 GT 框：person→人体框, helmet→头部框, smoking→烟蒂框
        if role == "helmet" and sample.head_boxes is not None and len(sample.head_boxes) > 0:
            human["boxes"] = sample.head_boxes
        elif role == "smoking" and sample.cigarette_boxes is not None and len(sample.cigarette_boxes) > 0:
            human["boxes"] = sample.cigarette_boxes
        elif sample.person_boxes is not None and len(sample.person_boxes) > 0:
            human["boxes"] = sample.person_boxes

        if sample.helmet_labels is not None:
            human["helmet"] = sample.helmet_labels
        if sample.smoking_labels is not None:
            human["smoking"] = sample.smoking_labels
        # 对齐 keypoints 和 person_boxes: kpt_annotations.json 与 labels 可能数量不同
        if sample.keypoints is not None and "boxes" in human:
            n_box = len(human["boxes"])
            n_kpt = len(sample.keypoints)
            if n_kpt < n_box:
                pad = torch.zeros(n_box - n_kpt, 17, 3, dtype=sample.keypoints.dtype)
                human["keypoints"] = torch.cat([sample.keypoints, pad])
                human["kpt_mask"] = torch.cat([
                    torch.ones(n_kpt, dtype=torch.bool),
                    torch.zeros(n_box - n_kpt, dtype=torch.bool)])
            elif n_kpt > n_box:
                human["keypoints"] = sample.keypoints[:n_box]
                human["kpt_mask"] = torch.ones(n_box, dtype=torch.bool)
            else:
                human["keypoints"] = sample.keypoints
                human["kpt_mask"] = torch.ones(n_box, dtype=torch.bool)
        elif sample.keypoints is not None:
            human["keypoints"] = sample.keypoints
        human["loss_weights"] = _make_human_loss_weights(role, sample)

    anomaly = {}
    if sample.anomaly_boxes is not None and len(sample.anomaly_boxes) > 0:
        anomaly["boxes"] = sample.anomaly_boxes
        anomaly["labels"] = sample.anomaly_labels

    return {"human": human, "anomaly": anomaly}


def _make_human_loss_weights(role: str, sample: VigilSample) -> Dict[str, float]:
    """根据数据集实际提供的内容决定哪些损失参与计算。"""
    if role == "person":
        return {
            "box": 1.0,
            "person": 1.0,
            "helmet": 0.0,           # 无 helmet 标注
            "smoking": 1.0,           # 默认 smoking=0，多数人不抽烟
            "kpt": 1.0 if sample.keypoints is not None else 0.0,
        }
    elif role == "helmet":
        return {
            "box": 0.0,               # 无 person_boxes
            "person": 0.0,
            "helmet": 1.0,
            "smoking": 0.0,           # 无 smoking 标注
            "kpt": 0.0,
        }
    elif role == "smoking":
        # 与 helmet 对称: 烟蒂框用于 assigner 匹配, 仅计算 smoking loss
        return {
            "box": 0.0,
            "person": 0.0,
            "helmet": 0.0,
            "smoking": 1.0,
            "kpt": 0.0,
        }
    return {"box": 1.0, "person": 1.0, "helmet": 1.0, "smoking": 1.0, "kpt": 1.0}


# ── make_multi_dataset ──

def make_multi_dataset(dataset_specs: Dict[str, dict],
                       augment: bool = True, input_size: int = 640,
                       split: str = "train") -> ConcatDataset:
    """根据配置创建多数据集合并。

    dataset_specs 来自 train.yaml:
      {"person": {"path": "data/processed/person"}, ...}
    """
    datasets = []
    for name, spec in dataset_specs.items():
        path = spec["path"]
        if not os.path.exists(path):
            print(f"  [skip] {name}: not found at {path}")
            continue

        ds = UnifiedDataset(path, dataset_name=name, split=split,
                            augment=augment, input_size=input_size)
        datasets.append(ds)
        print(f"  [{name}] {len(ds)} samples from {path}/{split}")

    return ConcatDataset(datasets) if datasets else None

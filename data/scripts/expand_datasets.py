"""
数据集扩充脚本 — 从 data/raw/ 转换到 data/processed/ 并划分 train/val。
保持与当前训练数据集格式一致: YOLO txt + data.yaml。

用法:
    python data/scripts/expand_datasets.py              # 处理所有数据集
    python data/scripts/expand_datasets.py --dry-run    # 仅预览，不写文件
    python data/scripts/expand_datasets.py --dataset helmet  # 仅处理指定数据集
"""
import os
import sys
import json
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VAL_RATIO = 0.2
RANDOM_SEED = 42


def ensure_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def write_yolo_label(path, class_id, cx, cy, w, h):
    """写入 YOLO 格式标签 (归一化坐标)。"""
    with open(path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def write_data_yaml(processed_dir, dataset_name, names_dict):
    """写入 data.yaml。"""
    yaml_path = processed_dir / "data.yaml"
    names_str = ", ".join(f'"{k}": "{v}"' for k, v in names_dict.items())
    content = f"""path: "{processed_dir.as_posix()}"
train: "images/train"
val: "images/val"
names: {{{names_str}}}
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def split_train_val(items: List[Tuple], val_ratio: float = VAL_RATIO):
    """随机划分 train/val，保持可复现。"""
    random.seed(RANDOM_SEED)
    items = sorted(items)
    random.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]


# ── Hardhat VOC → YOLO ──

def process_hardhat(dry_run=False):
    """
    Hardhat 数据集: VOC XML → YOLO 格式。
    仅保留 "helmet" (class 0) 和 "head" (class 1) 标注，
    "person" 标注跳过 (非头部框，不用于 helmet 训练)。
    """
    src_dir = RAW_DIR / "Hardhat"
    dst_dir = PROCESSED_DIR / "helmet"
    img_src = src_dir / "images"
    ann_src = src_dir / "annotations"

    if not src_dir.exists():
        print("  [skip] Hardhat not found")
        return

    CLASS_MAP = {"helmet": 0, "head": 1}  # person 跳过

    samples = []
    for xml_path in sorted(ann_src.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text
        img_path = img_src / filename
        if not img_path.exists():
            continue

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[name]
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)
            # 转 YOLO 归一化
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            labels.append((cls_id, cx, cy, bw, bh))

        if labels:
            samples.append((img_path, labels))

    if not samples:
        print("  [skip] No valid Hardhat samples")
        return

    train_items, val_items = split_train_val(samples)

    for split, items in [("train", train_items), ("val", val_items)]:
        img_dst = dst_dir / "images" / split
        lbl_dst = dst_dir / "labels" / split
        ensure_dirs(img_dst, lbl_dst)

        for i, (img_path, labels) in enumerate(items):
            ext = img_path.suffix
            new_name = f"helmet_{i:05d}{ext}"
            if not dry_run:
                shutil.copy2(img_path, img_dst / new_name)
                with open(lbl_dst / f"helmet_{i:05d}.txt", "w") as f:
                    for cls_id, cx, cy, bw, bh in labels:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    if not dry_run:
        write_data_yaml(dst_dir, "helmet",
                        {"0": "helmet_on", "1": "helmet_off", "2": "helmet_none"})

    print(f"  [helmet] {len(train_items)} train + {len(val_items)} val "
          f"(from {len(samples)} Hardhat samples)")


# ── D-Fire ──

def process_fire(dry_run=False):
    """D-Fire 数据集: 已有 YOLO 格式 + train/val 划分，直接复制。"""
    src_dir = RAW_DIR / "fire"
    dst_dir = PROCESSED_DIR / "fire_smoke"

    if not src_dir.exists():
        print("  [skip] D-Fire not found")
        return

    all_samples = []
    for split_name in ["train", "val"]:
        img_dir = src_dir / split_name
        lbl_dir = src_dir / "labels" / split_name
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                all_samples.append((img_path, lbl_path, split_name))

    if not all_samples:
        print("  [skip] No valid D-Fire samples")
        return

    # D-Fire 自带 train/val 划分，直接沿用
    for split in ["train", "val"]:
        items = [(img, lbl) for img, lbl, s in all_samples if s == split]
        img_dst = dst_dir / "images" / split
        lbl_dst = dst_dir / "labels" / split
        ensure_dirs(img_dst, lbl_dst)

        for i, (img_path, lbl_path) in enumerate(items):
            new_name = f"fire_{split}_{i:05d}.jpg"
            if not dry_run:
                shutil.copy2(img_path, img_dst / new_name)
                shutil.copy2(lbl_path, lbl_dst / f"fire_{split}_{i:05d}.txt")

    if not dry_run:
        write_data_yaml(dst_dir, "fire_smoke", {"0": "fire"})

    n_train = sum(1 for _, _, s in all_samples if s == "train")
    n_val = sum(1 for _, _, s in all_samples if s == "val")
    print(f"  [fire_smoke] {n_train} train + {n_val} val (D-Fire original splits)")


# ── Water Leak ──

def process_water_leak(dry_run=False):
    """积水/漏水数据集: 合并 Indoor + Outdoor + Rotated，YOLO 格式。"""
    src_base = (RAW_DIR / "Dataset of Stagnant Water and Wet Surface with Annotations"
                / "Stagnant water and Wet surface Dataset")
    dst_dir = PROCESSED_DIR / "water_leak"

    if not src_base.exists():
        print("  [skip] Water Leak not found")
        return

    samples = []
    for sub in ["Indoor", "Outdoor", "Rotated"]:
        sub_dir = src_base / sub
        if not sub_dir.exists():
            continue
        for img_path in sorted(sub_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = sub_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                samples.append((img_path, lbl_path))

    if not samples:
        print("  [skip] No valid Water Leak samples")
        return

    train_items, val_items = split_train_val(samples)

    for split, items in [("train", train_items), ("val", val_items)]:
        img_dst = dst_dir / "images" / split
        lbl_dst = dst_dir / "labels" / split
        ensure_dirs(img_dst, lbl_dst)

        for i, (img_path, lbl_path) in enumerate(items):
            ext = img_path.suffix.lower()
            new_name = f"leak_{i:05d}{ext}"
            if not dry_run:
                shutil.copy2(img_path, img_dst / new_name)
                with open(lbl_path) as f_in:
                    content = f_in.read()
                with open(lbl_dst / f"leak_{i:05d}.txt", "w") as f_out:
                    f_out.write(content)

    if not dry_run:
        write_data_yaml(dst_dir, "water_leak", {"0": "stagnant_water"})

    print(f"  [water_leak] {len(train_items)} train + {len(val_items)} val "
          f"(from {len(samples)} samples)")


# ── Smoking (CigDet) ──

def process_smoking(dry_run=False):
    """CigDet 数据集: YOLO 格式，合并 train/test 后重新划分。"""
    src_dir = RAW_DIR / "CigDet_dataset"
    dst_dir = PROCESSED_DIR / "smoking"

    if not src_dir.exists():
        print("  [skip] CigDet not found")
        return

    samples = []
    for sub in ["train", "test"]:
        sub_dir = src_dir / sub
        if not sub_dir.exists():
            continue
        for img_path in sorted(sub_dir.glob("*.jpg")):
            lbl_path = sub_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                samples.append((img_path, lbl_path))

    if not samples:
        print("  [skip] No valid CigDet samples")
        return

    train_items, val_items = split_train_val(samples)

    for split, items in [("train", train_items), ("val", val_items)]:
        img_dst = dst_dir / "images" / split
        lbl_dst = dst_dir / "labels" / split
        ensure_dirs(img_dst, lbl_dst)

        for i, (img_path, lbl_path) in enumerate(items):
            new_name = f"smoke_{i:05d}.jpg"
            if not dry_run:
                shutil.copy2(img_path, img_dst / new_name)
                with open(lbl_path) as f_in:
                    content = f_in.read()
                with open(lbl_dst / f"smoke_{i:05d}.txt", "w") as f_out:
                    f_out.write(content)

    if not dry_run:
        write_data_yaml(dst_dir, "smoking", {"0": "smoking"})

    print(f"  [smoking] {len(train_items)} train + {len(val_items)} val "
          f"(from {len(samples)} CigDet samples)")


# ── COCO Person (含关键点) ──

def process_coco_person(dry_run=False):
    """从 COCO val2017 抽取 person 检测 + 关键点数据，合并为一个数据集。
    - 所有 person 图像: YOLO bbox 标签
    - 有 keypoints 的子集: 额外写入 kpt_annotations.json
    """
    src_dir = RAW_DIR / "coco2017"
    dst_dir = PROCESSED_DIR / "person"
    bbox_file = src_dir / "annotations" / "instances_val2017.json"
    kpt_file = src_dir / "annotations" / "person_keypoints_val2017.json"

    if not bbox_file.exists():
        print("  [skip] COCO annotations not found")
        return

    with open(bbox_file) as f:
        bbox_coco = json.load(f)

    img_map = {img["id"]: img for img in bbox_coco["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in bbox_coco["categories"]}
    person_cat_ids = [cid for cid, name in cat_map.items() if name == "person"]

    # 收集所有 person 框标注
    anns_by_img = {}
    for ann in bbox_coco["annotations"]:
        if ann["category_id"] not in person_cat_ids:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # 收集关键点标注 (额外的 keypoints 字段)
    kpt_by_img = {}
    if kpt_file.exists():
        with open(kpt_file) as f:
            kpt_coco = json.load(f)
        for ann in kpt_coco["annotations"]:
            kps = ann.get("keypoints", [])
            if len(kps) == 51 and any(kps[i * 3 + 2] > 0 for i in range(17)):
                kpt_by_img.setdefault(ann["image_id"], []).append({
                    "bbox": ann["bbox"],
                    "keypoints": kps,
                    "category_id": ann["category_id"],
                })

    samples = []
    for img_id, anns in anns_by_img.items():
        img_info = img_map[img_id]
        img_path = src_dir / "val2017" / img_info["file_name"]
        if not img_path.exists():
            continue
        img_w, img_h = img_info["width"], img_info["height"]
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            bw = w / img_w
            bh = h / img_h
            labels.append((0, cx, cy, bw, bh))
        kpts = kpt_by_img.get(img_id, [])
        samples.append((img_path, img_id, labels, kpts))

    if not samples:
        print("  [skip] No valid COCO person samples")
        return

    train_items, val_items = split_train_val(samples)

    # 构建 kpt_annotations.json (供 dataset.py 的关键点查找)
    kpt_annotations = []
    kpt_ann_id = 0

    for split, items in [("train", train_items), ("val", val_items)]:
        img_dst = dst_dir / "images" / split
        lbl_dst = dst_dir / "labels" / split
        ensure_dirs(img_dst, lbl_dst)

        for img_path, img_id, labels, kpts in items:
            new_name = img_path.name
            if not dry_run:
                shutil.copy2(img_path, img_dst / new_name)
                with open(lbl_dst / f"{img_path.stem}.txt", "w") as f:
                    for cls_id, cx, cy, bw, bh in labels:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                for kpt_ann in kpts:
                    kpt_annotations.append({
                        "id": kpt_ann_id,
                        "image_id": img_id,
                        "bbox": kpt_ann["bbox"],
                        "keypoints": kpt_ann["keypoints"],
                    })
                    kpt_ann_id += 1

    if not dry_run:
        write_data_yaml(dst_dir, "person", {"0": "person"})
        if kpt_annotations:
            with open(dst_dir / "kpt_annotations.json", "w") as f:
                json.dump({"annotations": kpt_annotations}, f)

    n_kpt = sum(1 for _, _, _, k in samples if k)
    print(f"  [person] {len(train_items)} train + {len(val_items)} val "
          f"(from {len(samples)} COCO images, {n_kpt} with keypoints)")


# ── 主入口 ──

DATASETS = {
    "helmet":     process_hardhat,
    "fire_smoke": process_fire,
    "water_leak": process_water_leak,
    "smoking":    process_smoking,
    "person":     process_coco_person,
}


def main():
    parser = argparse.ArgumentParser(description="Vigil 数据集扩充")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写文件")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="仅处理指定数据集 (默认全部)")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] 仅预览，不会写入文件\n")

    names = [args.dataset] if args.dataset else DATASETS.keys()

    for name in names:
        fn = DATASETS[name]
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Processing {name}...")
        fn(dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()

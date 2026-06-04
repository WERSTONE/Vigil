"""
原始数据 → processed 格式转换 (hardhat, cig)

hardhat: COCO JSON → YOLO txt
cig: 直接复制 (已是 YOLO 格式)
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def convert_hardhat():
    """hardhat COCO JSON → YOLO txt, 限制到 4000 张."""
    print("[hardhat] Converting COCO → YOLO ...")
    src = ROOT / "data/hardhat"
    dst = ROOT / "data/processed/helmet"
    coco_path = src / "_annotations.coco.json"

    with open(coco_path) as f:
        coco = json.load(f)

    # 建立 image_id → annotations 映射
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # 建立 image_id → file_name 映射
    id_to_img = {img["id"]: img for img in coco["images"]}

    # 限制数量: 随机选 4000
    random.Random(SEED).shuffle(coco["images"])
    selected = coco["images"][:4000]

    # 清理目标目录
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            d = dst / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    n_train = int(len(selected) * 0.8)
    print(f"  {len(selected)} images → {n_train} train / {len(selected)-n_train} val")

    for i, img_info in enumerate(selected):
        split = "train" if i < n_train else "val"
        fname = img_info["file_name"]
        img_src = src / fname

        if not img_src.exists():
            print(f"  WARNING: {fname} not found on disk, skipping")
            continue

        # 复制图片并重命名
        ext = Path(fname).suffix
        new_name = f"helmet_{i:05d}{ext}"
        shutil.copy2(img_src, dst / "images" / split / new_name)

        # 写 YOLO 标签
        w, h = img_info["width"], img_info["height"]
        lbl_path = dst / "labels" / split / f"helmet_{i:05d}.txt"
        img_id = img_info["id"]
        with open(lbl_path, "w") as f:
            for a in anns_by_img.get(img_id, []):
                cls_id = a["category_id"]  # 0=hardhat(helmet_on), 1=no-hardhat(helmet_off)
                x, y, bw, bh = a["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # 写 data.yaml
    with open(dst / "data.yaml", "w") as f:
        f.write('path: "' + str(dst.resolve()) + '"\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('names: {0: "helmet_on", 1: "helmet_off"}\n')

    print(f"  Done: {len(selected)} images → {dst}")


def convert_cig():
    """cig CigDet → YOLO 直接复制."""
    print("[cig] Copying ...")
    src = ROOT / "data/cig/CigDet_dataset"
    dst = ROOT / "data/processed/smoking"

    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            d = dst / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # 合并 train + test → train/val
    all_pairs = []
    for split_name in ["train", "test"]:
        d = src / split_name
        if not d.exists():
            continue
        for txt_path in sorted(d.glob("*.txt")):
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = d / f"{txt_path.stem}{ext}"
                if img_path.exists():
                    all_pairs.append((img_path, txt_path))
                    break

    random.Random(SEED).shuffle(all_pairs)
    n_train = int(len(all_pairs) * 0.8)

    print(f"  {len(all_pairs)} pairs → {n_train} train / {len(all_pairs)-n_train} val")

    for i, (img_path, lbl_path) in enumerate(all_pairs):
        split = "train" if i < n_train else "val"
        ext = img_path.suffix
        new_name = f"smoke_{i:05d}{ext}"
        shutil.copy2(img_path, dst / "images" / split / new_name)
        shutil.copy2(lbl_path, dst / "labels" / split / f"smoke_{i:05d}.txt")

    # 清理 letterbox_cache
    cache = dst / "letterbox_cache"
    if cache.exists():
        shutil.rmtree(cache)

    with open(dst / "data.yaml", "w") as f:
        f.write('path: "' + str(dst.resolve()) + '"\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('names: {0: "cigarette"}\n')

    print(f"  Done: {len(all_pairs)} pairs → {dst}")


if __name__ == "__main__":
    random.seed(SEED)
    convert_hardhat()
    print()
    convert_cig()
    print("\nDone.")

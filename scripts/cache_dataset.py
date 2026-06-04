"""预处理数据集: 将图像统一 letterbox 到 640×640 并缓存为 uint8 .npy 文件.

用法:
    python scripts/cache_dataset.py                    # 全部数据集
    python scripts/cache_dataset.py --dataset person   # 仅 person
    python scripts/cache_dataset.py --split val        # 仅验证集

开销最大的 cv2.resize 只做一次, HSV/flip 保留在训练时随机.
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

CACHE_DIR_NAME = "letterbox_cache"
TARGET_SIZE = 640
FILL_COLOR = 114


def letterbox(img, target_size=640):
    """等比缩放 + 居中填充, 返回 uint8 图像."""
    h, w = img.shape[:2]
    r = target_size / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = target_size - new_w
    dh = target_size - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(FILL_COLOR,) * 3)
    return img, r, (left, top)


def cache_split(root, dataset_name, split):
    """对某个数据集的某个 split 做缓存."""
    img_dir = os.path.join(root, "images", split)
    if not os.path.exists(img_dir):
        print(f"  [skip] {root}/{dataset_name}/{split} — not found")
        return 0

    cache_dir = os.path.join(root, CACHE_DIR_NAME, split)
    os.makedirs(cache_dir, exist_ok=True)

    meta = {}  # stem → (scale, pad_l, pad_t)
    count = 0

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        cache_path = os.path.join(cache_dir, stem + ".npy")

        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            print(f"  [warn] corrupt: {fname}")
            continue

        img_lb, scale, (pad_l, pad_t) = letterbox(img, TARGET_SIZE)
        np.save(cache_path, img_lb)  # uint8, 640×640×3, ~1.2MB per file
        meta[stem] = {"scale": scale, "pad_l": int(pad_l), "pad_t": int(pad_t)}
        count += 1

    # 保存元数据
    meta_path = os.path.join(cache_dir, "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"  [{dataset_name}/{split}] cached {count} images → {cache_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Cache dataset images to letterbox .npy")
    parser.add_argument("--dataset", default=None, help="仅处理某个数据集 (person/helmet/...)")
    parser.add_argument("--split", default=None, help="仅处理 train 或 val")
    args = parser.parse_args()

    datasets = ["person", "helmet", "fire_smoke", "smoking", "water_leak"]
    if args.dataset:
        datasets = [args.dataset]

    splits = ["train", "val"] if args.split is None else [args.split]

    total = 0
    for ds in datasets:
        for sp in splits:
            root = os.path.join("data", "processed", ds)
            if os.path.exists(root):
                total += cache_split(root, ds, sp)

    print(f"\nDone — {total} images cached")


if __name__ == "__main__":
    main()

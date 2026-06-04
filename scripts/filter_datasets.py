"""
数据筛选 + 伪标注脚本 (修复版)

流程:
  1. 预裁剪: helmet→2500, fire→2000, smoking 全保留 (557<2000)
  2. YOLOv8x person 检测 + 匹配 + 添加 person 标签
  3. helmet/smoking: 剔除 person_count > gt_count 的图片
  4. fire: person 标签全部添加，不剔除
  5. 最终裁到 max_total, 重新 80/20 分 train/val
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
PERSON_CONF = 0.35


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def collect_samples(dataset_dir):
    """收集数据集中的 image-label 对 (不区分 train/val)."""
    base = ROOT / dataset_dir
    samples = []
    for split in ["train", "val"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                lbl_path = None
            samples.append({"img": img_path, "lbl": lbl_path})
    return samples


def pre_reduce(dataset_dir, keep_count):
    """随机裁到 keep_count 张, 直接删除多余文件 (无 YOLO, 快速)."""
    samples = collect_samples(dataset_dir)
    if len(samples) <= keep_count:
        print(f"  {len(samples)} total, no reduction needed")
        return samples
    random.Random(SEED).shuffle(samples)
    kept = samples[:keep_count]
    removed = samples[keep_count:]
    for s in removed:
        s["img"].unlink(missing_ok=True)
        if s["lbl"] and s["lbl"].exists():
            s["lbl"].unlink(missing_ok=True)
    print(f"  {len(samples)} → {len(kept)} (deleted {len(removed)})")
    return kept


def detect_persons(model, img_path):
    """YOLO person 检测, 返回 [[x1,y1,x2,y2,conf], ...]."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    results = model(img, conf=PERSON_CONF, classes=[0], verbose=False)
    det = results[0].boxes
    if det is None or len(det) == 0:
        return []
    persons = []
    for box in det:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        persons.append([x1, y1, x2, y2, float(box.conf[0])])
    return persons


def read_label_file(lbl_path):
    """读取标签文件, 返回 [(cls_id, cx, cy, w, h), ...] 和原始行列表."""
    boxes = []
    lines = []
    if lbl_path and lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append((cls_id, cx, cy, bw, bh))
                lines.append(line)
    return boxes, lines


# ══════════════════════════════════════════════
# 匹配逻辑
# ══════════════════════════════════════════════

def match_head_to_person(head_xyxy, person_boxes):
    """head 中心必须在 person 框内, 优先选面积最小的 person."""
    hcx = (head_xyxy[0] + head_xyxy[2]) / 2
    hcy = (head_xyxy[1] + head_xyxy[3]) / 2
    best_idx, best_area = -1, float("inf")
    for i, pb in enumerate(person_boxes):
        if pb[0] <= hcx <= pb[2] and pb[1] <= hcy <= pb[3]:
            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            if area < best_area:
                best_area, best_idx = area, i
    return best_idx


def match_cigarette_to_person(cig_xyxy, person_boxes):
    """cigarette 中心到 person 中心距离 < 0.5*person_diag."""
    ccx = (cig_xyxy[0] + cig_xyxy[2]) / 2
    ccy = (cig_xyxy[1] + cig_xyxy[3]) / 2
    best_idx, best_dist = -1, float("inf")
    for i, pb in enumerate(person_boxes):
        pcx = (pb[0] + pb[2]) / 2
        pcy = (pb[1] + pb[3]) / 2
        diag = np.sqrt((pb[2] - pb[0]) ** 2 + (pb[3] - pb[1]) ** 2)
        dist = np.sqrt((ccx - pcx) ** 2 + (ccy - pcy) ** 2)
        if dist < 0.5 * diag and dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx


# ══════════════════════════════════════════════
# 文件重组 (修复版: 先移后删)
# ══════════════════════════════════════════════

PREFIX_MAP = {"helmet": "helmet", "smoking": "smoke", "fire_smoke": "fire"}


def _resplit(base, kept_samples):
    """重新分配 80% train / 20% val, 重新编号.
    安全操作: 先移到临时目录 → 清旧目录 → 移入新目录."""

    random.Random(SEED).shuffle(kept_samples)
    n_train = int(len(kept_samples) * 0.8)
    prefix = PREFIX_MAP.get(base.name, base.name)
    tmp = base / "_tmp_resplit"
    tmp.mkdir(parents=True, exist_ok=True)

    # Step 1: 全部移到临时目录 (新命名避免冲突)
    mapping = {}  # old_img_path → (split, tmp_img, tmp_lbl)
    train_idx, val_idx = 0, 0
    for i, s in enumerate(kept_samples):
        split = "train" if i < n_train else "val"
        idx = train_idx if split == "train" else val_idx
        if split == "train":
            train_idx += 1
        else:
            val_idx += 1

        ext = s["img"].suffix
        new_stem = f"{prefix}_{idx:05d}"
        tmp_img = tmp / f"{new_stem}{ext}"
        tmp_lbl = tmp / f"{new_stem}.txt"

        if s["img"].exists():
            shutil.move(str(s["img"]), str(tmp_img))
        if s["lbl"] and s["lbl"].exists():
            shutil.move(str(s["lbl"]), str(tmp_lbl))

        mapping[str(s["img"])] = (split, tmp_img, tmp_lbl)

    # Step 2: 清理旧目录结构
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            d = base / sub / split
            if d.exists():
                shutil.rmtree(d)

    # Step 3: 重建目录并移入
    for split in ["train", "val"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

    for old_path, (split, tmp_img, tmp_lbl) in mapping.items():
        dst_img = base / "images" / split / tmp_img.name
        dst_lbl = base / "labels" / split / tmp_lbl.name
        if tmp_img.exists():
            shutil.move(str(tmp_img), str(dst_img))
        if tmp_lbl.exists():
            shutil.move(str(tmp_lbl), str(dst_lbl))

    # Step 4: 清理临时目录
    shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════
# 数据集处理
# ══════════════════════════════════════════════

def process_helmet(model, max_total):
    """helmet: 检测 person → 过滤 person>gt → 补齐 person 框 → 裁到 max_total."""
    base = ROOT / "data/processed/helmet"
    samples = collect_samples("data/processed/helmet")
    random.Random(SEED).shuffle(samples)
    total = len(samples)
    print(f"  total={total}")

    valid, stat_gt, stat_nomatch, stat_noperson = [], 0, 0, 0

    for i, s in enumerate(samples):
        img = cv2.imread(str(s["img"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 读取 head GT (class 0=helmet_on, 1=helmet_off)
        all_boxes, all_lines = read_label_file(s["lbl"])
        head_boxes = [(cls_id, cx, cy, bw, bh) for cls_id, cx, cy, bw, bh in all_boxes
                      if cls_id in (0, 1)]
        if not head_boxes:
            continue

        persons = detect_persons(model, s["img"])
        n_heads = len(head_boxes)
        n_persons = len(persons)

        # 过滤: person > head → 标注不完整, 删除
        if n_persons > n_heads:
            s["img"].unlink(missing_ok=True)
            if s["lbl"] and s["lbl"].exists():
                s["lbl"].unlink(missing_ok=True)
            stat_gt += 1
            continue

        # 匹配 head → person
        matched = set()
        for cls_id, cx, cy, bw, bh in head_boxes:
            head_xyxy = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
            p_idx = match_head_to_person(head_xyxy, persons)
            if p_idx >= 0:
                matched.add(p_idx)

        if not matched:
            stat_nomatch += 1
            valid.append(s)
            continue

        # 添加 person 框 (class 3)
        new_lines = list(all_lines)
        for p_idx in sorted(matched):
            pb = persons[p_idx]
            cx = ((pb[0] + pb[2]) / 2) / w
            cy = ((pb[1] + pb[3]) / 2) / h
            nw = (pb[2] - pb[0]) / w
            nh = (pb[3] - pb[1]) / h
            new_lines.append(f"3 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if s["lbl"]:
            with open(s["lbl"], "w") as f:
                f.write("\n".join(new_lines) + "\n")

        valid.append(s)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] valid={len(valid)} "
                  f"person>gt={stat_gt} no_match={stat_nomatch}")

    # 裁到 max_total
    if len(valid) > max_total:
        random.Random(SEED).shuffle(valid)
        removed = valid[max_total:]
        valid = valid[:max_total]
        for s in removed:
            s["img"].unlink(missing_ok=True)
            if s["lbl"] and s["lbl"].exists():
                s["lbl"].unlink(missing_ok=True)

    _resplit(base, valid)
    print(f"  → final={len(valid)} (person>gt={stat_gt} no_match={stat_nomatch})")
    return len(valid)


def process_smoking(model, max_total):
    """smoking: 检测 person → 过滤 person>gt → 补齐 person 框."""
    base = ROOT / "data/processed/smoking"
    samples = collect_samples("data/processed/smoking")
    random.Random(SEED).shuffle(samples)
    total = len(samples)
    print(f"  total={total}")

    valid, stat_gt, stat_nomatch = [], 0, 0

    for i, s in enumerate(samples):
        img = cv2.imread(str(s["img"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        all_boxes, all_lines = read_label_file(s["lbl"])
        cig_boxes = [(cls_id, cx, cy, bw, bh) for cls_id, cx, cy, bw, bh in all_boxes
                     if cls_id == 0]
        if not cig_boxes:
            continue

        persons = detect_persons(model, s["img"])
        n_persons = len(persons)
        n_cigs = len(cig_boxes)

        if n_persons > n_cigs:
            s["img"].unlink(missing_ok=True)
            if s["lbl"] and s["lbl"].exists():
                s["lbl"].unlink(missing_ok=True)
            stat_gt += 1
            continue

        matched = set()
        for cls_id, cx, cy, bw, bh in cig_boxes:
            cig_xyxy = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
            p_idx = match_cigarette_to_person(cig_xyxy, persons)
            if p_idx >= 0:
                matched.add(p_idx)

        if not matched:
            stat_nomatch += 1
            valid.append(s)
            continue

        new_lines = list(all_lines)
        for p_idx in sorted(matched):
            pb = persons[p_idx]
            cx = ((pb[0] + pb[2]) / 2) / w
            cy = ((pb[1] + pb[3]) / 2) / h
            nw = (pb[2] - pb[0]) / w
            nh = (pb[3] - pb[1]) / h
            new_lines.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if s["lbl"]:
            with open(s["lbl"], "w") as f:
                f.write("\n".join(new_lines) + "\n")

        valid.append(s)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] valid={len(valid)} "
                  f"person>gt={stat_gt} no_match={stat_nomatch}")

    if len(valid) > max_total:
        random.Random(SEED).shuffle(valid)
        removed = valid[max_total:]
        valid = valid[:max_total]
        for s in removed:
            s["img"].unlink(missing_ok=True)
            if s["lbl"] and s["lbl"].exists():
                s["lbl"].unlink(missing_ok=True)

    _resplit(base, valid)
    print(f"  → final={len(valid)} (person>gt={stat_gt} no_match={stat_nomatch})")
    return len(valid)


def process_fire(model, max_total):
    """fire_smoke: 检测 person 并添加标签, 不剔除."""
    base = ROOT / "data/processed/fire_smoke"
    samples = collect_samples("data/processed/fire_smoke")
    random.Random(SEED).shuffle(samples)
    total = len(samples)
    print(f"  total={total}")

    person_added = 0
    for i, s in enumerate(samples):
        img = cv2.imread(str(s["img"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        persons = detect_persons(model, s["img"])
        all_boxes, all_lines = read_label_file(s["lbl"])

        new_lines = list(all_lines)
        for pb in persons:
            cx = ((pb[0] + pb[2]) / 2) / w
            cy = ((pb[1] + pb[3]) / 2) / h
            nw = (pb[2] - pb[0]) / w
            nh = (pb[3] - pb[1]) / h
            new_lines.append(f"2 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            person_added += 1

        if s["lbl"]:
            with open(s["lbl"], "w") as f:
                f.write("\n".join(new_lines) + "\n")

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] persons_added={person_added}")

    if len(samples) > max_total:
        random.Random(SEED).shuffle(samples)
        removed = samples[max_total:]
        samples = samples[:max_total]
        for s in removed:
            s["img"].unlink(missing_ok=True)
            if s["lbl"] and s["lbl"].exists():
                s["lbl"].unlink(missing_ok=True)

    _resplit(base, samples)
    print(f"  → final={len(samples)} persons_added={person_added}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-helmet", type=int, default=3000)
    parser.add_argument("--max-smoking", type=int, default=9999)
    parser.add_argument("--max-fire", type=int, default=2000)
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--skip-fire", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # ═══ Step 1: 预裁剪 (不跑模型) ═══
    print("Step 1: Pre-reduce ...")
    print("[helmet]")
    pre_reduce("data/processed/helmet", min(4000, int(args.max_helmet * 1.25)))
    if not args.skip_fire:
        print("[fire_smoke]")
        pre_reduce("data/processed/fire_smoke", min(4000, int(args.max_fire * 1.25)))
    print(f"[smoking] skip pre-reduce\n")

    # ═══ Step 2: YOLO 处理 ═══
    print(f"Step 2: Loading {args.model} ...")
    model = YOLO(args.model)
    print("OK.\n")

    print(f"[helmet] YOLO + filter + person labels (max={args.max_helmet})")
    n_h = process_helmet(model, args.max_helmet)
    print()

    print(f"[smoking] YOLO + filter + person labels (max={args.max_smoking})")
    n_s = process_smoking(model, args.max_smoking)
    print()

    if not args.skip_fire:
        print(f"[fire_smoke] YOLO + person labels (no filter, max={args.max_fire})")
        process_fire(model, args.max_fire)
    else:
        print("[fire_smoke] SKIP\n")

    print(f"Done. helmet={n_h} smoking={n_s}")


if __name__ == "__main__":
    main()

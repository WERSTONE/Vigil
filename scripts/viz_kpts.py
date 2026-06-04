"""可视化各数据集中的人框+关键点，每个数据集抽3张存到 data/viz_kpts/"""

import cv2, shutil, random
from pathlib import Path
import numpy as np

SEED = 42
random.seed(SEED)

DATASETS = ["helmet", "fire_smoke", "smoking", "water_leak", "person"]
OUT = Path("data/viz_kpts")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# COCO keypoint 骨架连线
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 脸
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
]
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255),
          (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0)]


def draw_one(img_path, lbl_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:]))

            # 画框 (所有类别)
            cx, cy, bw, bh = vals[0], vals[1], vals[2], vals[3]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            if cls_id == 0:  # person
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 关键点
                if len(vals) >= 55:  # 5 + 51 = 56 values total
                    kpts = []
                    for i in range(17):
                        kx = int(vals[4 + i*3] * w)
                        ky = int(vals[4 + i*3 + 1] * h)
                        kv = vals[4 + i*3 + 2]
                        kpts.append((kx, ky, kv))

                    # 画骨架
                    for (a, b), c in zip(SKELETON, COLORS[:len(SKELETON)]):
                        if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
                            cv2.line(img, (kpts[a][0], kpts[a][1]),
                                     (kpts[b][0], kpts[b][1]), (0, 255, 255), 2)

                    # 画关节点
                    for kx, ky, kv in kpts:
                        if kv > 0.5:
                            cv2.circle(img, (kx, ky), 3, (0, 0, 255), -1)

            elif cls_id == 1:
                color = (255, 0, 0)  # blue: helmet_on / fire / cigarette / water
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            elif cls_id == 2:
                color = (0, 0, 255)  # red: helmet_off
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.imwrite(str(out_path), img)


for ds in DATASETS:
    ds_dir = Path(f"data/processed/{ds}")
    if not ds_dir.exists():
        continue
    (OUT / ds).mkdir(exist_ok=True)

    pairs = []
    for img_path in sorted(ds_dir.rglob("images/*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rel = img_path.relative_to(ds_dir / "images")
        lbl_path = ds_dir / "labels" / rel.with_suffix(".txt")
        if not lbl_path.exists():
            lbl_path = ds_dir / "labels" / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    # 采样
    sample = random.sample(pairs, min(5, len(pairs)))
    for i, (img_p, lbl_p) in enumerate(sample):
        out_p = OUT / ds / f"{ds}_{i}.jpg"
        draw_one(img_p, lbl_p, out_p)
        print(f"  {out_p}")

print("\nDone → data/viz_kpts/")

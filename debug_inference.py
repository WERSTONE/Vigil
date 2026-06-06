"""快速测试: v2 修复后的推理输出."""
import torch
import numpy as np
import sys
import os

os.chdir("D:/Vigil")
sys.path.insert(0, "D:/Vigil")

print("="*60)
print("V2 Inference Test (after fixes)")
print("="*60)

from models.registry import create_model
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

model = create_model("vigil_v2").to(device)
model.eval()

# 找一张训练图测试
from pathlib import Path
img_dir = Path("data/processed/person/images")
imgs = sorted(img_dir.glob("*.jpg"))
if not imgs:
    img_dir = Path("data/processed/fire_smoke/images")
    imgs = sorted(img_dir.glob("*.jpg"))

if imgs:
    img_path = str(imgs[0])
    print(f"Test image: {img_path}")

    # 读取标签
    lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    print(f"  GT: class={parts[0]}, boxes={parts[1:5]}")

    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    print(f"  Image size: {w}x{h}")

    # 转为 RGB (detect 接口要求 RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        result = model.detect(frame_rgb)

    print(f"\nDetection result:")
    for cls_name, entry in result.items():
        n = len(entry["boxes"])
        print(f"  {cls_name}: {n} detections")
        if n > 0:
            for i in range(min(3, n)):
                box = entry["boxes"][i]
                score = entry["scores"][i].item() if torch.is_tensor(entry["scores"]) else entry["scores"][i]
                print(f"    [{i}] box={box.tolist() if torch.is_tensor(box) else box}, score={score:.4f}")
else:
    print("No test images found!")

# 也测试一下直接用 forward + decode
print(f"\n--- Raw decode test ---")
from train.dataset import UnifiedDataset, collate_fn
from torch.utils.data import DataLoader
ds = UnifiedDataset("data/processed/person", "test", augment=False)
loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
batch = next(iter(loader))
sample = batch[0]

img = sample.image.unsqueeze(0).to(device)
with torch.no_grad():
    head_outs = model.forward(img)

from models.vigil_v2.head import decode_outputs_v2
boxes, scores, kpts, helmet, smoking = decode_outputs_v2(
    head_outs, model.strides, model.reg_max, score_thresh=0.05)

print(f"All candidates: boxes={boxes.shape}, scores={scores.shape}")
if boxes.shape[1] > 0:
    max_score, best_cls = scores[0].max(dim=-1)
    top_idx = max_score.argsort(descending=True)[:5]
    print(f"Top 5 predictions:")
    for i, idx in enumerate(top_idx):
        idx = idx.item()
        cls_names = ["person", "fire", "water"]
        print(f"  [{i}] {cls_names[best_cls[idx].item()]}: "
              f"score={max_score[idx].item():.4f}, "
              f"box=[{boxes[0,idx,0].item():.0f},{boxes[0,idx,1].item():.0f},"
              f"{boxes[0,idx,2].item():.0f},{boxes[0,idx,3].item():.0f}]")

print(f"\nGT person_boxes: {sample.person_boxes.tolist() if sample.person_boxes.shape[0] > 0 else 'none'}")

print("\nDone.")

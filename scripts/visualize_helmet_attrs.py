"""Create a contact sheet for checking helmet_attr labels."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path("data/processed/helmet")
OUT = Path("logs/helmet_attr_check.jpg")
TILE_W, TILE_H = 360, 300
SAMPLES_PER_CLASS = 6


def parse_person_rows(label_path: Path, img_w: int, img_h: int):
    rows = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 58 or parts[0] != "0":
            continue
        vals = [float(x) for x in parts[1:]]
        cx, cy, bw, bh = vals[:4]
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        rows.append(((x1, y1, x2, y2), int(float(parts[-2]))))
    return rows


def collect_samples():
    buckets = {0: [], 1: []}
    used_images = {0: set(), 1: set()}
    for label_path in sorted((ROOT / "labels").glob("*.txt")):
        img_path = ROOT / "images" / f"{label_path.stem}.jpg"
        if not img_path.exists():
            img_path = ROOT / "images" / f"{label_path.stem}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for box, attr in parse_person_rows(label_path, w, h):
            if (
                attr in buckets
                and len(buckets[attr]) < SAMPLES_PER_CLASS
                and img_path not in used_images[attr]
            ):
                buckets[attr].append((img_path, box, attr))
                used_images[attr].add(img_path)
        if all(len(v) >= SAMPLES_PER_CLASS for v in buckets.values()):
            break
    return buckets


def make_tile(img_path: Path, box, attr: int):
    img = cv2.imread(str(img_path))
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    color = (40, 190, 70) if attr == 0 else (40, 80, 230)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    label = f"helmet_attr={attr}"
    cv2.rectangle(img, (x1, max(0, y1 - 30)), (min(w - 1, x1 + 185), y1), color, -1)
    cv2.putText(img, label, (x1 + 6, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    scale = min(TILE_W / w, TILE_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE_H, TILE_W, 3), 245, dtype=np.uint8)
    top, left = (TILE_H - nh) // 2, (TILE_W - nw) // 2
    tile[top:top + nh, left:left + nw] = resized
    cv2.putText(tile, img_path.name, (8, TILE_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return tile


def main():
    buckets = collect_samples()
    rows = []
    for attr in (0, 1):
        tiles = [make_tile(*sample) for sample in buckets[attr]]
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet)
    print(OUT.resolve())
    print({k: len(v) for k, v in buckets.items()})


if __name__ == "__main__":
    main()

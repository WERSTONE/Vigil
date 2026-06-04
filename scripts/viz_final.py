"""各数据集可视化 10 张，显示 person 框+关键点+属性标注."""

import cv2, shutil, random
from pathlib import Path

SEED = 42
random.seed(SEED)

DATASETS = ["person", "helmet", "fire_smoke", "smoking", "water_leak"]
OUT = Path("data/viz_final")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16),
]

HELMET_NAMES = {0: "helmet_ON", 1: "helmet_OFF"}
SMOKE_NAMES = {0: "no_smoke", 1: "SMOKING"}
OBJ_NAMES = {"fire_smoke": {1: "fire"}, "water_leak": {1: "water"}}


def draw_one(img_path, lbl_path, out_path, ds_name):
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
            cx, cy, bw, bh = vals[0], vals[1], vals[2], vals[3]
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            if cls_id == 0 and len(vals) >= 57:
                # person
                helmet_attr = int(vals[55])
                smoke_attr = int(vals[56])

                # 框颜色：根据头盔属性
                if helmet_attr == 0:
                    box_color = (0, 255, 0)    # 绿=戴了
                else:
                    box_color = (0, 165, 255)  # 橙=没戴

                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

                # 属性文字
                h_text = HELMET_NAMES.get(helmet_attr, "?")
                s_text = SMOKE_NAMES.get(smoke_attr, "?")
                label = f"{h_text} | {s_text}"
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                # 关键点
                if len(vals) >= 55:
                    kpts = [(int(vals[4+i*3]*w), int(vals[5+i*3]*h), vals[6+i*3])
                            for i in range(17)]
                    # 骨架
                    for a, b in SKELETON:
                        if kpts[a][2] > 0.5 and kpts[b][2] > 0.5:
                            cv2.line(img, (kpts[a][0], kpts[a][1]),
                                     (kpts[b][0], kpts[b][1]), (0, 255, 255), 1)
                    # 关节点
                    for kx, ky, kv in kpts:
                        if kv > 0.5:
                            cv2.circle(img, (kx, ky), 2, (0, 0, 255), -1)

            elif cls_id == 1 and ds_name in ("fire_smoke", "water_leak"):
                # fire / water
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                obj_name = OBJ_NAMES.get(ds_name, {}).get(cls_id, f"cls{cls_id}")
                cv2.putText(img, obj_name, (x1, y2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imwrite(str(out_path), img)


for ds in DATASETS:
    ds_dir = Path(f"data/processed/{ds}")
    if not ds_dir.exists():
        continue
    (OUT / ds).mkdir(exist_ok=True)

    pairs = []
    for ip in sorted(ds_dir.rglob("images/*")):
        if ip.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rel = ip.relative_to(ds_dir / "images")
        lp = ds_dir / "labels" / rel.with_suffix(".txt")
        if not lp.exists():
            lp = ds_dir / "labels" / (ip.stem + ".txt")
        if lp.exists():
            pairs.append((ip, lp))

    sample = random.sample(pairs, min(10, len(pairs)))
    for i, (ip, lp) in enumerate(sample):
        draw_one(ip, lp, OUT / ds / f"{ds}_{i:02d}.jpg", ds)

    print(f"{ds}: {len(sample)} saved")

# 图例
legend = OUT / "LEGEND.txt"
legend.write_text(
    "GREEN box  = helmet ON (戴了)\n"
    "ORANGE box = helmet OFF (没戴)\n"
    "RED dots   = keypoints\n"
    "YELLOW lines = skeleton\n"
    "BLUE box   = fire / water detection\n"
    "Label text = helmet_status | smoking_status\n"
)
print(f"\nDone → {OUT}/")

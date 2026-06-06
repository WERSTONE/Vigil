"""调试脚本 v3: 验证 assigner 修复后的效果."""
import torch
import numpy as np
import sys
import os

os.chdir("D:/Vigil")
sys.path.insert(0, "D:/Vigil")

def debug_v2_fixed():
    print("="*60)
    print("V2 Post-Fix Debug")
    print("="*60)

    from models.registry import create_model
    from train.dataset import UnifiedDataset, collate_fn
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = create_model("vigil_v2").to(device)
    print(f"Params: {model.num_params/1e6:.2f}M")

    ds = UnifiedDataset("data/processed/person", "test", augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    batch = next(iter(loader))
    sample = batch[0]
    model.train()

    print(f"\n=== Single sample ===")
    print(f"person_boxes: {sample.person_boxes.shape[0]}, detect_boxes: {sample.detect_boxes.shape[0]}")

    losses = model.compute_loss(sample)
    print("Losses:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")

    # Check assignment per level
    device_t = next(model.parameters()).device
    img = sample.image.unsqueeze(0).to(device_t)
    head_outs = model.forward(img)
    feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]
    pred_scores, pred_boxes = model._decode_for_assigner(head_outs, feat_sizes)
    gt_boxes, gt_classes, attrs = model._build_targets(sample, device_t)
    targets = model.assigner(pred_scores, pred_boxes, gt_boxes, gt_classes, attrs, feat_sizes, model.strides)

    for lvl, t in enumerate(targets):
        if t is not None:
            n_pos = len(t["gt_boxes"])
            n_person = (t["gt_classes"] == 0).sum().item()
            print(f"  L{lvl} (s={model.strides[lvl]}): {n_pos} pos ({n_person} person)")
            if n_pos > 0:
                # GT box info
                gt_b = t["gt_boxes"]
                gt_w = (gt_b[0, 2] - gt_b[0, 0]).item()
                gt_h = (gt_b[0, 3] - gt_b[0, 1]).item()
                print(f"    GT box: {gt_w:.0f}x{gt_h:.0f}, max_side={max(gt_w, gt_h):.0f}")
                # 检查是否有 cell center 不在 GT 框内的情况
                gx, gy = t["grid_xy"][:, 0], t["grid_xy"][:, 1]
                stride = model.strides[lvl]
                cx = (gx.float() + 0.5) * stride
                cy = (gy.float() + 0.5) * stride
                in_box = (cx >= gt_b[:, 0]) & (cx <= gt_b[:, 2]) & (cy >= gt_b[:, 1]) & (cy <= gt_b[:, 3])
                n_outside = (~in_box).sum().item()
                if n_outside > 0:
                    print(f"    WARNING: {n_outside}/{n_pos} cells outside GT box!")
                else:
                    print(f"    All {n_pos} cells inside GT box [OK]")
        else:
            print(f"  L{lvl} (s={model.strides[lvl]}): None")

    # ── 10-step training ──
    print(f"\n=== 10-step training (fixed batch) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    model.train()
    fixed_batch = next(iter(loader))

    for step in range(10):
        total_loss = torch.tensor(0.0, device=device)
        all_num_pos = 0
        loss_components = {}

        for s in fixed_batch:
            losses = model.compute_loss(s)
            total_loss += losses["total"]
            all_num_pos += losses.get("num_pos", 0)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    loss_components[k] = loss_components.get(k, 0) + v.item()

        avg_loss = total_loss / len(fixed_batch)
        optimizer.zero_grad()
        avg_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()

        if step == 0 or step == 9:
            comp_str = " ".join(f"{k}={v/len(fixed_batch):.3f}" for k, v in sorted(loss_components.items()))
            print(f"  step {step+1:2d}: loss={avg_loss.item():.4f} grad={grad_norm.item():.4f} pos={all_num_pos} | {comp_str}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        debug_v2_fixed()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

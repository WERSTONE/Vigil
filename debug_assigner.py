"""验证 v2 assigner→loss 的数据流一致性."""
import torch
import sys, os
os.chdir("D:/Vigil")
sys.path.insert(0, "D:/Vigil")

from models.registry import create_model
from train.dataset import UnifiedDataset, collate_fn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model("vigil_v2").to(device)
model.train()

# 用一个有 person + fire 的数据集来测试混合场景
# person 只有 person boxes, fire_smoke 只有 fire boxes
# 我们用 person 数据集，手动添加一个假 fire box 测试混合场景

ds = UnifiedDataset("data/processed/person", "test", augment=False)
loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
batch = next(iter(loader))
sample = batch[0]

# 手动添加一个 fake detect box 模拟混合场景
if sample.detect_boxes.shape[0] == 0:
    sample.detect_boxes = torch.tensor([[100.0, 100.0, 150.0, 150.0]])
    sample.detect_classes = torch.tensor([1])  # fire

print(f"person_boxes: {sample.person_boxes.shape[0]}, detect_boxes: {sample.detect_boxes.shape[0]}")
print(f"person_boxes: {sample.person_boxes.tolist() if sample.person_boxes.shape[0] > 0 else 'none'}")
print(f"detect_boxes: {sample.detect_boxes.tolist()}")
print(f"detect_classes: {sample.detect_classes.tolist()}")

# 手动运行 compute_loss 内部流程
img = sample.image.unsqueeze(0).to(device)
gt_boxes, gt_classes, attrs = model._build_targets(sample, device)
print(f"\ngt_boxes: {gt_boxes.shape}, gt_classes: {gt_classes.tolist()}")
for k, v in attrs.items():
    print(f"  attrs[{k}]: {v.shape if v is not None else 'None'}")

head_outs = model.forward(img)
feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]
pred_scores, pred_boxes = model._decode_for_assigner(head_outs, feat_sizes)
targets = model.assigner(pred_scores, pred_boxes, gt_boxes, gt_classes, attrs, feat_sizes, model.strides)

print(f"\n=== Assigner output ===")
for lvl, t in enumerate(targets):
    if t is not None:
        n_p = (t["gt_classes"] == 0).sum().item()
        n_f = (t["gt_classes"] == 1).sum().item()
        n_w = (t["gt_classes"] == 2).sum().item()
        print(f"L{lvl}: {len(t['gt_classes'])} pos (person={n_p}, fire={n_f}, water={n_w})")
        if t["gt_kpts"] is not None:
            print(f"  gt_kpts: {t['gt_kpts'].shape}")
            # 验证 kpt 数量 = person 数量
            assert t["gt_kpts"].shape[0] == n_p, f"KPT MISMATCH: {t['gt_kpts'].shape[0]} != {n_p}"
        else:
            print(f"  gt_kpts: None (n_p={n_p})")
            assert n_p == 0, f"Person entries exist but gt_kpts is None!"
        if t["gt_helmet"] is not None:
            print(f"  gt_helmet: {t['gt_helmet'].shape}")
            assert t["gt_helmet"].shape[0] == n_p
        if t["gt_smoking"] is not None:
            print(f"  gt_smoking: {t['gt_smoking'].shape}")
            assert t["gt_smoking"].shape[0] == n_p
    else:
        print(f"L{lvl}: None")

# 运行 loss 验证无 crash
print(f"\n=== Loss test ===")
losses = model.compute_loss(sample)
for k, v in losses.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.item():.4f}")
    else:
        print(f"  {k}: {v}")

# 反向传播验证
losses["total"].backward()
print("Backward: OK")

print("\nAll checks passed!")

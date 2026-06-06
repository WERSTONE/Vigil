"""等容量解耦检测头 — 所有分支相同深度和通道数.

v3 相比 v2: kpt/attr 分支不再用 64 通道瓶颈，所有分支平等:
  cls:  3-layer tower (in_ch → in_ch) → 3 classes
  reg:  3-layer tower (in_ch → in_ch) → 4×reg_max DFL
  kpt:  3-layer tower (in_ch → in_ch) → 17×3 keypoints
  attr: 3-layer tower (in_ch → in_ch) → 2 attributes (helmet, smoking)
"""

import torch
import torch.nn as nn
from models.common import Conv


def _make_tower(in_ch, depth):
    if depth == 0:
        return nn.Identity()
    layers = [Conv(in_ch, in_ch, 3) for _ in range(depth)]
    return nn.Sequential(*layers)


class VigilHeadV3(nn.Module):
    """等容量多任务检测头.

    Args:
        in_ch: 输入通道 (来自 neck)
        num_classes: 类别数 (person/fire/water = 3)
        num_kpts: 关键点数 (COCO 17)
        reg_max: DFL bin 数
        tower_depth: 每个分支的 conv 层数
    """

    def __init__(self, in_ch=256, num_classes=3, num_kpts=17, reg_max=16,
                 tower_depth=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_kpts = num_kpts
        self.reg_max = reg_max

        # ── 分类分支 ──
        self.cls_tower = _make_tower(in_ch, tower_depth)
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        # ── 回归分支 ──
        self.reg_tower = _make_tower(in_ch, tower_depth)
        self.reg_pred = nn.Conv2d(in_ch, 4 * reg_max, 1)

        # ── 关键点分支 (等容量, 不再瓶颈) ──
        self.kpt_tower = _make_tower(in_ch, tower_depth)
        self.kpt_pred = nn.Conv2d(in_ch, num_kpts * 3, 1)

        # ── 属性分支 (等容量) ──
        self.attr_tower = _make_tower(in_ch, tower_depth)
        self.attr_pred = nn.Conv2d(in_ch, 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_pred.bias, -4.595)   # 降低初始假阳性

    def forward(self, features):
        """features: List[[B, C, H, W]] (3 个尺度) → dict of List[Tensor]."""
        outs = {"cls": [], "reg": [], "kpt": [], "attr": []}
        for f in features:
            outs["cls"].append(self.cls_pred(self.cls_tower(f)))
            outs["reg"].append(self.reg_pred(self.reg_tower(f)))
            outs["kpt"].append(self.kpt_pred(self.kpt_tower(f)))
            outs["attr"].append(self.attr_pred(self.attr_tower(f)))
        return outs


# ═══════════════════════════════════════════════════════════════
# DFL 解码 (推理用)
# ═══════════════════════════════════════════════════════════════

def _make_grid(nx, ny, device):
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device), indexing="ij")
    return torch.stack((xv, yv), 2).float()


def _dfl_decode(reg_pred, reg_max, stride, grid):
    B, _, H, W = reg_pred.shape
    N = H * W
    reg = reg_pred.view(B, 4, reg_max, N)
    reg = reg.softmax(dim=-2)
    proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
    reg = (reg * proj.view(1, 1, reg_max, 1)).sum(dim=-2)
    reg = reg * stride

    g = grid.view(1, N, 2) + 0.5 * stride
    cx = g[..., 0:1].transpose(1, 2)
    cy = g[..., 1:2].transpose(1, 2)

    l, t = reg[:, 0:1], reg[:, 1:2]
    r, b = reg[:, 2:3], reg[:, 3:4]

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    return torch.cat([x1, y1, x2, y2], dim=1).transpose(1, 2)


def decode_outputs_v3(head_outs, strides, reg_max, score_thresh=0.05):
    """多级 head 输出 → 检测候选."""
    device = head_outs["cls"][0].device
    B = head_outs["cls"][0].shape[0]

    all_boxes, all_scores = [], []
    all_kpts, all_helmet, all_smoke = [], [], []

    for lvl, stride in enumerate(strides):
        _, _, H, W = head_outs["cls"][lvl].shape
        N = H * W

        cls_pred = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, N, -1)
        scores = cls_pred.sigmoid()

        grid = _make_grid(W, H, device) * stride
        boxes = _dfl_decode(head_outs["reg"][lvl], reg_max, stride, grid)

        kpt_pred = head_outs["kpt"][lvl]
        kpt_pred = kpt_pred.permute(0, 2, 3, 1).reshape(B, N, 17, 3)
        grid_center = grid.view(1, N, 1, 2) + 0.5 * stride
        kpt_xy = kpt_pred[..., :2] * stride + grid_center
        kpt_vis = kpt_pred[..., 2:3]
        kpts = torch.cat([kpt_xy, kpt_vis], dim=-1)

        attr_pred = head_outs["attr"][lvl].permute(0, 2, 3, 1).reshape(B, N, 2)
        helmet_logit = attr_pred[..., 0]
        smoke_logit = attr_pred[..., 1]

        mask = (scores > score_thresh).any(dim=-1)
        for b in range(B):
            m = mask[b]
            if m.any():
                all_boxes.append(boxes[b][m].unsqueeze(0))
                all_scores.append(scores[b][m].unsqueeze(0))
                all_kpts.append(kpts[b][m].unsqueeze(0))
                all_helmet.append(helmet_logit[b][m].unsqueeze(0))
                all_smoke.append(smoke_logit[b][m].unsqueeze(0))

    if all_boxes:
        boxes_out = torch.cat(all_boxes, dim=1)
        scores_out = torch.cat(all_scores, dim=1)
        kpts_out = torch.cat(all_kpts, dim=1)
        helmet_out = torch.cat(all_helmet, dim=1)
        smoke_out = torch.cat(all_smoke, dim=1)
    else:
        boxes_out = torch.zeros(B, 0, 4, device=device)
        scores_out = torch.zeros(B, 0, 3, device=device)
        kpts_out = torch.zeros(B, 0, 17, 3, device=device)
        helmet_out = torch.zeros(B, 0, device=device)
        smoke_out = torch.zeros(B, 0, device=device)

    return boxes_out, scores_out, kpts_out, helmet_out, smoke_out

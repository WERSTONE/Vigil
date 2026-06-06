"""统一检测头: cls + bbox + kpts + helmet + smoking, 权重跨尺度共享."""

import torch
import torch.nn as nn
from models.common import Conv


def _make_tower(in_ch, num_tower, dropout):
    if num_tower == 0:
        return nn.Identity()
    layers = [Conv(in_ch, in_ch, 3)]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    for _ in range(num_tower - 1):
        layers.append(Conv(in_ch, in_ch, 3))
    return nn.Sequential(*layers)


class VigilHead(nn.Module):
    """FCOS-style 统一检测头.

    每个格点输出:
        cls:     [B, 4, H, W]   — bg/person/fire/water
        bbox:    [B, 4, H, W]   — ltrb offset
        obj:     [B, 1, H, W]   — centerness
        kpts:    [B, 51, H, W]  — 17关键点 × (dx, dy, vis)
        helmet:  [B, 1, H, W]   — 安全帽 logit
        smoking: [B, 1, H, W]   — 吸烟 logit
    """

    def __init__(self, in_ch, num_classes=4, num_kpts=17, num_tower=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_kpts = num_kpts

        self.cls_tower = _make_tower(in_ch, num_tower, dropout)
        self.reg_tower = _make_tower(in_ch, num_tower, dropout)
        self.attr_tower = _make_tower(in_ch, num_tower, dropout)

        self.cls_pred    = nn.Conv2d(in_ch, num_classes, 1)
        self.reg_pred    = nn.Conv2d(in_ch, 4, 1)
        self.obj_pred    = nn.Conv2d(in_ch, 1, 1)
        self.kpt_pred    = nn.Conv2d(in_ch, num_kpts * 3, 1)
        self.helmet_pred = nn.Conv2d(in_ch, 1, 1)
        self.smoke_pred  = nn.Conv2d(in_ch, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_pred.bias, -4.595)  # Focal Loss prior

    def forward(self, features):
        """features: List[[B, C, H, W]]  →  dict of List[Tensor] per level."""
        outs = {"cls": [], "bbox": [], "obj": [],
                "kpts": [], "helmet": [], "smoking": []}
        for f in features:
            outs["cls"].append(self.cls_pred(self.cls_tower(f)))

            reg_feat = self.reg_tower(f)
            outs["bbox"].append(self.reg_pred(reg_feat))
            outs["obj"].append(self.obj_pred(reg_feat))

            attr = self.attr_tower(f)
            outs["kpts"].append(self.kpt_pred(attr))
            outs["helmet"].append(self.helmet_pred(attr))
            outs["smoking"].append(self.smoke_pred(attr))
        return outs


# ── 解码 ──

def _make_grid(nx, ny, device):
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device), indexing="ij")
    return torch.stack((xv, yv), 2).float()


def decode_outputs(head_outs, strides, score_thresh=0.05):
    """多级 head 输出 → 检测候选.

    Returns:
        boxes:   [B, N, 4]  xyxy
        scores:  [B, N, 3]  person/fire/water
        kpts:    [B, N, 17, 3]  xyv
        helmet:  [B, N]  logits
        smoking: [B, N]  logits
    """
    B = head_outs["cls"][0].shape[0]
    device = head_outs["cls"][0].device

    all_boxes, all_scores = [], []
    all_kpts, all_helmet, all_smoke = [], [], []

    for lvl, stride in enumerate(strides):
        _, _, H, W = head_outs["cls"][lvl].shape
        N = H * W

        cls_t = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, N, 4)
        reg_t = head_outs["bbox"][lvl].permute(0, 2, 3, 1).reshape(B, N, 4)
        obj_t = head_outs["obj"][lvl].permute(0, 2, 3, 1).reshape(B, N, 1)

        grid = _make_grid(W, H, device).view(1, N, 2) * stride

        # ltrb → xyxy
        offsets = reg_t.exp() * stride
        l, t, r, b = offsets[..., 0], offsets[..., 1], offsets[..., 2], offsets[..., 3]
        boxes = torch.stack([
            grid[..., 0] - l, grid[..., 1] - t,
            grid[..., 0] + r, grid[..., 1] + b,
        ], dim=-1)

        # scores: cls_sigmoid * centerness, 丢弃 bg (class 0)
        scores = cls_t.sigmoid() * obj_t.sigmoid()   # [B, N, 4]
        scores = scores[:, :, 1:]                     # [B, N, 3]

        kpt_t  = head_outs["kpts"][lvl].permute(0, 2, 3, 1).reshape(B, N, 17, 3)
        helm_t = head_outs["helmet"][lvl].permute(0, 2, 3, 1).reshape(B, N)
        smok_t = head_outs["smoking"][lvl].permute(0, 2, 3, 1).reshape(B, N, 1)

        mask = (scores > score_thresh).any(dim=-1)  # [B, N]
        for b in range(B):
            m = mask[b]
            if m.any():
                all_boxes.append(boxes[b][m].unsqueeze(0))
                all_scores.append(scores[b][m].unsqueeze(0))
                all_kpts.append(kpt_t[b][m].unsqueeze(0))
                all_helmet.append(helm_t[b][m].unsqueeze(0))
                all_smoke.append(smok_t[b][m].unsqueeze(0))

    if all_boxes:
        boxes   = torch.cat(all_boxes, dim=1)
        scores  = torch.cat(all_scores, dim=1)
        kpts    = torch.cat(all_kpts, dim=1)
        helmet  = torch.cat(all_helmet, dim=1)
        smoking = torch.cat(all_smoke, dim=1).squeeze(-1)
    else:
        boxes   = torch.zeros(B, 0, 4, device=device)
        scores  = torch.zeros(B, 0, 3, device=device)
        kpts    = torch.zeros(B, 0, 17, 3, device=device)
        helmet  = torch.zeros(B, 0, device=device)
        smoking = torch.zeros(B, 0, device=device)

    return boxes, scores, kpts, helmet, smoking

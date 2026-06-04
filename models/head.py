"""统一检测头: cls + bbox + kpts + helmet + smoking, 权重跨尺度共享."""

import torch
import torch.nn as nn
from models.common import Conv


class VigilHead(nn.Module):
    """FCOS-style 统一检测头, 每个格点输出所有属性.

    输出 (per FPN level):
        cls:     [B, num_cls, H, W]   — 4 类 (bg/person/fire/water)
        bbox:    [B, 4, H, W]        — ltrb offset (正值)
        obj:     [B, 1, H, W]        — centerness
        kpts:    [B, 51, H, W]       — 17关键点 × (dx, dy, vis)
        helmet:  [B, 1, H, W]        — 戴安全帽 logit
        smoking: [B, 1, H, W]        — smoking logit
    """

    def __init__(self, in_ch, num_classes=4, num_kpts=17, num_tower=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_kpts = num_kpts

        # 分类塔
        cls_tower = [Conv(in_ch, in_ch, 3), nn.Dropout2d(dropout)] if dropout > 0 else []
        cls_tower += [Conv(in_ch, in_ch, 3) for _ in range(num_tower - 1)] if num_tower > 1 else []
        cls_tower += [Conv(in_ch, in_ch, 3)] if num_tower == 0 else []
        self.cls_tower = nn.Sequential(*cls_tower) if cls_tower else nn.Identity()
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        # 回归塔
        reg_tower = [Conv(in_ch, in_ch, 3), nn.Dropout2d(dropout)] if dropout > 0 else []
        reg_tower += [Conv(in_ch, in_ch, 3) for _ in range(num_tower - 1)] if num_tower > 1 else []
        reg_tower += [Conv(in_ch, in_ch, 3)] if num_tower == 0 else []
        self.reg_tower = nn.Sequential(*reg_tower) if reg_tower else nn.Identity()
        self.reg_pred = nn.Conv2d(in_ch, 4, 1)
        self.obj_pred = nn.Conv2d(in_ch, 1, 1)

        # 人体属性塔 (共享)
        attr_tower = [Conv(in_ch, in_ch, 3), nn.Dropout2d(dropout)] if dropout > 0 else []
        attr_tower += [Conv(in_ch, in_ch, 3) for _ in range(num_tower - 1)] if num_tower > 1 else []
        attr_tower += [Conv(in_ch, in_ch, 3)] if num_tower == 0 else []
        self.attr_tower = nn.Sequential(*attr_tower)
        self.kpt_pred = nn.Conv2d(in_ch, num_kpts * 3, 1)
        self.helmet_pred = nn.Conv2d(in_ch, 1, 1)
        self.smoke_pred = nn.Conv2d(in_ch, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Focal Loss prior: pi = 0.01 → bias = -4.595
        nn.init.constant_(self.cls_pred.bias, -4.595)

    def forward(self, features):
        """features: List[[B, C, H, W]] from neck.

        Returns dict with keys: cls, bbox, obj, kpts, helmet, smoking.
        Each value is a List[Tensor] per level.
        """
        outs = {"cls": [], "bbox": [], "obj": [],
                "kpts": [], "helmet": [], "smoking": []}
        for f in features:
            outs["cls"].append(self.cls_pred(self.cls_tower(f)))
            outs["bbox"].append(self.reg_pred(self.reg_tower(f)))
            outs["obj"].append(self.obj_pred(self.reg_tower(f)))

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
    """将多级 head 输出解码为检测结果.

    Args:
        head_outs: dict with cls/bbox/obj/kpts/helmet/smoking (each List[Tensor])
        strides: 各级 stride 列表
        score_thresh: 初始分数阈值

    Returns:
        boxes:    [B, N, 4]  xyxy
        scores:   [B, N]
        classes:  [B, N]  int (0=person, 1=fire, 2=water)
        kpts:     [B, N, 17, 3]  xyv
        helmet:   [B, N]     logits
        smoking:  [B, N]     logits
    """
    B = head_outs["cls"][0].shape[0]
    device = head_outs["cls"][0].device
    all_boxes, all_scores, all_cls = [], [], []
    all_kpts, all_helmet, all_smoke = [], [], []

    for lvl, stride in enumerate(strides):
        cls_t = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, -1, 4)       # [B, N, 4]
        reg_t = head_outs["bbox"][lvl].permute(0, 2, 3, 1).reshape(B, -1, 4)       # [B, N, 4]
        obj_t = head_outs["obj"][lvl].permute(0, 2, 3, 1).reshape(B, -1, 1)        # [B, N, 1]
        _, _, H, W = head_outs["cls"][lvl].shape

        # 网格坐标
        grid = _make_grid(W, H, device)               # [H, W, 2]
        locs = grid.reshape(1, -1, 2) * stride        # [1, N, 2]

        # ltrb offset → xyxy
        offsets = reg_t.exp() * stride                 # [B, N, 4]
        l, t, r, b = offsets[..., 0], offsets[..., 1], offsets[..., 2], offsets[..., 3]
        x1 = locs[..., 0] - l
        y1 = locs[..., 1] - t
        x2 = locs[..., 0] + r
        y2 = locs[..., 1] + b
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # 分数: cls_sigmoid * centerness
        scores = cls_t.sigmoid() * obj_t.sigmoid()     # [B, N, 4]
        # 跳过 bg (class 0), 只保留 person/fire/water
        scores = scores[:, :, 1:]                      # [B, N, 3]
        cls_idx = torch.arange(1, 4, device=device).reshape(1, 1, -1).expand(B, scores.shape[1], -1)

        # 过滤低分
        mask = scores > score_thresh

        # 平铺
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(cls_idx.expand(B, -1, -1))

        # 人体属性 (仅 person 类使用时有效)
        N_per_lvl = H * W
        kpt_t = head_outs["kpts"][lvl].permute(0, 2, 3, 1).reshape(B, N_per_lvl, 17, 3)
        helm_t = head_outs["helmet"][lvl].permute(0, 2, 3, 1).reshape(B, N_per_lvl)      # [B, N]
        smok_t = head_outs["smoking"][lvl].permute(0, 2, 3, 1).reshape(B, N_per_lvl, 1)

        all_kpts.append(kpt_t)
        all_helmet.append(helm_t)
        all_smoke.append(smok_t)

    boxes = torch.cat(all_boxes, dim=1)
    scores = torch.cat(all_scores, dim=1)
    classes = torch.cat(all_cls, dim=1)
    kpts = torch.cat(all_kpts, dim=1)
    helmet = torch.cat(all_helmet, dim=1)
    smoking = torch.cat(all_smoke, dim=1).squeeze(-1)

    return boxes, scores, classes, kpts, helmet, smoking

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass
from models.common import Conv


# ── 任务头 ──

class HumanAnalysisHead(nn.Module):
    """Person detect + attributes + keypoints. One anchor predicts [bbox(4)+person(1)+helmet(2)+smoking(1)+kpts(51)]."""
    def __init__(self, in_channels, num_keypoints=17, reg_max=16):
        super().__init__()
        attr_dim = 4  # person + helmet(on/off) + smoking
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, attr_dim, 1) for c in in_channels])
        self.reg_preds = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in in_channels])
        self.kpt_preds = nn.ModuleList([nn.Conv2d(c, num_keypoints * 3, 1) for c in in_channels])

    def forward(self, features):
        cls_outs, reg_outs, kpt_outs = [], [], []
        for f, cl, rl, kl in zip(features, self.cls_preds, self.reg_preds, self.kpt_preds):
            cls_outs.append(cl(f))
            reg_outs.append(rl(f))
            kpt_outs.append(kl(f))
        return cls_outs, reg_outs, kpt_outs


class SceneAnomalyHead(nn.Module):
    """Anomaly detect + mask coeffs. [bbox(4)+cls(2:fire/water)+mask(32)]."""
    def __init__(self, in_channels, num_classes=2, mask_dim=32, reg_max=16):
        super().__init__()
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for c in in_channels])
        self.reg_preds = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in in_channels])
        self.mask_preds = nn.ModuleList([nn.Conv2d(c, mask_dim, 1) for c in in_channels])

    def forward(self, features):
        cls_outs, reg_outs, mask_outs = [], [], []
        for f, cl, rl, ml in zip(features, self.cls_preds, self.reg_preds, self.mask_preds):
            cls_outs.append(cl(f))
            reg_outs.append(rl(f))
            mask_outs.append(ml(f))
        return cls_outs, reg_outs, mask_outs


class ProtoBranch(nn.Module):
    """YOLACT-style prototype masks. N3(80x80)->up->[32,160,160]."""
    def __init__(self, in_ch=64, proto_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            Conv(in_ch, in_ch, 3), Conv(in_ch, in_ch, 3), Conv(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, proto_dim, 1),
        )

    def forward(self, x):
        return F.interpolate(self.net(x), scale_factor=2.0, mode="bilinear", align_corners=False)


# ── 解码 ──

@dataclass
class DecodedPerson:
    bbox: List[float]
    confidence: float
    helmet_status: int          # 0=on, 1=off
    helmet_conf: float
    smoking_conf: float
    keypoints: List             # [17,3]


@dataclass
class DecodedAnomaly:
    bbox: List[float]
    class_id: int               # 0=fire, 1=water
    class_name: str
    confidence: float
    mask_coeffs: List[float]    # [32]


def _make_grid(nx, ny, device):
    yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
    return torch.stack((xv, yv), 2).float()


def _box_iou_batch(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter + 1e-16)


def decode_human_outputs(ha_cls, ha_reg, ha_kpt, input_shape, conf_threshold=0.25, iou_threshold=0.45, reg_max=16):
    all_bboxes, all_confs, all_helmets, all_smokings, all_kpts = [], [], [], [], []

    for cls_t, reg_t, kpt_t in zip(ha_cls, ha_reg, ha_kpt):
        B, _, H, W = cls_t.shape
        cls_t = cls_t.permute(0, 2, 3, 1)
        reg_t = reg_t.permute(0, 2, 3, 1)
        kpt_t = kpt_t.permute(0, 2, 3, 1)

        reg_t = reg_t.view(B, H, W, 4, reg_max).softmax(-1)
        reg_t = (reg_t @ torch.arange(reg_max, device=reg_t.device, dtype=reg_t.dtype))

        grid = _make_grid(W, H, cls_t.device)
        stride = input_shape[0] / W
        reg_t[..., :2] = (reg_t[..., :2] + grid) * stride
        reg_t[..., 2:4] = reg_t[..., 2:4] * stride * 2

        cx, cy, w, h = reg_t[..., 0], reg_t[..., 1], reg_t[..., 2], reg_t[..., 3]
        bboxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        kpt_flat = kpt_t.reshape(B, -1, 51)
        kpt_flat[..., 0::3] *= stride
        kpt_flat[..., 1::3] *= stride

        all_bboxes.append(bboxes.reshape(B, -1, 4))
        all_confs.append(cls_t[..., 0:1].reshape(B, -1).sigmoid())
        all_helmets.append(cls_t[..., 1:3].reshape(B, -1, 2))
        all_smokings.append(cls_t[..., 3:4].reshape(B, -1).sigmoid())
        all_kpts.append(kpt_flat)

    bboxes = torch.cat(all_bboxes, dim=1)[0]
    confs = torch.cat(all_confs, dim=1)[0]
    helmets = torch.cat(all_helmets, dim=1)[0]
    smokings = torch.cat(all_smokings, dim=1)[0]
    kpts = torch.cat(all_kpts, dim=1)[0]

    keep = confs > conf_threshold
    if not keep.any():
        return []

    bboxes, confs = bboxes[keep], confs[keep]
    helmets, smokings = helmets[keep], smokings[keep]
    kpts = kpts[keep]

    order = confs.argsort(descending=True)
    keep_indices = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep_indices.append(order.item())
            break
        idx = order[0].item()
        keep_indices.append(idx)
        ious = _box_iou_batch(bboxes[idx:idx + 1], bboxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]

    results = []
    for idx in keep_indices[:50]:
        k = kpts[idx].view(17, 3)
        results.append(DecodedPerson(
            bbox=bboxes[idx].clamp(0).tolist(),
            confidence=confs[idx].detach().item(),
            helmet_status=int(helmets[idx].argmax()),
            helmet_conf=helmets[idx].max().detach().item(),
            smoking_conf=smokings[idx].detach().item(),
            keypoints=k.tolist(),
        ))
    return results


def decode_anomaly_outputs(sa_cls, sa_reg, sa_mask, proto, input_shape, conf_threshold=0.15, iou_threshold=0.45):
    CLASS_NAMES = ["fire", "water"]
    all_bboxes, all_scores, all_classes, all_coeffs = [], [], [], []

    for cls_t, reg_t, mask_t in zip(sa_cls, sa_reg, sa_mask):
        B, _, H, W = cls_t.shape
        cls_t = cls_t.permute(0, 2, 3, 1)
        reg_t = reg_t.permute(0, 2, 3, 1)
        mask_t = mask_t.permute(0, 2, 3, 1)

        reg_t = reg_t.view(B, H, W, 4, 16).softmax(-1)
        reg_t = (reg_t @ torch.arange(16, device=reg_t.device, dtype=reg_t.dtype))

        grid = _make_grid(W, H, cls_t.device)
        strides = input_shape[0] / W
        reg_t[..., :2] = (reg_t[..., :2] + grid) * strides
        reg_t[..., 2:4] = reg_t[..., 2:4] * strides * 2

        cx, cy, w, h = reg_t[..., 0], reg_t[..., 1], reg_t[..., 2], reg_t[..., 3]
        bboxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        scores = cls_t.sigmoid()
        max_sc, max_cl = scores.max(dim=-1)

        all_bboxes.append(bboxes.reshape(B, -1, 4))
        all_scores.append(max_sc.reshape(B, -1))
        all_classes.append(max_cl.reshape(B, -1))
        all_coeffs.append(mask_t.reshape(B, -1, 32))

    bboxes = torch.cat(all_bboxes, dim=1)[0]
    scores = torch.cat(all_scores, dim=1)[0]
    classes = torch.cat(all_classes, dim=1)[0]
    coeffs = torch.cat(all_coeffs, dim=1)[0]

    keep = scores > conf_threshold
    if not keep.any():
        return []
    bboxes, scores, classes, coeffs = bboxes[keep], scores[keep], classes[keep], coeffs[keep]

    keep_final = []
    for cls_id in range(2):
        mask_c = classes == cls_id
        if not mask_c.any():
            continue
        b_c, s_c, idx_map = bboxes[mask_c], scores[mask_c], mask_c.nonzero().squeeze(-1)
        order = s_c.argsort(descending=True)
        while order.numel() > 0:
            if order.numel() == 1:
                keep_final.append(idx_map[order[0].item()].item())
                break
            idx = order[0].item()
            keep_final.append(idx_map[idx].item())
            ious = _box_iou_batch(b_c[idx:idx + 1], b_c[order[1:]])[0]
            order = order[1:][ious < iou_threshold]

    return [DecodedAnomaly(
        bbox=bboxes[i].clamp(0).tolist(),
        class_id=int(classes[i]),
        class_name=CLASS_NAMES[int(classes[i])],
        confidence=scores[i].detach().item(),
        mask_coeffs=coeffs[i].tolist(),
    ) for i in keep_final[:30]]

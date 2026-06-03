import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import math

from models.head import _make_grid
from models.assigner import TaskAlignedAssigner

KPT_SIGMAS = torch.tensor([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
    0.072, 0.062, 0.062, 1.007, 1.007, 0.087, 0.087, 0.089, 0.089,
])


# ── Focal Loss ──

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Sigmoid + Focal Loss, 自动处理正负样本极度不平衡。

    Args:
        logits: [*] 原始 logits (未经过 sigmoid)
        targets: [*] 0.0 或 1.0, 与 logits 同 shape
        alpha: 正样本权重系数 (负样本为 1-alpha)
        gamma: 聚焦参数, 越大越压制 easy samples
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


# ── IoU / CIoU ──

def bbox_iou(pred, target, xyxy=True, mode="ciou"):
    if not xyxy:
        pred = torch.cat([pred[..., :2] - pred[..., 2:4] / 2,
                          pred[..., :2] + pred[..., 2:4] / 2], dim=-1)
        target = torch.cat([target[..., :2] - target[..., 2:4] / 2,
                            target[..., :2] + target[..., 2:4] / 2], dim=-1)
    lt = torch.max(pred[..., :2], target[..., :2])
    rb = torch.min(pred[..., 2:], target[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    area_t = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    iou = inter / (area_p + area_t - inter + 1e-16)
    if mode == "iou":
        return iou

    lt_e = torch.min(pred[..., :2], target[..., :2])
    rb_e = torch.max(pred[..., 2:], target[..., 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    c2 = wh_e[..., 0] ** 2 + wh_e[..., 1] ** 2 + 1e-16
    cp = (pred[..., :2] + pred[..., 2:]) / 2
    ct = (target[..., :2] + target[..., 2:]) / 2
    d2 = (cp[..., 0] - ct[..., 0]) ** 2 + (cp[..., 1] - ct[..., 1]) ** 2
    if mode == "diou":
        return iou - d2 / c2

    w_p, h_p = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
    w_t, h_t = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
    v = (4 / (math.pi ** 2)) * ((torch.atan(w_t / (h_t + 1e-16)) -
                                  torch.atan(w_p / (h_p + 1e-16))) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-16)
    return iou - (d2 / c2 + v * alpha)


# ── 解码辅助 ──

def _decode_preds(cls_list, reg_list, input_size, reg_max=16):
    """将多尺度 Head 输出解码为扁平预测框和分数。"""
    all_boxes, all_scores = [], []
    for cls_t, reg_t in zip(cls_list, reg_list):
        _, _, H, W = cls_t.shape
        cls_t = cls_t.permute(0, 2, 3, 1)
        reg_t = reg_t.permute(0, 2, 3, 1)

        reg_t = reg_t.view(1, H, W, 4, reg_max).softmax(-1)
        reg_t = (reg_t @ torch.arange(reg_max, device=reg_t.device, dtype=reg_t.dtype))

        grid = _make_grid(W, H, cls_t.device)
        stride = input_size[0] / W
        reg_t[..., :2] = (reg_t[..., :2] + grid) * stride
        reg_t[..., 2:4] = reg_t[..., 2:4] * stride * 2

        cx, cy, w, h = reg_t[..., 0], reg_t[..., 1], reg_t[..., 2], reg_t[..., 3]
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        scores = cls_t[..., 0:1].reshape(1, -1).sigmoid()

        all_boxes.append(boxes.reshape(1, -1, 4))
        all_scores.append(scores)

    return (torch.cat(all_boxes, dim=1)[0],
            torch.cat(all_scores, dim=1)[0])


# ── OKS 关键点损失 ──

def _oks_loss(pred_kpts, gt_kpts, gt_boxes, fg_mask, matched_gt_idx, kpt_mask=None, eps=1e-8):
    """计算正样本上的 OKS 损失。pred_kpts: [N, 51], gt_kpts: [M, 17, 3]
    kpt_mask: [M] bool, 标记哪些 GT 框有有效关键点"""
    if fg_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_kpts.device)

    sigmas = KPT_SIGMAS.to(pred_kpts.device)
    p = pred_kpts[fg_mask].view(-1, 17, 3)                  # [P, 17, 3]
    gt_idx = matched_gt_idx[fg_mask]                         # [P]
    g = gt_kpts[gt_idx].to(pred_kpts.device)                # [P, 17, 3]

    # 只计算有有效关键点的正样本
    if kpt_mask is not None:
        valid = kpt_mask[gt_idx]                            # [P], cpu
        if not valid.any():
            return torch.tensor(0.0, device=pred_kpts.device)
        p, g, gt_idx = p[valid], g[valid], gt_idx[valid]

    gt_w = gt_boxes[gt_idx, 2] - gt_boxes[gt_idx, 0]
    gt_h = gt_boxes[gt_idx, 3] - gt_boxes[gt_idx, 1]
    area = (gt_w * gt_h).clamp(min=1).sqrt()                # [P']
    sigmas = sigmas.view(1, -1)                              # [1, 17]

    d2 = (p[..., :2] - g[..., :2]).pow(2).sum(dim=-1)       # [P', 17]
    k2 = (2 * sigmas) ** 2 * area.unsqueeze(-1) + eps       # [P', 17]
    oks = (d2 / (-2 * k2)).exp()

    visible = (g[..., 2] > 0).float()
    loss = 1 - (oks * visible).sum() / visible.sum().clamp(min=1)
    return loss


# ── BCE Loss ──

class HumanLoss(nn.Module):
    def __init__(self, box_w=7.5, cls_w=0.5, helmet_w=1.0, smoking_w=1.0, kpt_w=12.0):
        super().__init__()
        self.box_w, self.cls_w, self.helmet_w, self.smoking_w, self.kpt_w = (
            box_w, cls_w, helmet_w, smoking_w, kpt_w)
        self.assigner = TaskAlignedAssigner()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, ha_cls, ha_reg, ha_kpt, targets):
        """
        ha_cls:  List[[B,4,Hi,Wi]]  [person, helmet×2, smoking]
        ha_reg:  List[[B,64,Hi,Wi]]  bbox DFL
        ha_kpt:  List[[B,51,Hi,Wi]]  17 keypoints
        targets: {"boxes": [M,4], "helmet": [M]|None, "smoking": [M]|None,
                   "keypoints": [M,17,3]|None,
                   "loss_weights": {"box": float, "person": float, ...}}
        """
        device = ha_cls[0].device
        zero = torch.tensor(0.0, device=device)
        input_size = (ha_cls[0].shape[2] * 4, ha_cls[0].shape[3] * 4)
        lw = targets.get("loss_weights", {})

        losses = {"box": zero, "person": zero, "helmet": zero, "smoking": zero, "kpt": zero}

        gt_boxes = targets.get("boxes")
        if gt_boxes is None or len(gt_boxes) == 0:
            losses["total"] = zero
            return losses

        pred_boxes, pred_scores = _decode_preds(ha_cls, ha_reg, input_size)
        gt_boxes = gt_boxes.to(device)
        fg_mask, matched_gt_idx, target_boxes = self.assigner(pred_boxes, pred_scores, gt_boxes)

        # Box Loss (CIoU)
        if lw.get("box", 1.0) > 0 and fg_mask.any():
            ciou = bbox_iou(pred_boxes[fg_mask], target_boxes[fg_mask], xyxy=True, mode="ciou")
            losses["box"] = (1 - ciou).mean() * self.box_w

        # Person Loss (Focal, 全量位置)
        if lw.get("person", 1.0) > 0:
            all_scores = torch.cat(
                [c.permute(0, 2, 3, 1).reshape(-1, 4)[:, 0:1] for c in ha_cls], dim=0)
            person_targets = torch.zeros(len(all_scores), device=device)
            if fg_mask.any():
                person_targets[fg_mask] = 1.0
            losses["person"] = focal_loss(
                all_scores.squeeze(-1), person_targets,
                alpha=0.25, gamma=2.0) * self.cls_w

        # Helmet Loss (CE)
        if lw.get("helmet", 1.0) > 0 and fg_mask.any():
            gt_helmet = targets.get("helmet")
            if gt_helmet is not None:
                helmet_logits = torch.cat(
                    [c.permute(0, 2, 3, 1).reshape(-1, 4)[:, 1:3] for c in ha_cls], dim=0)
                gt_h = gt_helmet[matched_gt_idx[fg_mask]].to(device)
                losses["helmet"] = F.cross_entropy(
                    helmet_logits[fg_mask], gt_h.long()) * self.helmet_w

        # Smoking Loss (BCE)
        if lw.get("smoking", 1.0) > 0 and fg_mask.any():
            smoking_logits = torch.cat(
                [c.permute(0, 2, 3, 1).reshape(-1, 4)[:, 3:4] for c in ha_cls], dim=0)
            gt_smoking = targets.get("smoking")
            if gt_smoking is not None:
                gt_s = gt_smoking[matched_gt_idx[fg_mask]].float().to(device)
            else:
                gt_s = torch.zeros(fg_mask.sum(), device=device)
            losses["smoking"] = self.bce(
                smoking_logits[fg_mask].squeeze(-1), gt_s).mean() * self.smoking_w

        # Keypoint Loss (OKS)
        if lw.get("kpt", 1.0) > 0:
            if targets.get("keypoints") is not None and fg_mask.any():
                kpt_flat = torch.cat(
                    [k.permute(0, 2, 3, 1).reshape(1, -1, 51) for k in ha_kpt], dim=1)[0]
                losses["kpt"] = _oks_loss(
                    kpt_flat, targets["keypoints"], gt_boxes, fg_mask, matched_gt_idx,
                    kpt_mask=targets.get("kpt_mask")) * self.kpt_w

        losses["total"] = (
            losses["box"] + losses["person"] + losses["helmet"] +
            losses["smoking"] + losses["kpt"])
        return losses


class AnomalyLoss(nn.Module):
    def __init__(self, box_w=7.5, cls_w=1.5, mask_w=1.0):
        super().__init__()
        self.box_w, self.cls_w, self.mask_w = box_w, cls_w, mask_w
        self.assigner = TaskAlignedAssigner()

    def forward(self, sa_cls, sa_reg, sa_mask, proto, targets):
        """
        sa_cls:  List[[B,2,Hi,Wi]]  fire/water
        sa_reg:  List[[B,64,Hi,Wi]] bbox DFL
        sa_mask: List[[B,32,Hi,Wi]] mask coefficients
        proto:   [B,32,160,160]     prototype masks
        targets: {"boxes": [M,4]|None, "labels": [M]|None, "masks": [M,H,W]|None}
        """
        device = sa_cls[0].device
        zero = torch.tensor(0.0, device=device)
        input_size = (sa_cls[0].shape[2] * 8, sa_cls[0].shape[3] * 8)

        losses = {"box": zero, "cls": zero, "mask": zero}

        gt_boxes = targets.get("boxes")
        gt_labels = targets.get("labels")

        if gt_boxes is None or len(gt_boxes) == 0:
            # 无 GT 标注 → 不贡献 Anomaly 流损失
            losses["total"] = zero
            return losses

        gt_boxes = gt_boxes.to(device)
        gt_labels = gt_labels.to(device)

        # 解码预测 — 取每个位置最大得分类别作为 assign 用分数
        pred_boxes, _ = _decode_preds(
            [c[:, :1, :, :] for c in sa_cls], sa_reg, input_size)
        cls_scores = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(-1, 2) for c in sa_cls], dim=0)
        pred_scores = cls_scores.sigmoid().max(dim=-1).values

        fg_mask, matched_gt_idx, target_boxes = self.assigner(pred_boxes, pred_scores, gt_boxes)

        # Box Loss
        if fg_mask.any():
            ciou = bbox_iou(pred_boxes[fg_mask], target_boxes[fg_mask], xyxy=True, mode="ciou")
            losses["box"] = (1 - ciou).mean() * self.box_w

        # Classification Loss (Focal, 全量位置, fire/water 非互斥)
        cls_targets = torch.zeros_like(cls_scores)           # [N, 2]
        if fg_mask.any():
            tgt_labels = gt_labels[matched_gt_idx[fg_mask]]
            cls_targets[fg_mask, tgt_labels] = 1.0
        losses["cls"] = focal_loss(cls_scores, cls_targets,
                                   alpha=0.25, gamma=2.0) * self.cls_w

        # Mask Loss (BCE, 暂不启用 — 需 GT 掩码标注)
        gt_masks = targets.get("masks")
        if gt_masks is not None and fg_mask.any() and proto is not None:
            coeffs = torch.cat(
                [m.permute(0, 2, 3, 1).reshape(1, -1, 32) for m in sa_mask], dim=1)[0]
            pred_masks = (coeffs[fg_mask] @ proto[0].view(32, -1)).sigmoid()
            gt_m = gt_masks[matched_gt_idx[fg_mask]].view(pred_masks.shape[0], -1).to(device)
            losses["mask"] = self.bce(pred_masks, gt_m).mean() * self.mask_w

        losses["total"] = losses["box"] + losses["cls"] + losses["mask"]
        return losses


class UncertaintyWeighting(nn.Module):
    """Kendall et al. 2018: L = Σ(1/(2σ²)·L_i + log σ)"""
    def __init__(self, n_tasks=2, init_std=1.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.ones(n_tasks) * math.log(init_std ** 2))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_var[i])
            total += precision * loss + self.log_var[i]
        return total * 0.5


class VigilMultiTaskLoss(nn.Module):
    def __init__(self, human_loss=None, anomaly_loss=None, balance="manual",
                 h_w=1.0, a_w=0.5):
        super().__init__()
        self.human_loss = human_loss or HumanLoss()
        self.anomaly_loss = anomaly_loss or AnomalyLoss()
        self.balance = balance
        self.balancer = UncertaintyWeighting(2) if balance == "uncertainty" else None
        self.h_w, self.a_w = h_w, a_w

    def forward(self, outputs, targets):
        hl = self.human_loss(
            outputs["ha_cls"], outputs["ha_reg"], outputs["ha_kpt"],
            targets.get("human", {}))
        al = self.anomaly_loss(
            outputs["sa_cls"], outputs["sa_reg"], outputs["sa_mask"],
            outputs["proto"], targets.get("anomaly", {}))

        if self.balancer:
            total = self.balancer([hl["total"], al["total"]])
        else:
            total = self.h_w * hl["total"] + self.a_w * al["total"]

        return {"total": total,
                "human_total": hl["total"], "anomaly_total": al["total"],
                "human_box": hl["box"], "human_helmet": hl["helmet"],
                "human_smoking": hl["smoking"], "human_kpt": hl["kpt"],
                "anomaly_box": al["box"], "anomaly_cls": al["cls"],
                "anomaly_mask": al["mask"]}

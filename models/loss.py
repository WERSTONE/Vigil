"""
Vigil 多任务损失: HumanLoss + AnomalyLoss + UncertaintyWeighting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import math


def bbox_iou(pred, target, xyxy=False, mode="ciou"):
    if not xyxy:
        pred = torch.cat([pred[..., :2] - pred[..., 2:4] / 2, pred[..., :2] + pred[..., 2:4] / 2], dim=-1)
        target = torch.cat([target[..., :2] - target[..., 2:4] / 2, target[..., :2] + target[..., 2:4] / 2], dim=-1)
    lt = torch.max(pred[..., :2], target[..., :2])
    rb = torch.min(pred[..., 2:], target[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    area_t = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    iou = inter / (area_p + area_t - inter + 1e-16)
    if mode == "iou": return iou

    lt_e = torch.min(pred[..., :2], target[..., :2])
    rb_e = torch.max(pred[..., 2:], target[..., 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    c2 = wh_e[..., 0] ** 2 + wh_e[..., 1] ** 2 + 1e-16
    cp = (pred[..., :2] + pred[..., 2:]) / 2
    ct = (target[..., :2] + target[..., 2:]) / 2
    d2 = (cp[..., 0] - ct[..., 0]) ** 2 + (cp[..., 1] - ct[..., 1]) ** 2
    if mode == "diou": return iou - d2 / c2

    w_p, h_p = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
    w_t, h_t = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
    v = (4 / (math.pi ** 2)) * ((torch.atan(w_t / (h_t + 1e-16)) - torch.atan(w_p / (h_p + 1e-16))) ** 2)
    with torch.no_grad(): alpha = v / (1 - iou + v + 1e-16)
    return iou - (d2 / c2 + v * alpha)


class HumanLoss(nn.Module):
    """L = λ_box·CIoU + λ_cls·BCE(person) + λ_helm·CE(helmet) + λ_smok·BCE(smoking) + λ_kpt·OKS"""
    def __init__(self, box_w=7.5, cls_w=0.5, helmet_w=1.0, smoking_w=1.0, kpt_w=12.0):
        super().__init__()
        self.box_w, self.cls_w, self.helmet_w, self.smoking_w, self.kpt_w = box_w, cls_w, helmet_w, smoking_w, kpt_w

    def forward(self, ha_cls, ha_reg, ha_kpt, targets):
        device = ha_cls[0].device
        zero = torch.tensor(0.0, device=device)
        losses = {"box": zero, "person": zero, "helmet": zero, "smoking": zero, "kpt": zero}
        losses["total"] = (self.box_w * losses["box"] + self.cls_w * losses["person"] +
                           self.helmet_w * losses["helmet"] + self.smoking_w * losses["smoking"] +
                           self.kpt_w * losses["kpt"])
        return losses


class AnomalyLoss(nn.Module):
    """L = λ_box·CIoU + λ_cls·Focal(cls) + λ_mask·BCE(mask). Focal γ=2.0 for class imbalance."""
    def __init__(self, box_w=7.5, cls_w=1.5, mask_w=1.0, focal_gamma=2.0):
        super().__init__()
        self.box_w, self.cls_w, self.mask_w, self.focal_gamma = box_w, cls_w, mask_w, focal_gamma

    def forward(self, sa_cls, sa_reg, sa_mask, proto, targets):
        device = sa_cls[0].device
        zero = torch.tensor(0.0, device=device)
        losses = {"box": zero, "cls": zero, "mask": zero}
        losses["total"] = self.box_w * losses["box"] + self.cls_w * losses["cls"] + self.mask_w * losses["mask"]
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
    def __init__(self, human_loss=None, anomaly_loss=None, balance="manual"):
        super().__init__()
        self.human_loss = human_loss or HumanLoss()
        self.anomaly_loss = anomaly_loss or AnomalyLoss()
        self.balance = balance
        self.balancer = UncertaintyWeighting(2) if balance == "uncertainty" else None
        self.h_w, self.a_w = 1.0, 0.5

    def forward(self, outputs, targets):
        hl = self.human_loss(outputs["ha_cls"], outputs["ha_reg"], outputs["ha_kpt"], targets.get("human", {}))
        al = self.anomaly_loss(outputs["sa_cls"], outputs["sa_reg"], outputs["sa_mask"], outputs["proto"], targets.get("anomaly", {}))
        if self.balancer:
            total = self.balancer([hl["total"], al["total"]])
        else:
            total = self.h_w * hl["total"] + self.a_w * al["total"]
        return {"total": total, "human_total": hl["total"], "anomaly_total": al["total"],
                "human_box": hl["box"], "human_helmet": hl["helmet"], "human_smoking": hl["smoking"],
                "human_kpt": hl["kpt"], "anomaly_box": al["box"], "anomaly_cls": al["cls"], "anomaly_mask": al["mask"]}

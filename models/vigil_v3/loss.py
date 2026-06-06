"""统一损失 v3 — Varifocal cls + CIoU + DFL + OKS kpt + Focal BCE attr.

v3 改进:
  - cls 正样本 target 用 IoU 软化 (Varifocal/DEIM 风格), 低 IoU 匹配不被强制预测 1.0
  - w_helm/w_smoke 提高, 让属性 loss 在 total 中占可见比例
  - DFL 用 sqrt(IoU) 权重 (同 v2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

KPT_SIGMAS = torch.tensor([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
    0.072, 0.062, 0.062, 1.007, 1.007, 0.087, 0.087, 0.089, 0.089,
])


# ═══════════════════════════════════════════════════════════════
# IoU 工具
# ═══════════════════════════════════════════════════════════════

def _iou_xyxy(pred, target):
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_p + area_t - inter + 1e-16
    return inter / union, inter, union


# ═══════════════════════════════════════════════════════════════
# 分类损失 (Varifocal-style: 正样本 target = IoU)
# ═══════════════════════════════════════════════════════════════

def _varifocal_loss(pred_logits, target_score, alpha=0.75, gamma=2.0):
    """Varifocal Loss: 正样本 target 为 IoU (非 1.0), 负样本 target=0.

    DEIM/TOOD 风格: 低 IoU 匹配不给满 target 1.0, 避免模型被迫对低质匹配
    高置信, 从而缓解冷启动期的问题。

    Args:
        pred_logits: [N, C] logits
        target_score: [N, C] float, 正样本=IoU, 负样本=0
    """
    pred_score = pred_logits.sigmoid()
    weight = alpha * pred_score.pow(gamma) * (1 - target_score) + target_score
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none")
    return (weight * bce).sum()


# ═══════════════════════════════════════════════════════════════
# 框回归损失
# ═══════════════════════════════════════════════════════════════

def _ciou_loss(pred_xyxy, target_xyxy, eps=1e-7):
    iou, inter, union = _iou_xyxy(pred_xyxy, target_xyxy)

    px = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    py = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    tx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    ty = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    lt_e = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb_e = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    c2 = ((rb_e[:, 0] - lt_e[:, 0]) ** 2 +
          (rb_e[:, 1] - lt_e[:, 1]) ** 2).clamp(min=eps)

    pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=eps)
    ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=eps)
    tw = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=eps)
    th = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=eps)

    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return (1 - iou + rho2 / c2 + alpha * v).mean()


# ═══════════════════════════════════════════════════════════════
# DFL 分布损失
# ═══════════════════════════════════════════════════════════════

def _dfl_loss(pred_dist, target, weight=None, reg_max=16):
    target = target.clamp(0, reg_max - 1 - 1e-6)
    tl = target.long()
    tr = (tl + 1).clamp(0, reg_max - 1)
    wl = tr.float() - target
    wr = target - tl.float()
    loss = (F.cross_entropy(pred_dist, tl, reduction="none") * wl +
            F.cross_entropy(pred_dist, tr, reduction="none") * wr)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


# ═══════════════════════════════════════════════════════════════
# 属性损失
# ═══════════════════════════════════════════════════════════════

def _focal_bce(pred_logits, target, gamma=2.0, pos_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = torch.where(target > 0.5, pos_weight, 1.0)
    return (alpha_t * (1 - pt) ** gamma * bce).sum()


# ═══════════════════════════════════════════════════════════════
# 统一损失
# ═══════════════════════════════════════════════════════════════

class VigilLossV3(nn.Module):

    def __init__(self, w_box=3.0, w_cls=1.0, w_dfl=1.5,
                 w_kpt=20.0, w_helm=8.0, w_smoke=10.0,
                 reg_max=16, kpt_sigmas=None):
        super().__init__()
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_dfl = w_dfl
        self.w_kpt = w_kpt
        self.w_helm = w_helm
        self.w_smoke = w_smoke
        self.reg_max = reg_max
        self.register_buffer("sigmas",
            kpt_sigmas if kpt_sigmas is not None else KPT_SIGMAS)

    def forward(self, head_outs, assign_targets, strides, feat_sizes):
        device = head_outs["cls"][0].device
        B = head_outs["cls"][0].shape[0]

        loss_cls = torch.tensor(0.0, device=device)
        loss_ciou = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        loss_kpt = torch.tensor(0.0, device=device)
        loss_helm = torch.tensor(0.0, device=device)
        loss_smoke = torch.tensor(0.0, device=device)
        total_pos = 0
        total_person_pos = 0

        proj = torch.arange(self.reg_max, device=device, dtype=torch.float32)

        for lvl, (stride, (H, W)) in enumerate(zip(strides, feat_sizes)):
            targets = assign_targets[lvl]
            N_lvl = H * W

            cls_p_all = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, N_lvl, 3)
            cls_tgt_all = torch.zeros(B, N_lvl, 3, device=device)

            if targets is None:
                loss_cls += _varifocal_loss(
                    cls_p_all.reshape(-1, 3),
                    cls_tgt_all.reshape(-1, 3))
                continue

            N_pos = len(targets["gt_boxes"])
            total_pos += N_pos

            grid = targets["grid_xy"].to(device)
            gt_boxes = targets["gt_boxes"].to(device)
            gt_classes = targets["gt_classes"].to(device)
            batch_idx = targets["batch_idx"].to(device)

            gx, gy = grid[:, 0], grid[:, 1]

            reg_p = head_outs["reg"][lvl][batch_idx, :, gy, gx]
            kpt_p = head_outs["kpt"][lvl][batch_idx, :, gy, gx]
            attr_p = head_outs["attr"][lvl][batch_idx, :, gy, gx]

            # ── DFL 解码预测框 ──
            reg_rs = reg_p.view(N_pos, 4, self.reg_max)
            reg_probs = reg_rs.softmax(dim=-1)
            reg_delta = (reg_probs * proj.view(1, 1, self.reg_max)).sum(dim=-1)
            reg_delta = reg_delta * stride

            locs_x = (gx.float() + 0.5) * stride
            locs_y = (gy.float() + 0.5) * stride

            l, t = reg_delta[:, 0], reg_delta[:, 1]
            r, b = reg_delta[:, 2], reg_delta[:, 3]

            pred_xyxy = torch.stack([
                locs_x - l, locs_y - t,
                locs_x + r, locs_y + b,
            ], dim=-1)

            iou, _, _ = _iou_xyxy(pred_xyxy, gt_boxes)

            # ── 填充 cls target (Varifocal: 正样本 target=IoU) ──
            for j in range(N_pos):
                b_j = batch_idx[j].item()
                flat_j = gy[j].item() * W + gx[j].item()
                # Varifocal target=IoU, 但加 0.3 下限避免冷启动时正样本无梯度
                cls_tgt_all[b_j, flat_j, gt_classes[j].item()] = iou[j].detach().clamp(min=0.3)

            loss_cls += _varifocal_loss(
                cls_p_all.reshape(-1, 3),
                cls_tgt_all.reshape(-1, 3))

            # ── 框回归 (CIoU) ──
            loss_ciou += _ciou_loss(pred_xyxy, gt_boxes)

            # ── DFL ──
            gt_l_bin = ((locs_x - gt_boxes[:, 0]) / stride).clamp(0, self.reg_max - 1e-6)
            gt_t_bin = ((locs_y - gt_boxes[:, 1]) / stride).clamp(0, self.reg_max - 1e-6)
            gt_r_bin = ((gt_boxes[:, 2] - locs_x) / stride).clamp(0, self.reg_max - 1e-6)
            gt_b_bin = ((gt_boxes[:, 3] - locs_y) / stride).clamp(0, self.reg_max - 1e-6)
            gt_bins = torch.stack([gt_l_bin, gt_t_bin, gt_r_bin, gt_b_bin], dim=1)

            iou_d = iou.detach().clamp(min=0.01).sqrt()
            loss_dfl += _dfl_loss(
                reg_rs.reshape(-1, self.reg_max),
                gt_bins.reshape(-1),
                weight=iou_d.repeat_interleave(4),
                reg_max=self.reg_max,
            )

            # ── 人体属性 ──
            person_mask = gt_classes == 0
            if person_mask.any():
                n_person = person_mask.sum().item()
                total_person_pos += n_person
                p_idx = person_mask.nonzero(as_tuple=True)[0].to(device)
                p_boxes = gt_boxes[p_idx]
                p_locs = torch.stack([locs_x[p_idx], locs_y[p_idx]], dim=1)

                if targets["gt_kpts"] is not None:
                    gt_k_idx = targets["gt_kpts"].to(device)
                    pk = kpt_p[p_idx].view(-1, 17, 3)
                    pk_xy = pk[..., :2] * stride + p_locs.unsqueeze(1)
                    gk_xy = gt_k_idx[..., :2]

                    area = ((p_boxes[:, 2] - p_boxes[:, 0]) *
                            (p_boxes[:, 3] - p_boxes[:, 1])).clamp(min=1).sqrt()
                    sigmas = self.sigmas.view(1, 17).to(device)
                    d2 = (pk_xy - gk_xy).pow(2).sum(dim=-1)
                    k2 = (2 * sigmas) ** 2 * area.unsqueeze(-1) + 1e-8
                    oks = (d2 / (-2 * k2)).exp()
                    visible = (gt_k_idx[..., 2] > 0).float()
                    n_vis = visible.sum().clamp(min=1)
                    loss_kpt += (1 - (oks * visible).sum() / n_vis)

                if targets["gt_helmet"] is not None:
                    gt_h = targets["gt_helmet"].to(device).float()
                    loss_helm += _focal_bce(attr_p[p_idx, 0], 1 - gt_h, gamma=2.0, pos_weight=2.0)

                if targets["gt_smoking"] is not None:
                    gt_s = targets["gt_smoking"].to(device).float()
                    loss_smoke += _focal_bce(attr_p[p_idx, 1], gt_s, gamma=2.0, pos_weight=4.0)

        num_imgs = max(B, 1)

        return {
            "cls":    self.w_cls   * (loss_cls / max(total_pos, 1)),
            "ciou":   self.w_box   * loss_ciou / num_imgs,
            "dfl":    self.w_dfl   * loss_dfl / num_imgs,
            "kpt":    self.w_kpt   * (loss_kpt / max(total_person_pos, 1)),
            "helmet": self.w_helm  * (loss_helm / max(total_person_pos, 1)),
            "smoke":  self.w_smoke * (loss_smoke / max(total_person_pos, 1)),
            "num_pos": total_pos,
        }

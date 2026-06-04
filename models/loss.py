"""统一损失: Varifocal cls + WIoU v3 bbox + OKS kpt + BCE attributes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 关键点 OKS sigmas (COCO 17 kpts) ──
KPT_SIGMAS = torch.tensor([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
    0.072, 0.062, 0.062, 1.007, 1.007, 0.087, 0.087, 0.089, 0.089,
])


def _iou_xyxy(pred, target):
    """向量化 IoU, pred/target [N, 4] xyxy."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_p + area_t - inter + 1e-16
    return inter / union, inter, union


def _wiou_v3_loss(pred_xyxy, target_xyxy, delta=1.5):
    """WIoU v3: 动态非单调聚焦机制, 抑制低质量 anchor 梯度.

    Args:
        pred_xyxy: [N, 4]
        target_xyxy: [N, 4]
        delta: 异常度阈值 (default 1.5, 越小越激进)

    Reference: WIoU v3 (arXiv:2301.10051)
    """
    N = pred_xyxy.shape[0]
    device = pred_xyxy.device

    # 中心点距离 R_WIoU
    px = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    py = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    tx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    ty = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
    # 外接矩形宽高
    lt_e = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb_e = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    w_e, h_e = (rb_e[:, 0] - lt_e[:, 0]).clamp(min=1e-8), (rb_e[:, 1] - lt_e[:, 1]).clamp(min=1e-8)
    r_w = torch.exp(((px - tx) ** 2 + (py - ty) ** 2) / (w_e ** 2 + h_e ** 2 + 1e-8))

    # IoU
    iou, inter, union = _iou_xyxy(pred_xyxy, target_xyxy)
    liou = 1 - iou  # [N]

    # L_WIoU v1
    l_w1 = r_w * liou

    # 异常度 β = L*_IoU / mean(L*_IoU)
    with torch.no_grad():
        beta = liou / (liou.mean().clamp(min=1e-8) + 1e-8)

    # 非单调聚焦系数 r = β / (δ * α^(β-δ)), 取 α=1.9 (paper recommended)
    alpha = 1.9
    r = beta / (delta * (alpha ** (beta - delta)) + 1e-8)
    r = r.detach()

    return (r * l_w1).mean()


def _focal_bce(pred_logits, target, gamma=2.0, pos_weight=1.0):
    """Focal BCE: 聚焦难样本, pos_weight 提升少数类."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = torch.where(target > 0.5, pos_weight, 1.0)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def _varifocal_loss(pred, target, iou_target, alpha=0.75, gamma=2.0):
    """Varifocal Loss: IoU-aware 分类损失.

    Args:
        pred: [N, C] logits
        target: [N, C] 正样本 soft label (IoU 值), 负样本 0
        iou_target: [N, C] or None — 目标 IoU (仅正样本非零)
        alpha: 负样本衰减因子
        gamma: 聚焦参数
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)

    # 正样本: weight = target (IoU), 聚焦 = (1-pt)^gamma
    # 负样本: weight = alpha * pt^gamma (降低大量易分负样本的影响)
    pos_weight = target
    neg_weight = alpha * pt.pow(gamma)

    weight = torch.where(target > 0, pos_weight, neg_weight)
    return (weight * bce).mean()


class VigilLoss(nn.Module):
    """统一多任务损失 (WIoU v3 + Varifocal).

    Args:
        w_box, w_cls, w_obj, w_kpt, w_helm, w_smoke: 权重
    """

    def __init__(self, w_box=3.0, w_cls=1.0, w_obj=1.0,
                 w_kpt=5.0, w_helm=1.0, w_smoke=1.0,
                 kpt_sigmas=None):
        super().__init__()
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_obj = w_obj
        self.w_kpt = w_kpt
        self.w_helm = w_helm
        self.w_smoke = w_smoke
        self.register_buffer("sigmas",
            kpt_sigmas if kpt_sigmas is not None else KPT_SIGMAS)

    def forward(self, head_outs, assign_targets, strides, feat_sizes):
        device = head_outs["cls"][0].device
        B = head_outs["cls"][0].shape[0]

        loss_cls = torch.tensor(0.0, device=device)
        loss_bbox = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_kpt = torch.tensor(0.0, device=device)
        loss_helm = torch.tensor(0.0, device=device)
        loss_smoke = torch.tensor(0.0, device=device)
        total_pos = 0

        for lvl, (stride, (H, W)) in enumerate(zip(strides, feat_sizes)):
            targets = assign_targets[lvl]
            if targets is None:
                # 无正样本 → 仅计算负样本 obj loss
                obj_pred = head_outs["obj"][lvl].reshape(B, -1)
                for b in range(B):
                    loss_obj += F.binary_cross_entropy_with_logits(
                        obj_pred[b], torch.zeros(H * W, device=device))
                continue

            N_pos = len(targets["gt_boxes"])
            total_pos += N_pos

            grid = targets["grid_xy"].to(device)
            gt_boxes = targets["gt_boxes"].to(device)
            gt_classes = targets["gt_classes"].to(device)
            batch_idx = targets["batch_idx"].to(device)

            cls_lvl = head_outs["cls"][lvl]
            reg_lvl = head_outs["bbox"][lvl]
            obj_lvl = head_outs["obj"][lvl]
            kpt_lvl = head_outs["kpts"][lvl]
            helm_lvl = head_outs["helmet"][lvl]
            smok_lvl = head_outs["smoking"][lvl]

            gx, gy = grid[:, 0], grid[:, 1]

            cls_p = cls_lvl[batch_idx, :, gy, gx]                # [N_pos, 4]
            reg_p = reg_lvl[batch_idx, :, gy, gx]                # [N_pos, 4]
            obj_p = obj_lvl[batch_idx, 0, gy, gx]                # [N_pos]
            kpt_p = kpt_lvl[batch_idx, :, gy, gx]                # [N_pos, 51]
            helm_p = helm_lvl[batch_idx, 0, gy, gx]              # [N_pos]
            smok_p = smok_lvl[batch_idx, 0, gy, gx]              # [N_pos]

            # ── 解碼预测框 ──
            locs = (grid.float() + 0.5) * stride
            offsets = reg_p.exp() * stride
            l, t, r, b = offsets[:, 0], offsets[:, 1], offsets[:, 2], offsets[:, 3]
            pred_xyxy = torch.stack([
                locs[:, 0] - l, locs[:, 1] - t,
                locs[:, 0] + r, locs[:, 1] + b,
            ], dim=-1)

            # ── IoU 计算 (Varifocal 和 WIoU 共用) ──
            iou, _, _ = _iou_xyxy(pred_xyxy, gt_boxes)

            # ── 分类损失 (Varifocal) ──
            cls_target = torch.zeros(N_pos, 4, device=device)
            cls_target[range(N_pos), gt_classes + 1] = iou  # soft label = IoU
            loss_cls += _varifocal_loss(cls_p, cls_target, cls_target)

            # ── 回归损失 (WIoU v3) ──
            loss_bbox += _wiou_v3_loss(pred_xyxy, gt_boxes)

            # ── Objectness (BCE + IoU 软标签) ──
            loss_obj += F.binary_cross_entropy_with_logits(
                obj_p, iou.detach())

            # ── 人体属性 (仅 person) ──
            person_mask = gt_classes == 0
            if person_mask.any():
                n_person = person_mask.sum().item()
                p_idx = person_mask.nonzero(as_tuple=True)[0].to(device)  # 在 gt_boxes 中的索引
                p_kpt_idx = torch.arange(n_person)                         # 在 gt_kpts 中的索引 (person-only)
                p_boxes = gt_boxes[p_idx]

                # 关键点 (OKS)
                if targets["gt_kpts"] is not None:
                    gt_k = targets["gt_kpts"][p_kpt_idx].to(device)
                    pk = kpt_p[p_idx].view(-1, 17, 3)
                    plocs = locs[p_idx]
                    pk_xy = pk[..., :2] * stride + plocs.unsqueeze(1)
                    gk_xy = gt_k[..., :2]

                    area = ((p_boxes[:, 2] - p_boxes[:, 0]) *
                            (p_boxes[:, 3] - p_boxes[:, 1])).clamp(min=1).sqrt()
                    sigmas = self.sigmas.view(1, 17).to(device)
                    d2 = (pk_xy - gk_xy).pow(2).sum(dim=-1)
                    k2 = (2 * sigmas) ** 2 * area.unsqueeze(-1) + 1e-8
                    oks = (d2 / (-2 * k2)).exp()
                    visible = (gt_k[..., 2] > 0).float()
                    n_vis = visible.sum().clamp(min=1)
                    loss_kpt += (1 - (oks * visible).sum() / n_vis)

                # 头盔 (BCE: target=1 if helmet_on, target=0 if helmet_off)
                if targets["gt_helmet"] is not None:
                    gt_h = targets["gt_helmet"][p_kpt_idx].to(device).float()
                    loss_helm += _focal_bce(helm_p[p_idx], 1 - gt_h, gamma=2.0, pos_weight=1.5)

                # 吸烟 (Focal BCE, 少数类 up-weight)
                if targets["gt_smoking"] is not None:
                    gt_s = targets["gt_smoking"][p_kpt_idx].to(device).float()
                    loss_smoke += _focal_bce(smok_p[p_idx], gt_s, gamma=2.0, pos_weight=3.0)

        num_imgs = max(B, 1)
        return {
            "cls":    self.w_cls   * loss_cls / num_imgs,
            "bbox":   self.w_box   * loss_bbox / num_imgs,
            "obj":    self.w_obj   * loss_obj / num_imgs,
            "kpt":    self.w_kpt   * (loss_kpt / max(total_pos, 1)),
            "helmet": self.w_helm  * (loss_helm / max(total_pos, 1)),
            "smoke":  self.w_smoke * (loss_smoke / max(total_pos, 1)),
            "num_pos": total_pos,
        }

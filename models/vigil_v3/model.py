"""VigilModel v3 — 注意力驱动多任务检测器.

v3 vs v2 核心改进:
  - Area Attention backbone (YOLOv12), 前两阶段 CNN + 后两阶段 Attention
  - SE-enhanced Gather-Distribute neck, 通道数 128→256
  - 等容量 head: 所有分支 tower_depth=3, ch=256 (消除 kpt/attr 64ch 瓶颈)
  - Varifocal cls: 正样本 target=IoU 而非 1.0, 冷启动友好
  - TaskAlignedAssignerV3: beta 6.0→2.0, 可选 beta 预热
  - 损失权重重新平衡: w_helm=8, w_smoke=10 (属性梯度不再被淹没)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2

from models.base import VigilModelBase
from models.registry import register_model
from models.vigil_v3.backbone import AttentionCSPDarkNet
from models.vigil_v3.neck import AttentionGDNeck
from models.vigil_v3.head import VigilHeadV3, decode_outputs_v3, _dfl_decode, _make_grid
from models.vigil_v3.assigner import TaskAlignedAssignerV3
from models.vigil_v3.loss import VigilLossV3


class VigilModelV3(VigilModelBase, nn.Module):
    """v3 注意力多任务检测器.

    Args:
        backbone_w: backbone 宽度系数 (1.0=完整精度)
        neck_ch: neck 输出通道
        reg_max: DFL bin 数
        tower_depth: head 各分支 conv 深度
        w_*: 损失权重
        assigner_topk: TAL 每 GT 正样本数
        beta_warmup_epochs: assigner beta 预热 epoch 数
    """

    def __init__(self, backbone_w=1.0, neck_ch=256, reg_max=16,
                 tower_depth=3,
                 w_box=3.0, w_cls=1.0, w_dfl=1.5,
                 w_kpt=20.0, w_helm=8.0, w_smoke=10.0,
                 assigner_topk=13, beta=2.0, beta_warmup_epochs=5):
        super().__init__()
        self.backbone = AttentionCSPDarkNet(w=backbone_w)
        self.neck = AttentionGDNeck(
            in_channels=self.backbone.out_channels[1:4],
            out_ch=neck_ch,
        )
        self.head = VigilHeadV3(neck_ch, num_classes=3, reg_max=reg_max,
                                 tower_depth=tower_depth)
        self.strides = [8, 16, 32]
        self.reg_max = reg_max
        self._input_size = (640, 640)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.assigner = TaskAlignedAssignerV3(
            topk=assigner_topk, alpha=1.0, beta=beta,
            beta_warmup_epochs=beta_warmup_epochs)
        self.loss_fn = VigilLossV3(
            w_box=w_box, w_cls=w_cls, w_dfl=w_dfl,
            w_kpt=w_kpt, w_helm=w_helm, w_smoke=w_smoke,
            reg_max=reg_max)

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ═══════════════════════════════════════════════════════════
    # 训练
    # ═══════════════════════════════════════════════════════════

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.head(neck_feats)

    def compute_loss(self, sample):
        device = next(self.parameters()).device
        img = sample.image.unsqueeze(0).to(device)

        gt_boxes, gt_classes, attrs = self._build_targets(sample, device)

        head_outs = self.forward(img)
        feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]

        pred_scores, pred_boxes = self._decode_for_assigner(head_outs, feat_sizes)

        targets = self.assigner(
            pred_scores, pred_boxes,
            gt_boxes, gt_classes, attrs,
            feat_sizes, self.strides)

        losses = self.loss_fn(head_outs, targets, self.strides, feat_sizes)
        total = (losses["cls"] + losses["ciou"] + losses["dfl"] +
                 losses["kpt"] + losses["helmet"] + losses["smoke"])
        losses["total"] = total
        return losses

    def set_epoch(self, epoch):
        """设置当前 epoch (用于 assigner beta 预热)."""
        self.assigner.set_epoch(epoch)

    def _build_targets(self, sample, device):
        gt_boxes, gt_classes = [], []
        n_p = len(sample.person_boxes)
        if n_p > 0:
            gt_boxes.append(sample.person_boxes)
            gt_classes.append(torch.zeros(n_p, dtype=torch.long, device=device))
        if sample.detect_boxes.numel() > 0:
            gt_boxes.append(sample.detect_boxes)
            gt_classes.append(sample.detect_classes.to(device))
        if not gt_boxes:
            return (torch.empty(0, 4, device=device),
                    torch.empty(0, dtype=torch.long, device=device), {})
        all_boxes = torch.cat(gt_boxes, dim=0).to(device)
        all_classes = torch.cat(gt_classes, dim=0)
        attrs = {}
        if n_p > 0:
            if sample.person_kpts.numel() > 0:
                attrs["kpts"] = sample.person_kpts.to(device)
            if sample.person_helmet.numel() > 0:
                attrs["helmet"] = sample.person_helmet.to(device)
            if sample.person_smoke.numel() > 0:
                attrs["smoking"] = sample.person_smoke.to(device)
        return all_boxes, all_classes, attrs

    @torch.no_grad()
    def _decode_for_assigner(self, head_outs, feat_sizes):
        device = head_outs["cls"][0].device
        pred_scores, pred_boxes = [], []
        for lvl, ((H, W), stride) in enumerate(zip(feat_sizes, self.strides)):
            cls_pred = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(1, H * W, 3)
            scores = cls_pred.sigmoid()
            pred_scores.append(scores)
            grid = _make_grid(W, H, device) * stride
            boxes = _dfl_decode(head_outs["reg"][lvl], self.reg_max, stride, grid)
            pred_boxes.append(boxes)
        return pred_scores, pred_boxes

    # ═══════════════════════════════════════════════════════════
    # 推理
    # ═══════════════════════════════════════════════════════════

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        sx, sy = w / self._input_size[0], h / self._input_size[1]

        tensor = self._preprocess(frame).to(next(self.parameters()).device)
        raw = self.forward(tensor)
        det = self._decode(raw)

        for entry in det.values():
            entry["boxes"][:, 0] *= sx
            entry["boxes"][:, 2] *= sx
            entry["boxes"][:, 1] *= sy
            entry["boxes"][:, 3] *= sy
        return det

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0)

    def _decode(self, raw_outputs, score_thresh=0.05) -> dict:
        boxes, scores, kpts, helmet, smoking = decode_outputs_v3(
            raw_outputs, self.strides, self.reg_max, score_thresh)
        boxes, scores = boxes[0], scores[0]
        kpts, helmet, smoking = kpts[0], helmet[0], smoking[0]
        max_scores, best_cls = scores.max(dim=-1)

        result = {}
        for cls_idx, cls_name in [(0, "person"), (1, "fire"), (2, "water")]:
            mask = best_cls == cls_idx
            if mask.any():
                entry = {"boxes": boxes[mask], "scores": max_scores[mask]}
                if cls_name == "person":
                    entry["kpts"]    = kpts[mask]
                    entry["helmet"]  = helmet[mask]
                    entry["smoking"] = smoking[mask]
                result[cls_name] = entry
        return result


# ═══════════════════════════════════════════════════════════════
# 模型注册
# ═══════════════════════════════════════════════════════════════

@register_model("vigil_v3")
def create_model(pretrained=None, **kwargs):
    model = VigilModelV3(**kwargs)
    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    return model

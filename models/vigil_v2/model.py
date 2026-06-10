"""VigilModel v2: 解耦多任务检测器.

v2 相比 v1 的核心改进:
  - 解耦 head (cls/reg/kpt/attr 独立分支, YOLOv8-style)
  - DFL 分布框回归 (替代 exp(ltrb))
  - TaskAlignedAssigner (动态 top-k, 替代 FCOS center)
  - CIoU + DFL loss (替代 WIoU v3)
  - Gather-Distribute Neck (替代 FPN+PAN)
  - 3 尺度 [8,16,32] (去掉冗余 stride 4)
  - 直接 cls score (去掉 centerness)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2

from models.base import VigilModelBase
from models.registry import register_model
from models.vigil_v2.backbone import CSPDarkNetV2
from models.vigil_v2.neck import GatherDistributeNeck
from models.vigil_v2.head import VigilHeadV2, decode_outputs_v2, _dfl_decode, _make_grid
from models.vigil_v2.assigner import TaskAlignedAssigner
from models.vigil_v2.loss import VigilLossV2


class VigilModelV2(VigilModelBase, nn.Module):

    def __init__(self, backbone_w=0.75, neck_ch=160, reg_max=16,
                 w_box=5.0, w_cls=1.0, w_dfl=12.0,
                 w_kpt=10.0, w_helm=10.0, w_smoke=10.0,
                 assigner_topk=20):
        super().__init__()
        self.backbone = CSPDarkNetV2(w=backbone_w)
        self.neck = GatherDistributeNeck(
            in_channels=self.backbone.out_channels[1:4],  # p3, p4, p5
            out_ch=neck_ch,
        )
        self.head = VigilHeadV2(neck_ch, num_classes=3, reg_max=reg_max)
        self.strides = [8, 16, 32]
        self.reg_max = reg_max
        self._input_size = (640, 640)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.assigner = TaskAlignedAssigner(topk=assigner_topk)
        self.loss_fn = VigilLossV2(
            w_box=w_box, w_cls=w_cls, w_dfl=w_dfl,
            w_kpt=w_kpt, w_helm=w_helm, w_smoke=w_smoke,
            reg_max=reg_max)

    @property
    def input_size(self):
        return self._input_size

    # ── 训练用 forward ──

    def forward(self, x):
        feats = self.backbone(x)            # p2, p3, p4, p5
        neck_feats = self.neck(feats[1:])   # p3, p4, p5
        return self.head(neck_feats)

    # ── 训练接口: compute_loss ──

    def compute_loss(self, sample):
        device = next(self.parameters()).device
        img = sample.image.unsqueeze(0).to(device)

        gt_boxes, gt_classes, attrs = self._build_targets(sample, device)

        head_outs = self.forward(img)
        # Loss/assigner 在 fp32 下计算避免溢出
        head_outs = {k: [t.float() for t in v] for k, v in head_outs.items()}
        feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]

        # 解码 (用于 assigner 的 alignment 计算)
        pred_scores, pred_boxes = self._decode_for_assigner(
            head_outs, feat_sizes)

        targets = self.assigner(
            pred_scores, pred_boxes,
            gt_boxes, gt_classes, attrs,
            feat_sizes, self.strides)

        losses = self.loss_fn(head_outs, targets, self.strides, feat_sizes)
        total = (losses["cls"] + losses["ciou"] + losses["dfl"] +
                 losses["kpt"] + losses["helmet"] + losses["smoke"])
        losses["total"] = total
        return losses

    def _build_targets(self, sample, device):
        """构建 GT tensor (与 v1 相同)."""
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
        """解码所有格点的得分和框 (不设阈值), 供 assigner 使用."""
        device = head_outs["cls"][0].device
        pred_scores, pred_boxes = [], []

        B = head_outs["cls"][0].shape[0]
        for lvl, ((H, W), stride) in enumerate(zip(feat_sizes, self.strides)):
            cls_pred = head_outs["cls"][lvl].permute(0, 2, 3, 1).reshape(B, H * W, 3)
            scores = cls_pred.sigmoid()
            pred_scores.append(scores)

            grid = _make_grid(W, H, device) * stride
            boxes = _dfl_decode(head_outs["reg"][lvl], self.reg_max, stride, grid)
            pred_boxes.append(boxes)

        return pred_scores, pred_boxes

    # ── 推理接口: detect ──

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
            if "kpts" in entry:
                entry["kpts"][..., 0] *= sx
                entry["kpts"][..., 1] *= sy
        return det

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0)

    def _decode(self, raw_outputs, score_thresh=0.0) -> dict:
        boxes, scores, kpts, helmet, smoking = decode_outputs_v2(
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

    # ── 验证用预测 (640×640 空间, per-class 独立, 不做 argmax) ──

    @torch.no_grad()
    def predict_val(self, sample):
        """Run detection on a preprocessed VigilSample.

        Uses per-class independent thresholding (no argmax), so a single
        grid position can contribute to multiple classes.  Applies per-class
        NMS before returning.

        Returns:
            boxes:   [K, 4] xyxy in 640×640
            scores:  [K]
            classes: [K] 0=person, 1=fire, 2=water
        """
        device = next(self.parameters()).device
        img = sample.image.unsqueeze(0).to(device)
        head_outs = self.forward(img)

        boxes_3d, scores_3d, _, _, _ = decode_outputs_v2(
            head_outs, self.strides, self.reg_max, score_thresh=0.01)

        boxes = boxes_3d[0]    # [N, 4]
        scores = scores_3d[0]  # [N, 3]

        all_boxes, all_scores, all_cls = [], [], []
        for c in range(3):
            cls_scores = scores[:, c]
            keep = cls_scores > 0.01
            if keep.any():
                c_boxes = boxes[keep]
                c_scores = cls_scores[keep]
                nms_k = _nms(c_boxes, c_scores, 0.6)
                all_boxes.append(c_boxes[nms_k])
                all_scores.append(c_scores[nms_k])
                all_cls.append(torch.full((len(nms_k),), c, dtype=torch.long, device=device))

        if all_boxes:
            return (torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_cls))
        return (torch.zeros(0, 4, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_classes(self):
        return 3


def _nms(boxes, scores, iou_thresh):
    """向量化 NMS, boxes [N,4] xyxy."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i.item())
        box_i = boxes[i]
        rest = boxes[order[1:]]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        lt = torch.max(box_i[:2], rest[:, :2])
        rb = torch.min(box_i[2:], rest[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        iou = inter / (area_i + area_rest - inter + 1e-8)
        order = order[1:][iou <= iou_thresh]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


@register_model("vigil_v2")
def create_model(pretrained=None, **kwargs):
    model = VigilModelV2(**kwargs)

    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    return model

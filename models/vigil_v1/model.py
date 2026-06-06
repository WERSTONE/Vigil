"""VigilModel v1: Backbone + FPN-PAN Neck + 统一检测头."""

import torch
import torch.nn as nn
import numpy as np
import cv2

from models.base import VigilModelBase
from models.registry import register_model
from models.vigil_v1.backbone import CSPDarkNet
from models.vigil_v1.neck import FPNPANNeck
from models.vigil_v1.head import VigilHead, decode_outputs
from models.vigil_v1.assigner import CenterAssigner
from models.vigil_v1.loss import VigilLoss


class VigilModel(VigilModelBase, nn.Module):

    def __init__(self, backbone_w=0.5, neck_ch=128,
                 w_box=3.0, w_cls=1.0, w_obj=1.0,
                 w_kpt=5.0, w_helm=1.0, w_smoke=1.0):
        super().__init__()
        self.backbone = CSPDarkNet(w=backbone_w)
        self.neck = FPNPANNeck(
            in_channels=self.backbone.out_channels[1:5],
            out_ch=neck_ch,
        )
        self.head = VigilHead(neck_ch, num_classes=4)
        self.strides = [4, 8, 16, 32]
        self._input_size = (640, 640)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.assigner = CenterAssigner(self.strides)
        self.loss_fn = VigilLoss(
            w_box=w_box, w_cls=w_cls, w_obj=w_obj,
            w_kpt=w_kpt, w_helm=w_helm, w_smoke=w_smoke)

    @property
    def input_size(self):
        return self._input_size

    # ── 训练用 forward ──

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats[1:])
        return self.head(neck_feats)

    # ── 训练接口: compute_loss ← 训练器唯一调用点 ──

    def compute_loss(self, sample):
        device = next(self.parameters()).device
        img = sample.image.unsqueeze(0).to(device)

        gt_boxes, gt_classes, attrs = self._build_targets(sample, device)

        head_outs = self.forward(img)
        feat_sizes = [(t.shape[2], t.shape[3]) for t in head_outs["cls"]]

        targets = self.assigner(
            [gt_boxes], [gt_classes],
            [attrs] if attrs else [{}], feat_sizes)

        losses = self.loss_fn(head_outs, targets, self.strides, feat_sizes)
        total = (losses["cls"] + losses["bbox"] + losses["obj"] +
                 losses["kpt"] + losses["helmet"] + losses["smoke"])
        losses["total"] = total
        return losses

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
        return det

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0)

    def _decode(self, raw_outputs, score_thresh=0.05) -> dict:
        boxes, scores, kpts, helmet, smoking = decode_outputs(
            raw_outputs, self.strides, score_thresh)
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

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


@register_model("vigil_v1")
def create_model(pretrained=None, **kwargs):
    model = VigilModel(**kwargs)

    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    return model

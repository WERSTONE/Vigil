"""推理引擎 — V2 多任务 FCOS 模型."""

import torch
import numpy as np
import cv2
import time
from typing import List, Optional
from dataclasses import dataclass, field

from models.model import VigilModel, create_model
from models.head import decode_fcos_outputs, nms_per_class


@dataclass
class PersonResult:
    bbox: List[float]           # xyxy
    confidence: float
    helmet_status: int          # 0=on, 1=off, -1=unknown
    helmet_conf: float
    smoking_conf: float
    keypoints: List[List[float]]  # [17, 3]


@dataclass
class AnomalyResult:
    bbox: List[float]
    class_id: int               # 0=fire, 1=water
    class_name: str
    confidence: float


@dataclass
class InferenceResult:
    frame_id: int
    timestamp: float
    persons: List[PersonResult]
    anomalies: List[AnomalyResult]
    events: List[dict]
    latency_ms: float


class InferenceEngine:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model: VigilModel, config: dict, device=None):
        self.model = model
        self.device = device or config.get("inference", {}).get("device", "cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda().eval()
        else:
            self.model = self.model.cpu().eval()

        from postprocess.temporal import PostProcessor
        self.postprocessor = PostProcessor(config.get("postprocess", {}))

        self.input_size = tuple(config["model"]["input_size"])
        self.conf_person = config["inference"]["conf_threshold_person"]
        self.conf_anomaly = config["inference"]["conf_threshold_anomaly"]
        self.iou_threshold = config["inference"]["iou_threshold"]
        self.frame_count = 0
        self._warmup()

    def _warmup(self):
        dummy = torch.randn(1, 3, *self.input_size)
        if self.device == "cuda":
            dummy = dummy.cuda()
        with torch.no_grad():
            self.model(dummy)

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = np.transpose(img, (2, 0, 1))
        t = torch.from_numpy(img).unsqueeze(0)
        if self.device == "cuda":
            t = t.cuda()
        return t

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU matrix [N, M]."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        return inter / (area1[:, None] + area2 - inter + 1e-8)

    def _associate_to_persons(self, person_boxes, det_boxes, det_scores,
                               iou_thresh=0.1):
        """将辅助检测 (helmet/smoking) 关联到最近的人体框.

        Returns:
            per_person_scores: [num_persons] 每个 person 的最佳关联分数
        """
        n_p = len(person_boxes)
        if n_p == 0 or len(det_boxes) == 0:
            return torch.zeros(n_p, device=det_boxes.device)

        iou = self._box_iou(person_boxes, det_boxes)  # [P, D]
        best_iou, best_idx = iou.max(dim=1)            # [P]

        scores = torch.zeros(n_p, device=det_boxes.device)
        valid = best_iou > iou_thresh
        if valid.any():
            scores[valid] = det_scores[best_idx[valid]]
        return scores

    def infer(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        self.frame_count += 1
        tensor = self.preprocess(frame)

        with torch.no_grad():
            outputs = self.model(tensor)

        strides = outputs["strides"]

        # — 解码各头 —
        # Person
        p_boxes, p_scores = decode_fcos_outputs(
            *outputs["person"], strides, score_thresh=self.conf_person)
        p_boxes, p_scores = p_boxes[0], p_scores[0]  # squeeze batch
        if p_scores.numel() > 0:
            p_cls = p_scores.argmax(-1) if p_scores.dim() > 1 else \
                    torch.zeros(p_scores.shape[0], dtype=torch.long, device=p_scores.device)
            keep_boxes, keep_scores, _ = nms_per_class(
                p_boxes, p_scores.max(-1).values if p_scores.dim() > 1 else p_scores,
                p_cls, self.iou_threshold)
        else:
            keep_boxes, keep_scores = p_boxes[:0], p_scores[:0]

        # Keypoints — collect per-person
        kpt_per_person = []
        if len(keep_boxes) > 0:
            kpt_list = outputs["kpt"]
            # find which locations correspond to kept person detections
            # simplified: decode kpt at person box centers
            for box in keep_boxes:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                # find nearest FPN level
                best_kpt = None
                min_dist = float("inf")
                for lvl, (kpt_t, stride) in enumerate(zip(kpt_list, strides)):
                    kt = kpt_t[0].permute(1, 2, 0).reshape(-1, 51)  # [H*W, 51]
                    # approximate: pick center location
                    H, W = kpt_t.shape[2], kpt_t.shape[3]
                    grid_x = (cx / stride).long().clamp(0, W - 1)
                    grid_y = (cy / stride).long().clamp(0, H - 1)
                    idx = grid_y * W + grid_x
                    kt_val = kt[idx].view(17, 3)
                    kt_val[:, 0] *= stride
                    kt_val[:, 1] *= stride
                    dist = ((kt_val[5:11, :2].mean(0) -
                             torch.tensor([cx, cy])).pow(2).sum())
                    if dist < min_dist:
                        min_dist = dist
                        best_kpt = kt_val
                kpt_per_person.append(best_kpt.cpu().tolist() if best_kpt is not None else [])
        else:
            kpt_per_person = []

        # Helmet
        h_boxes, h_scores = decode_fcos_outputs(
            *outputs["helmet"], strides, score_thresh=0.15)
        h_boxes, h_scores = h_boxes[0], h_scores[0]
        if h_scores.numel() > 0:
            # 2 classes: helmet_on (0), helmet_off (1)
            helmet_cls = h_scores.argmax(-1)
            helmet_max = h_scores.max(-1).values
            hk, hs, hc = nms_per_class(h_boxes, helmet_max, helmet_cls, self.iou_threshold)
        else:
            hk, hs, hc = h_boxes[:0], h_scores[:0], torch.tensor([])

        # Smoking
        s_boxes, s_scores = decode_fcos_outputs(
            *outputs["smoking"], strides, score_thresh=0.1)
        s_boxes, s_scores = s_boxes[0], s_scores[0]
        if s_scores.numel() > 0:
            sk, ss, _ = nms_per_class(
                s_boxes, s_scores.squeeze(-1) if s_scores.dim() > 1 else s_scores,
                torch.zeros(len(s_boxes), dtype=torch.long, device=s_boxes.device),
                self.iou_threshold)
        else:
            sk, ss = s_boxes[:0], s_scores[:0]

        # Anomaly
        a_boxes, a_scores = decode_fcos_outputs(
            *outputs["anomaly"], strides, score_thresh=self.conf_anomaly)
        a_boxes, a_scores = a_boxes[0], a_scores[0]
        anomaly_results = []
        if a_scores.numel() > 0:
            CLASS_NAMES = ["fire", "water"]
            for cls_id in range(2):
                cls_mask = a_scores[:, cls_id] > self.conf_anomaly
                if not cls_mask.any():
                    continue
                a_c = a_boxes[cls_mask]
                a_s = a_scores[cls_mask, cls_id]
                a_cls = torch.full((len(a_c),), cls_id, dtype=torch.long, device=a_c.device)
                ak, as_, _ = nms_per_class(a_c, a_s, a_cls, self.iou_threshold)
                for i in range(len(ak)):
                    anomaly_results.append(AnomalyResult(
                        bbox=ak[i].clamp(0).tolist(),
                        class_id=cls_id,
                        class_name=CLASS_NAMES[cls_id],
                        confidence=as_[i].item(),
                    ))

        # — 关联 helmet/smoking 到 person —
        helmet_per_person = self._associate_to_persons(
            keep_boxes, hk, hs) if len(keep_boxes) > 0 and len(hk) > 0 else \
            torch.zeros(len(keep_boxes))
        helmet_cls_per_person = torch.full(
            (len(keep_boxes),), -1, dtype=torch.long)
        if len(keep_boxes) > 0 and len(hk) > 0:
            iou = self._box_iou(keep_boxes, hk)
            best_iou, best_idx = iou.max(dim=1)
            valid = best_iou > 0.1
            if valid.any() and len(hc) > 0:
                helmet_cls_per_person[valid] = hc[best_idx[valid]].long()

        smoking_per_person = self._associate_to_persons(
            keep_boxes, sk, ss) if len(keep_boxes) > 0 and len(sk) > 0 else \
            torch.zeros(len(keep_boxes))

        # — 构建 person 结果 —
        persons = []
        for i in range(len(keep_boxes)):
            h_stat = int(helmet_cls_per_person[i].item()) if i < len(helmet_cls_per_person) else -1
            persons.append(PersonResult(
                bbox=keep_boxes[i].clamp(0).tolist(),
                confidence=keep_scores[i].item() if keep_scores.dim() > 0 else 0.0,
                helmet_status=h_stat,
                helmet_conf=float(helmet_per_person[i].item()) if i < len(helmet_per_person) else 0.0,
                smoking_conf=float(smoking_per_person[i].item()) if i < len(smoking_per_person) else 0.0,
                keypoints=kpt_per_person[i] if i < len(kpt_per_person) else [],
            ))

        # — Rescale → 原始帧 —
        h, w = frame.shape[:2]
        scale_x, scale_y = w / self.input_size[0], h / self.input_size[1]
        for p in persons:
            p.bbox = [p.bbox[0] * scale_x, p.bbox[1] * scale_y,
                       p.bbox[2] * scale_x, p.bbox[3] * scale_y]
            for kp in p.keypoints:
                kp[0] *= scale_x
                kp[1] *= scale_y
        for a in anomaly_results:
            a.bbox = [a.bbox[0] * scale_x, a.bbox[1] * scale_y,
                       a.bbox[2] * scale_x, a.bbox[3] * scale_y]

        events = self.postprocessor.process_frame(persons, anomaly_results, h, w)
        latency = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            frame_id=self.frame_count, timestamp=time.time(),
            persons=persons, anomalies=anomaly_results,
            events=events, latency_ms=latency)


def create_inference_engine(model_path=None, config_path="config/config.yaml",
                            device="cpu", variant="n"):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = create_model(variant=variant, pretrained_path=model_path)
    return InferenceEngine(model, config, device=device)

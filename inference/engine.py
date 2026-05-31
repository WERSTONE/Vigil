"""
推理引擎 v2 — 双流架构
"""
import torch
import numpy as np
import cv2
import time
from typing import List
from dataclasses import dataclass

from models.model import (
    VigilMultiTaskModel, create_model,
    DecodedPerson, DecodedAnomaly,
    decode_human_outputs, decode_anomaly_outputs,
)


@dataclass
class InferenceResult:
    frame_id: int
    timestamp: float
    persons: List[DecodedPerson]
    anomalies: List[DecodedAnomaly]
    events: List[dict]
    latency_ms: float


class InferenceEngine:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model: VigilMultiTaskModel, config: dict, device=None):
        self.model = model
        self.device = device or config.get("inference", {}).get("device", "cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda().half()
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
        if self.device == "cuda": dummy = dummy.cuda().half()
        with torch.no_grad(): self.model(dummy)

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = np.transpose(img, (2, 0, 1))
        t = torch.from_numpy(img).unsqueeze(0)
        if self.device == "cuda": t = t.cuda().half()
        return t

    def infer(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        self.frame_count += 1
        tensor = self.preprocess(frame)
        with torch.no_grad():
            outputs = self.model(tensor)

        persons = decode_human_outputs(
            outputs["ha_cls"], outputs["ha_reg"], outputs["ha_kpt"],
            frame.shape[:2], self.conf_person, self.iou_threshold)

        anomalies = decode_anomaly_outputs(
            outputs["sa_cls"], outputs["sa_reg"], outputs["sa_mask"],
            outputs["proto"], frame.shape[:2], self.conf_anomaly, self.iou_threshold)

        h, w = frame.shape[:2]
        events = self.postprocessor.process_frame_v2(persons, anomalies, h, w)
        latency = (time.perf_counter() - t0) * 1000

        return InferenceResult(frame_id=self.frame_count, timestamp=time.time(),
                               persons=persons, anomalies=anomalies, events=events, latency_ms=latency)


def create_inference_engine(model_path=None, config_path="config/config.yaml", device="cpu", variant="n"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = create_model(variant=variant, pretrained_path=model_path)
    return InferenceEngine(model, config, device=device)

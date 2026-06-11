"""Register Ultralytics YOLOv8 detection for direct inference."""

from pathlib import Path

from models.registry import register_model
from models.ultralytics_adapter import UltralyticsVigilAdapter


DEFAULT_WEIGHTS = Path("checkpoints/yolov8/yolov8n.pt")


def _default_weights():
    return str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolov8n.pt"


@register_model("yolov8")
def create_model(pretrained=None, **kwargs):
    weights = pretrained or kwargs.pop("weights", _default_weights())
    return UltralyticsVigilAdapter(weights=weights, task="detect", **kwargs)

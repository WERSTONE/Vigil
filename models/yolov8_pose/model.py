"""Register Ultralytics YOLOv8-pose for direct inference."""

from pathlib import Path

from models.registry import register_model
from models.ultralytics_adapter import UltralyticsVigilAdapter


DEFAULT_WEIGHTS = Path("checkpoints/yolov8_pose/yolov8n-pose.pt")


def _default_weights():
    return str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolov8n-pose.pt"


@register_model("yolov8_pose")
@register_model("yolov8-pose")
def create_model(pretrained=None, **kwargs):
    weights = pretrained or kwargs.pop("weights", _default_weights())
    return UltralyticsVigilAdapter(weights=weights, task="pose", **kwargs)

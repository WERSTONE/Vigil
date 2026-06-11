from models.base import VigilModelBase
from models.registry import MODEL_REGISTRY, register_model, create_model, list_models

# 导入当前仓库内存在的模型模块以触发注册。
import models.vigil_v2.model  # noqa: F401
import models.yolov8.model  # noqa: F401
import models.yolov8_pose.model  # noqa: F401

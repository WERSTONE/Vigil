from models.base import VigilModelBase
from models.registry import MODEL_REGISTRY, register_model, create_model, list_models

# 导入模型模块触发注册
import models.vigil_v1.model  # noqa: F401 — 注册 "vigil_v1"
import models.vigil_v2.model  # noqa: F401 — 注册 "vigil_v2"
import models.vigil_v3.model  # noqa: F401 — 注册 "vigil_v3"

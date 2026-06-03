from models.backbone import CSPDarkNet
from models.neck import FPNPANNeck
from models.head import (
    HumanAnalysisHead, SceneAnomalyHead, ProtoBranch,
    DecodedPerson, DecodedAnomaly,
    decode_human_outputs, decode_anomaly_outputs,
)
import torch
import torch.nn as nn


class VigilMultiTaskModel(nn.Module):
    def __init__(self, backbone_w=0.25, backbone_d=0.33, num_keypoints=17,
                 anomaly_classes=2, mask_dim=32, reg_max=16):
        super().__init__()
        self.backbone = CSPDarkNet(w=backbone_w, d=backbone_d)
        bb_ch = self.backbone.feat_channels
        self.neck = FPNPANNeck(bb_ch, bb_ch)
        neck_ch = self.neck.out_channels
        self.ha_head = HumanAnalysisHead(neck_ch, num_keypoints, reg_max)
        self.sa_head = SceneAnomalyHead(neck_ch[1:], anomaly_classes, mask_dim, reg_max)
        self.proto_branch = ProtoBranch(neck_ch[1], mask_dim)

    def forward(self, x):
        neck_feats = self.neck(self.backbone(x))
        ha_cls, ha_reg, ha_kpt = self.ha_head(neck_feats)
        sa_cls, sa_reg, sa_mask = self.sa_head(neck_feats[1:])
        proto = self.proto_branch(neck_feats[1])
        return {"ha_cls": ha_cls, "ha_reg": ha_reg, "ha_kpt": ha_kpt,
                "sa_cls": sa_cls, "sa_reg": sa_reg, "sa_mask": sa_mask, "proto": proto}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(variant="n", pretrained_path=None):
    cfgs = {"n": (0.25, 0.33), "s": (0.50, 0.50)}
    w, d = cfgs.get(variant, cfgs["n"])
    model = VigilMultiTaskModel(backbone_w=w, backbone_d=d)
    if pretrained_path:
        _load_weights(model, pretrained_path)
    return model


def _load_weights(model, path):
    import warnings
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        warnings.warn(f"Checkpoint not found: {path}")
        return

    # Vigil 训练格式: {"model_state_dict": ..., "epoch": ...}
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[Vigil] Loaded full model from {path} (epoch {ckpt.get('epoch', '?')})")
        return

    if "model" in ckpt:
        ckpt = ckpt["model"]
    yolo_state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
    vig_state = model.state_dict()

    BB_MAP = [
        ("backbone.stem.0", "model.0"),
        ("backbone.stem.1", "model.1"),
        ("backbone.stage2",  "model.2"),
        ("backbone.down3",   "model.3"),
        ("backbone.stage3",  "model.4"),
        ("backbone.down4",   "model.5"),
        ("backbone.stage4",  "model.6"),
        ("backbone.down5",   "model.7"),
        ("backbone.stage5",  "model.8"),
        ("backbone.sppf",    "model.9"),
    ]

    loaded = 0
    for vig_prefix, yolo_prefix in BB_MAP:
        for vk in list(vig_state.keys()):
            if vk.startswith(vig_prefix):
                yk = vk.replace(vig_prefix, yolo_prefix)
                if yk in yolo_state and yolo_state[yk].shape == vig_state[vk].shape:
                    vig_state[vk] = yolo_state[yk]
                    loaded += 1

    model.load_state_dict(vig_state, strict=False)
    print(f"[Vigil] Loaded {loaded} backbone params from {path} ({len(BB_MAP)} layers)" if loaded else
          f"[Vigil] No weights loaded. Training from scratch.")


def export_onnx(model, path, size=(640, 640)):
    model.eval()
    dummy = torch.randn(1, 3, *size)
    output_names = []
    for prefix, n in [("ha_cls", 4), ("ha_reg", 4), ("ha_kpt", 4),
                       ("sa_cls", 3), ("sa_reg", 3), ("sa_mask", 3), ("proto", 1)]:
        output_names.extend([f"{prefix}_{i}" for i in range(n)])
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["input"], output_names=output_names,
                      dynamic_axes={**{"input": {0: "batch"}},
                                    **{n: {0: "batch"} for n in output_names}})
    print(f"Exported: {path}")


def export_torchscript(model, path, size=(640, 640)):
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, *size))
    traced.save(path)
    print(f"Exported: {path}")

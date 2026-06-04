"""VigilModel: Backbone + FPN-PAN Neck + 统一检测头."""

import torch
import torch.nn as nn
from models.backbone import CSPDarkNet
from models.neck import FPNPANNeck
from models.head import VigilHead


class VigilModel(nn.Module):
    """Vigil v3: 统一检测头 + 人体属性.

    - 共享 backbone + FPN-PAN neck (P2-P5)
    - 单检测头输出所有预测 (per-location):
        cls(4), bbox(4), obj(1), kpts(51), helmet(2), smoking(1)
    - 4 个检测尺度: P2(160×160), P3(80×80), P4(40×40), P5(20×20)
    """

    def __init__(self, backbone_w=0.5, neck_ch=128):
        super().__init__()
        self.backbone = CSPDarkNet(w=backbone_w)
        # backbone 输出 5 级 (P1-P5), neck 用 P2-P5
        self.neck = FPNPANNeck(
            in_channels=self.backbone.out_channels[1:5],
            out_ch=neck_ch,
        )
        self.head = VigilHead(neck_ch, num_classes=4)
        self.strides = [4, 8, 16, 32]  # P2-P5

    def forward(self, x):
        feats = self.backbone(x)           # [p1, p2, p3, p4, p5]
        neck_feats = self.neck(feats[1:])  # [n2, n3, n4, n5]
        return self.head(neck_feats)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(variant="n", pretrained=None):
    """工厂函数: 'n'=0.5x (~3M), 's'=1.0x (~8M)."""
    cfgs = {"n": 0.5, "s": 1.0}
    w = cfgs.get(variant, 0.5)
    model = VigilModel(backbone_w=w)

    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        # 允许部分加载 (只匹配 backbone)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[Vigil] Missing keys: {len(missing)} (head params)")
        print(f"[Vigil] Loaded {pretrained}")

    return model


def export_onnx(model, path, size=(640, 640)):
    model.eval()
    dummy = torch.randn(1, 3, *size)
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}})
    print(f"[Vigil] ONNX exported to {path}")


def export_torchscript(model, path, size=(640, 640)):
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, *size))
    traced.save(path)
    print(f"[Vigil] TorchScript exported to {path}")
